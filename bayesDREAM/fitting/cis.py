"""
Cis gene expression fitting for bayesDREAM.

This module contains the cis model and fitting logic.
"""

import gc
import os
import warnings
import numpy as np
import pandas as pd
import torch
import pyro
import pyro.distributions as dist
from pyro.distributions.transforms import iterated, affine_autoregressive
import pyro.optim as optim
import pyro.infer as infer



class CisFitter:
    """Handles cis gene expression fitting."""

    def __init__(self, model):
        """
        Initialize cis fitter.

        Parameters
        ----------
        model : _BayesDREAMCore
            The parent model instance
        """
        self.model = model

    def _t(self, x, dtype=torch.float32):
        return torch.as_tensor(x, dtype=dtype, device=self.model.device)

    def _to_cpu(self, x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu()
        return x

    ########################################################
    # Step 2: Fit cis effects (model_x)
    ########################################################
    def _model_x(
        self,
        N,
        G,
        guides_tensor,
        x_obs_tensor,
        sum_factor_tensor,
        mu_x_mean_tensor,
        mu_x_sd_tensor,
        sigma_eff_mean_tensor,
        sigma_eff_sd_tensor,
        C=None,
        groups_tensor=None,
        alpha_x_sample=None,
        target_per_guide_tensor=None,
        independent_mu_sigma=False,
        phi_x_precomputed=None,
        rate_alpha_precomputed=None,
        rate_beta_precomputed=None,
        scale_2_tensor=None,
    ):

        ###################################
        ## Technical covariate modelling ##
        ###################################
        if alpha_x_sample is not None:
            alpha_x = alpha_x_sample
        else:
            alpha_x = None
        
        ####################
        ## Overdispersion ##
        ####################
        if phi_x_precomputed is None:
            raise ValueError(
                "phi_x_precomputed is required. "
                "Run fit_ntc() before fit_cis() to estimate cis gene overdispersion from NTC cells."
            )
        phi_x_used = phi_x_precomputed

        ###############################
        ## Mixture model for x_eff_g ##
        ###############################
        if independent_mu_sigma:
            unique_targets = torch.unique(target_per_guide_tensor)
            mu_targets = {}
            sigma_targets = {}
            _hc_scale = scale_2_tensor if scale_2_tensor is not None else self._t(2.0)
            for t in unique_targets:
                mu_targets[int(t.item())] = pyro.sample(f"mu_target_{int(t.item())}", dist.Normal(mu_x_mean_tensor, mu_x_sd_tensor))
                sigma_targets[int(t.item())] = pyro.sample(f"sigma_target_{int(t.item())}", dist.HalfCauchy(scale=_hc_scale))
            # gather mu and sigma for each guide
            mu_target_tensor = torch.stack([mu_targets[int(t.item())] for t in unique_targets], dim=0)
            sigma_target_tensor = torch.stack([sigma_targets[int(t.item())] for t in unique_targets], dim=0)
            mu = mu_target_tensor[target_per_guide_tensor]
            sigma = sigma_target_tensor[target_per_guide_tensor]
        else:
            mu = pyro.sample("mu", dist.Normal(mu_x_mean_tensor, mu_x_sd_tensor))
            _hc_scale = scale_2_tensor if scale_2_tensor is not None else self._t(2.0)
            sigma = pyro.sample("sigma", dist.HalfCauchy(scale=_hc_scale))
            mu = mu.expand(G)
            sigma = sigma.expand(G)

        # Non-centered parameterization
        if rate_alpha_precomputed is not None:
            sigma_eff_alpha = pyro.sample("sigma_eff_alpha", dist.Exponential(rate_alpha_precomputed))
            sigma_eff_beta = pyro.sample("sigma_eff_beta", dist.Exponential(rate_beta_precomputed))
        elif (sigma_eff_mean_tensor >= 0.01) and (sigma_eff_sd_tensor >= 0.01):
            rate_alpha = (sigma_eff_sd_tensor ** 2) / (sigma_eff_mean_tensor ** 2)
            rate_beta = (sigma_eff_sd_tensor ** 2) / sigma_eff_mean_tensor
            sigma_eff_alpha = pyro.sample("sigma_eff_alpha", dist.Exponential(rate_alpha))
            sigma_eff_beta = pyro.sample("sigma_eff_beta", dist.Exponential(rate_beta))
        else:
            sigma_eff_alpha = pyro.sample("sigma_eff_alpha", dist.Gamma(self._t(1.0), self._t(0.01)))
            sigma_eff_beta = pyro.sample("sigma_eff_beta", dist.Gamma(self._t(1.0), self._t(0.01)))
        with pyro.plate("guides_plate", G):
            eps_x_eff_g = pyro.sample("eps_x_eff_g", dist.StudentT(df=self._t(3.0), loc=self._t(0.0), scale=self._t(1.0)))
            log2_x_eff_g = mu + sigma * eps_x_eff_g
            x_eff_g = pyro.deterministic("x_eff_g", 2.0 ** log2_x_eff_g)
        
            sigma_eff = pyro.sample("sigma_eff", dist.Gamma(sigma_eff_alpha, sigma_eff_beta))
                
        ##########################
        ## Cell-level variables ##
        ##########################
        if alpha_x is not None:
            # alpha_x shape [C]: already includes reference group (index 0 = 1.0)
            # or shape [C-1]: needs reference group prepended
            if alpha_x.shape[-1] == C:
                alpha_x_full = alpha_x
            else:
                ones_ = torch.ones(1, device=self.model.device)
                alpha_x_full = torch.cat([ones_, alpha_x], dim=-1)  # shape = [C]
            alpha_x_used = alpha_x_full[groups_tensor]  # shape = [N]
        else:
            alpha_x_used = torch.ones_like(sum_factor_tensor)  # shape = [N] or [S,N]

        # Aggregate guide effects to cells
        if self.model.is_high_moi:
            # High MOI: effects are additive in log2FC space.
            # log2FC for guide g = log2(x_eff_g[g]) - log2(NTC_mean)
            # For a cell with guides A, B: log2FC_cell = log2FC_A + log2FC_B
            # => log2(x_mean) = log2(NTC) + log2FC_cell
            #                 = sum(log2(x_eff_g)) - (n_guides - 1) * log2(NTC)
            # This is always finite since x_eff_g > 0 and NTC_mean > 0.

            # NTC reference: weighted mean of NTC guide effects
            # Use pre-computed mask (set in fit_cis before SVI loop to avoid pandas access here)
            ntc_mask = self.model._ntc_guide_mask
            cells_per_guide = self.model.guide_assignment_tensor.sum(dim=0)  # [G]
            weights = cells_per_guide / sigma_eff.clamp(min=1e-6)            # [G]
            ntc_weights = weights[ntc_mask]
            ntc_effects  = x_eff_g[ntc_mask]
            weighted_mean_NTC = (ntc_weights * ntc_effects).sum(dim=-1) / ntc_weights.sum()
            weighted_mean_NTC = pyro.deterministic("weighted_mean_NTC", weighted_mean_NTC)

            log2_x_eff_g  = torch.log2(x_eff_g.clamp(min=1e-12))                               # [G]
            log2_NTC      = torch.log2(weighted_mean_NTC.clamp(min=1e-12))                       # scalar
            guides_per_cell = self.model.guide_assignment_tensor.sum(dim=1).clamp(min=1)         # [N]
            sum_log2_effects = torch.matmul(self.model.guide_assignment_tensor, log2_x_eff_g)    # [N]
            log2_x_mean   = sum_log2_effects - (guides_per_cell - 1) * log2_NTC                 # [N]
            x_mean        = 2.0 ** log2_x_mean                                                   # [N]
            sigma_mean    = torch.matmul(self.model.guide_assignment_tensor, sigma_eff) / guides_per_cell  # [N]
        else:
            # Single guide per cell: use indexing (unchanged)
            x_mean = x_eff_g[..., guides_tensor]  # [N]
            sigma_mean = sigma_eff[..., guides_tensor]  # [N]

        ######################
        ## Cell-level plate ##
        ######################
        with pyro.plate("data_plate", N):
            log_x_true = pyro.sample( # use log2 of xtrue to allow small values of xtrue
                "log_x_true",
                # validate_args=False: Predictive()'s internal _guess_max_plate_nesting
                # probe (pyro/infer/predictive.py) traces this model UNCONDITIONED, straight
                # from the priors, purely to count plate dims -- the sampled values are
                # discarded, never reaching posterior_samples_x. eps_x_eff_g's StudentT(df=3)
                # and sigma/sigma_target's HalfCauchy priors are heavy-tailed with no upper
                # clamp on 2.0**log2_x_eff_g (line ~124), so that throwaway trace can
                # occasionally draw values extreme enough to overflow float32, producing
                # inf/nan here and crashing the ENTIRE fit_cis() call on an assertion over
                # values nobody uses. Same rationale as x_obs's NegativeBinomial below.
                dist.Normal(torch.log2(x_mean), sigma_mean, validate_args=False)
            )
            x_true = pyro.deterministic("x_true", 2.0 ** log_x_true)
            mu_obs = alpha_x_used * x_true * sum_factor_tensor
            pyro.sample(
                "x_obs",
                dist.NegativeBinomial(total_count=phi_x_used,
                                      logits=torch.log(mu_obs) - torch.log(phi_x_used),
                                      validate_args=False
                                     ), # important to use logits to allow small x_true
                obs=x_obs_tensor
            )

    def _model_x_marginal(self, x_obs, log_offset, guides, assign, ntc_mask, G, phi_ntc,
                          m0, z, log_w, refit_phi, group=None, n_groups=0, pool_scale=None):
        """
        Cis model with the per-cell latent integrated out.

        Guide levels, single guide (l_g = guide log2 level):
            no pooling:   l_g ~ N(m0, 3)
            pooling:      m_k ~ N(m0, 3) per target k,  l_g ~ N(m_k(g), tau)
        High MOI (l_i = b + sum_g A_ig d_g, d_g = 0 for NTC guides):
            b ~ N(m0, 3);  targeting guides d_g ~ N(0, 3), or with pooling
            m_k ~ N(0, 3),  d_g ~ N(m_k(g), tau)
        tau ~ HalfNormal(1) if pool_scale == 'learn', else tau = pool_scale (fixed).
        Cells:  log2 x_i ~ N(l, s),  s ~ HalfNormal(1)  (shared cell-level SD)
                y_i ~ NB(phi, alpha_i sf_i x_i)
                phi = phi_ntc, or phi ~ LogNormal(log phi_ntc, 1) if refit_phi
        log2 x_i is integrated out with Gauss-Hermite nodes z (weights exp(log_w)).
        """
        dev = x_obs.device
        if refit_phi:
            phi = pyro.sample("phi_x", dist.LogNormal(torch.log(phi_ntc), 1.0))
        else:
            phi = phi_ntc
        s = pyro.sample("sigma_x_cell", dist.HalfNormal(torch.ones((), device=dev)))

        centre = m0 if assign is None else torch.zeros((), device=dev)
        if pool_scale is None:
            loc, scale = centre, 3.0
        else:
            with pyro.plate("targets_plate", n_groups):
                m_k = pyro.sample("log2_x_target", dist.Normal(centre, 3.0))
            if pool_scale == 'learn':
                scale = pyro.sample("tau_guide", dist.HalfNormal(torch.ones((), device=dev)))
            else:
                scale = float(pool_scale)
            loc = m_k[group]

        if assign is None:
            with pyro.plate("guides_plate", G):
                l_g = pyro.sample("log2_x_eff_g", dist.Normal(loc, scale))
            l_cell = l_g[guides]
        else:
            b = pyro.sample("log2_x_ntc", dist.Normal(m0, 3.0))
            n_t = int((~ntc_mask).sum())
            with pyro.plate("targeting_guides_plate", n_t):
                d_t = pyro.sample("delta_g", dist.Normal(loc, scale))
            d = torch.zeros(G, device=dev).index_put((~ntc_mask).nonzero(as_tuple=True), d_t)
            l_cell = b + assign @ d
        log_mu = (l_cell.unsqueeze(-1) + s * z) * np.log(2.0) + log_offset.unsqueeze(-1)  # [N, K]
        lp = dist.NegativeBinomial(total_count=phi, logits=log_mu - torch.log(phi),
                                   validate_args=False).log_prob(x_obs.unsqueeze(-1))
        with pyro.plate("data_plate", x_obs.shape[0]):
            pyro.factor("x_obs", torch.logsumexp(lp + log_w, dim=-1))

    def _guide_target_labels(self, G):
        """One target label per guide (in guide_code / guide_assignment column order), or None."""
        _ntc_variants = {'ntc', 'NTC', 'non-targeting', 'non-targeting-control', 'Non-Targeting'}
        if self.model.is_high_moi:
            if 'target' in self.model.guide_meta.columns:
                labels = self.model.guide_meta['target'].tolist()
            elif getattr(self.model, 'guide_targets_dict', None):
                labels = []
                for guide in self.model.guide_meta['guide']:
                    targets = self.model.guide_targets_dict.get(guide, [])
                    labels.append(self.model.cis_gene if self.model.cis_gene in targets
                                  else ('ntc' if any(t in _ntc_variants for t in targets)
                                        else (targets[0] if targets else 'ntc')))
            else:
                return None
        else:
            if 'target' not in self.model.meta.columns:
                return None
            labels = self.model.meta.groupby('guide_code')['target'].first().reindex(range(G)).tolist()
        return ['ntc' if t in _ntc_variants else t for t in labels]

    def _fit_cis_marginal(self, N, G, guides_tensor, x_obs_tensor, log_offset, guide_means,
                          phi_ntc, lr, niters, nsamples, n_quadrature, refit_phi, tolerance,
                          pool_scale='learn', n_mean_field=2000, window=500, n_grid=64):
        dev = self.model.device
        guide_means = guide_means.to(dtype=torch.float32)  # float64 in high-MOI mode
        z_np, w_np = np.polynomial.hermite_e.hermegauss(n_quadrature)
        z = self._t(z_np)
        log_w = self._t(np.log(w_np / np.sqrt(2 * np.pi)))

        if self.model.is_high_moi:
            assign = self.model.guide_assignment_tensor.to(device=dev, dtype=torch.float32)
            ntc_mask = self.model._ntc_guide_mask
            if not ntc_mask.any():
                raise ValueError("cis_model='marginal' in high MOI mode needs at least one NTC guide.")
            b0 = guide_means[ntc_mask].mean()
            lev0 = guide_means[~ntc_mask] - b0                     # levels of modelled guides
            init = {"log2_x_ntc": b0, "delta_g": lev0}
        else:
            assign, ntc_mask = None, None
            b0 = guide_means.median()
            lev0 = guide_means.clone()
            init = {"log2_x_eff_g": lev0}
        init.update(sigma_x_cell=self._t(0.3), phi_x=phi_ntc, tau_guide=self._t(0.5))

        # Group the modelled guides by target for partial pooling
        group, n_groups = None, 0
        if pool_scale is not None:
            labels = self._guide_target_labels(G)
            if labels is None:
                warnings.warn("No guide targets found; fitting cis_model='marginal' without pooling.")
                pool_scale = None
            else:
                if ntc_mask is not None:
                    labels = [t for t, is_ntc in zip(labels, ntc_mask.tolist()) if not is_ntc]
                codes, uniq = pd.factorize(pd.Series(labels, dtype=object))
                group = torch.tensor(codes, dtype=torch.long, device=dev)
                n_groups = len(uniq)
                init["log2_x_target"] = torch.stack(
                    [lev0[group == k].mean() for k in range(n_groups)])
        args = (x_obs_tensor, log_offset, guides_tensor, assign, ntc_mask, G, phi_ntc,
                b0, z, log_w, refit_phi, group, n_groups, pool_scale)

        # Mean-field warm start, then a full-rank Gaussian over all (2G or so) parameters.
        pyro.clear_param_store()
        init_fn = pyro.infer.autoguide.initialization.init_to_value(values=init)
        guide = pyro.infer.autoguide.AutoNormal(self._model_x_marginal, init_loc_fn=init_fn, init_scale=0.05)
        losses = []
        phases = [(guide, min(n_mean_field, niters), lr), (None, niters, lr / 2)]
        for guide_p, max_it, lr_p in phases:
            if guide_p is None:
                med = {k: v.detach() for k, v in guide.median(*args).items()}
                pyro.clear_param_store()
                guide = pyro.infer.autoguide.AutoMultivariateNormal(
                    self._model_x_marginal,
                    init_loc_fn=pyro.infer.autoguide.initialization.init_to_value(values=med),
                    init_scale=0.02)
            svi = pyro.infer.SVI(self._model_x_marginal, guide, pyro.optim.Adam({"lr": lr_p}),
                                 loss=pyro.infer.Trace_ELBO())
            phase_losses = []
            w = window if guide_p is None else window // 2
            for step in range(max_it):
                phase_losses.append(svi.step(*args))
                if (step + 1) % w == 0 and step + 1 >= 2 * w:
                    prev = np.mean(phase_losses[-2 * w:-w])
                    cur = np.mean(phase_losses[-w:])
                    if (prev - cur) / abs(prev) < tolerance:
                        break
            losses += phase_losses
            print(f"[INFO] fit_cis (marginal): {'full-rank' if guide_p is None else 'mean-field'} "
                  f"phase stopped after {len(phase_losses)} steps, loss {np.mean(phase_losses[-w:]):.5e}")

        # Posterior draws of guide-level parameters
        with torch.no_grad():
            draws = [guide(*args) for _ in range(nsamples)]
        s_s = torch.stack([d["sigma_x_cell"] for d in draws])                       # [S]
        phi_s = (torch.stack([d["phi_x"] for d in draws]) if refit_phi
                 else phi_ntc.expand(nsamples))                                       # [S]
        if assign is None:
            l_g = torch.stack([d["log2_x_eff_g"] for d in draws])                    # [S, G]
            l_cell_s = l_g[:, guides_tensor]                                         # [S, N]
        else:
            b_s = torch.stack([d["log2_x_ntc"] for d in draws])                      # [S]
            d_full = torch.zeros(nsamples, G, device=dev)
            d_full[:, ~ntc_mask] = torch.stack([d["delta_g"] for d in draws])
            l_g = b_s.unsqueeze(-1) + d_full
            l_cell_s = b_s.unsqueeze(-1) + d_full @ assign.T

        # Per-cell log2 x: exact conditional posterior given each draw, on a z grid.
        zg = torch.linspace(-5.0, 5.0, n_grid, device=dev)
        dz = float(zg[1] - zg[0])
        log_x = torch.empty(nsamples, N)
        chunk = max(1, int(2e7 // (N * n_grid)))
        with torch.no_grad():
            for i in range(0, nsamples, chunk):
                lc, sc, ph = l_cell_s[i:i + chunk], s_s[i:i + chunk], phi_s[i:i + chunk]
                v = lc.unsqueeze(-1) + sc[:, None, None] * zg                         # [c, N, n_grid]
                ph3 = ph[:, None, None]
                lp = dist.NegativeBinomial(
                    total_count=ph3, logits=v * np.log(2.0) + log_offset[:, None] - torch.log(ph3),
                    validate_args=False).log_prob(x_obs_tensor[:, None]) - 0.5 * zg ** 2
                k = torch.distributions.Categorical(logits=lp).sample()               # [c, N]
                zk = zg[k] + (torch.rand_like(lc) - 0.5) * dz
                log_x[i:i + chunk] = (lc + sc[:, None] * zk).cpu()

        samples = {
            "log2_x_eff_g": l_g.cpu(),
            "x_eff_g": (2.0 ** l_g).cpu(),
            "sigma_x_cell": s_s.cpu(),
            "sigma_eff": s_s.unsqueeze(-1).expand(-1, G).cpu(),
            "phi_x": phi_s.cpu(),
            "log_x_true": log_x,
            "x_true": 2.0 ** log_x,
        }
        if assign is not None:
            samples["log2_x_ntc"] = b_s.cpu()
        for k in ("log2_x_target", "tau_guide"):
            if k in draws[0]:
                samples[k] = torch.stack([d[k] for d in draws]).cpu()
        self.model.loss_x = losses
        self.model.posterior_samples_cis = samples
        self.model.x_true = samples["x_true"].median(dim=0).values
        self.model.log2_x_true = samples["log_x_true"].median(dim=0).values
        pyro.clear_param_store()

    def fit_cis(
        self,
        technical_covariates: list[str] = None,
        sum_factor_col: str = 'sum_factor',
        cis_feature: str = None,
        manual_guide_effects: pd.DataFrame = None,
        prior_strength: float = 1.0,
        lr: float = None,
        niters: int = 100_000,
        nsamples: int = 1000,
        alpha_ewma: float = 0.05,
        epsilon: float = 1e-6,
        minibatch_size: int = None,
        predictive_on_cpu: bool = True,
        independent_mu_sigma: bool = False,
        cis_model: str = 'hierarchical',
        n_quadrature: int = 16,
        refit_phi: bool = True,
        tolerance: float = 1e-4,
        pool_scale='learn',
        **kwargs
    ):
        """
        Fits the cis effects (model_x) for your gene_of_interest.
        This step can be repeated multiple times with different priors
        or hyperparameters.

        Parameters
        ----------
        technical_covariates : list, optional
            Technical covariates for correction
        sum_factor_col : str
            Column name for size factors
        cis_feature : str, optional
            Feature ID to use as cis proxy from the primary modality.
            If None, uses self.model.cis_gene (must exist in primary modality).
            For ATAC: region ID (e.g., 'chr9:132283881-132284881')
            For genes: gene name
        manual_guide_effects : pd.DataFrame, optional
            Manual guide effect estimates as priors. DataFrame with columns:
            - guide: guide identifier (matches meta['guide'])
            - log2FC: expected log2 fold-change vs NTC
        prior_strength : float
            Weight for manual guide effects (default: 1.0)
            0 = ignore manual effects, higher = trust more
        lr : float, optional
            Learning rate for Adam. Default 1e-3 for 'hierarchical', 1e-2 for 'marginal'.
        niters : int
            Number of SVI iterations ('marginal': upper limit, see `tolerance`)
        nsamples : int
            Number of posterior samples
        predictive_on_cpu : bool
            If True (default), run posterior Predictive sampling on CPU after SVI
            to reduce GPU memory pressure. Set False to keep Predictive on GPU.
        alpha_ewma : float
            Exponential weight for smoothing the ELBO
        independent_mu_sigma : bool
            Whether to use independent mu/sigma per target type ('hierarchical' only)
        cis_model : str
            'hierarchical' (default): the original _model_x with per-cell latents and a
            mean-field guide. 'marginal': _model_x_marginal, which integrates the per-cell
            latent out by Gauss-Hermite quadrature, uses a full-rank Gaussian guide and
            stops on a windowed relative-improvement rule. See docs/CIS_MARGINAL_MODEL.md.
        n_quadrature : int
            'marginal' only: number of Gauss-Hermite nodes.
        refit_phi : bool
            'marginal' only: re-estimate the cis NB dispersion with a prior centred on the
            fit_ntc estimate. Needed because the NTC fit folds cell-level variation into phi.
        pool_scale : 'learn', float or None
            'marginal' only: partial pooling of guide levels within each target.
            'learn' (default): guides of a target ~ N(target mean, tau), tau ~ HalfNormal(1).
            A float fixes tau (log2 units). None: no pooling, each guide ~ N(m0, 3).
        tolerance : float
            'marginal' only: stop when the mean loss over the last 500 steps improves on the
            previous 500 by less than this (relative).
        kwargs :
            Additional arguments controlling priors, etc.
        """
        print("Running fit_cis...")

        if self.model.cis_gene is None:
            raise ValueError("self.model.cis_gene must be set.")
        if cis_model not in ('hierarchical', 'marginal'):
            raise ValueError("cis_model must be 'hierarchical' or 'marginal'")
        if not (pool_scale is None or pool_scale == 'learn'
                or (isinstance(pool_scale, (int, float)) and pool_scale > 0)):
            raise ValueError("pool_scale must be 'learn', a positive number or None")
        if cis_model == 'marginal' and independent_mu_sigma:
            raise ValueError("independent_mu_sigma is only supported with cis_model='hierarchical'")

        # fit_cis ALWAYS uses the 'cis' modality
        # This modality is created automatically when bayesDREAM is initialized

        # Get cis modality
        if 'cis' not in self.model.modalities:
            raise ValueError(
                "No 'cis' modality found. The 'cis' modality should be created automatically "
                "when bayesDREAM is initialized with a cis_gene. "
                "Make sure cis_gene parameter is set during initialization."
            )

        cis_modality = self.model.get_modality('cis')

        # Determine which feature to use as cis proxy
        if cis_feature is None:
            # Default: use the first (and typically only) feature in cis modality
            cis_feature_idx = cis_modality.feature_meta.index[0]
            # Get actual feature name from metadata (not just numeric index)
            if 'gene_name' in cis_modality.feature_meta.columns:
                cis_feature_name = cis_modality.feature_meta.loc[cis_feature_idx, 'gene_name']
            elif 'gene' in cis_modality.feature_meta.columns:
                cis_feature_name = cis_modality.feature_meta.loc[cis_feature_idx, 'gene']
            elif 'feature' in cis_modality.feature_meta.columns:
                cis_feature_name = cis_modality.feature_meta.loc[cis_feature_idx, 'feature']
            else:
                # Fallback to index if no name column found
                cis_feature_name = cis_feature_idx
            cis_feature = cis_feature_idx  # Use numeric index for actual data retrieval
            print(f"[INFO] Using cis feature '{cis_feature_name}' from 'cis' modality")
        else:
            # User specified explicit cis_feature
            if cis_feature not in cis_modality.feature_meta.index:
                raise ValueError(
                    f"cis_feature '{cis_feature}' not found in 'cis' modality.\n"
                    f"Available features: {cis_modality.feature_meta.index.tolist()}"
                )
            print(f"[INFO] Using cis_feature '{cis_feature}' from 'cis' modality")

        # Get counts for cis feature from cis modality
        if isinstance(cis_modality.counts, pd.DataFrame):
            cis_counts = cis_modality.counts.loc[cis_feature].values
        else:
            # numpy/sparse array - need to find index
            feature_idx = cis_modality.feature_meta.index.get_loc(cis_feature)
            if cis_modality.cells_axis == 1:
                cis_counts = cis_modality.counts[feature_idx, :]
            else:
                cis_counts = cis_modality.counts[:, feature_idx]
            # Densify if sparse (slicing sparse returns sparse row/col matrix)
            if hasattr(cis_counts, 'toarray'):
                cis_counts = cis_counts.toarray().ravel()

        # convert to gpu for fitting
        if self.model.alpha_x_prefit is not None and self.model.alpha_x_prefit.device != self.model.device:
            self.model.alpha_x_prefit = self.model.alpha_x_prefit.to(self.model.device)

        if technical_covariates:
            new_technical_group_code = self.model.meta.groupby(technical_covariates).ngroup()
            if "technical_group_code" in self.model.meta.columns:
                old_technical_group_code = self.model.meta["technical_group_code"].values
                groups_changed = not np.array_equal(old_technical_group_code, new_technical_group_code.values)
                if groups_changed:
                    warnings.warn("technical_group already set. Overwriting with new covariate grouping.")
                    if self.model.alpha_x_prefit is not None:
                        warnings.warn("Technical groups changed; discarding alpha_x_prefit and refitting.")
                        self.model.alpha_x_prefit = None
                else:
                    # Keep loaded alpha_x_prefit when grouping is unchanged (common in stepwise workflows)
                    pass

            self.model.meta["technical_group_code"] = new_technical_group_code
            C = self.model.meta['technical_group_code'].nunique()
            groups_tensor = torch.tensor(self.model.meta['technical_group_code'].values, dtype=torch.long, device=self.model.device)

            # alpha_x_prefit is required when technical_covariates are specified
            if self.model.alpha_x_prefit is None:
                raise ValueError(
                    f"Technical covariates provided but alpha_x_prefit not set. "
                    f"Run fit_ntc() on the primary modality ('{self.model.primary_modality}') before fit_cis() "
                    f"to estimate technical effects for the cis gene."
                )

        elif self.model.alpha_x_prefit is None:
            C = None
            groups_tensor = None
            warnings.warn("no alpha_x_prefit and no technical_covariates provided, assuming no confounding effect.")
        else:
            # alpha_x_prefit exists but no new technical_covariates specified
            # Use existing technical groups
            C = self.model.meta['technical_group_code'].nunique()
            groups_tensor = torch.tensor(self.model.meta['technical_group_code'].values, dtype=torch.long, device=self.model.device)
        
        N = self.model.meta.shape[0]

        # Handle G (number of guides) differently for high MOI vs single-guide mode
        if self.model.is_high_moi:
            G = self.model.guide_assignment.shape[1]  # Number of guides
            guides_tensor = None  # Not used in high MOI mode
        else:
            G = self.model.meta['guide_code'].nunique()
            guides_tensor = torch.tensor(self.model.meta['guide_code'].values, dtype=torch.long, device=self.model.device)

        # Validate minimum data requirements for stable initialization
        min_guides = 3
        min_cells_per_guide = 5
        if self.model.is_high_moi:
            cells_per_guide = self.model.guide_assignment.sum(axis=0)  # [G]
            n_adequate = int((cells_per_guide >= min_cells_per_guide).sum())
        else:
            guide_counts = self.model.meta['guide_code'].value_counts()
            n_adequate = int((guide_counts >= min_cells_per_guide).sum())
        if n_adequate < min_guides:
            raise ValueError(
                f"fit_cis requires at least {min_guides} guides with >= {min_cells_per_guide} cells each, "
                f"but only {n_adequate} guides meet this criterion. "
                f"Subsetting to too few cells/guides will produce unreliable initializations."
            )

        # Use cis_counts from modality-specific lookup (or traditional self.model.counts)
        x_obs_tensor = torch.tensor(cis_counts, dtype=torch.float32, device=self.model.device)

        # ========================================================================
        # MANUAL GUIDE EFFECTS INFRASTRUCTURE
        # ========================================================================
        # If user provides manual guide effects (log2FC estimates), prepare them
        # as tensors that can be used as priors in the Pyro model.
        manual_guide_log2fc_tensor = None
        manual_guide_mask_tensor = None

        if manual_guide_effects is not None:
            raise NotImplementedError(
                "manual_guide_effects is not yet implemented. "
                "The infrastructure for guide-level priors is planned but not yet integrated into _model_x."
            )
        # ========================================================================
        if independent_mu_sigma:
            if ('target' not in self.model.meta.columns):
                raise ValueError("independent_mu_sigma is True, self.model.meta['target'] column not found.")
            elif self.model.meta['target'].nunique() < 2:
                raise ValueError("independent_mu_sigma is True, but only 1 target type found in self.model.meta['target'] column.")

            ### BUILD target_per_guide_tensor [G] based on guide → target
            if self.model.is_high_moi:
                # High MOI: derive one target label per guide for grouping mu/sigma.
                # Two sources: guide_meta['target'] (simple) or guide_targets_dict (many-to-many).
                _ntc_variants = {'ntc', 'NTC', 'non-targeting', 'non-targeting-control', 'Non-Targeting'}

                if 'target' in self.model.guide_meta.columns:
                    guide_target_labels = self.model.guide_meta['target'].tolist()
                elif hasattr(self.model, 'guide_targets_dict') and self.model.guide_targets_dict:
                    # Derive a single representative target per guide.
                    # Priority: cis_gene > any NTC variant > first target.
                    def _primary_target(targets):
                        if self.model.cis_gene and self.model.cis_gene in targets:
                            return self.model.cis_gene
                        for t in targets:
                            if t in _ntc_variants:
                                return 'ntc'
                        return targets[0] if targets else 'ntc'

                    guide_target_labels = [
                        _primary_target(self.model.guide_targets_dict.get(row['guide'], ['ntc']))
                        for _, row in self.model.guide_meta.iterrows()
                    ]
                else:
                    raise ValueError(
                        "independent_mu_sigma=True in high MOI mode requires either "
                        "guide_meta['target'] or guide_targets_dict."
                    )

                target_factorized, target_unique = pd.factorize(guide_target_labels)
                target_per_guide_tensor = torch.tensor(target_factorized, dtype=torch.long, device=self.model.device)
                print(f"[INFO] independent_mu_sigma (high MOI): {len(target_unique)} unique targets")
            else:
                # Single-guide mode: one target code per guide (raises if a guide has multiple targets)
                self.model.meta['target_code'] = pd.factorize(self.model.meta['target'])[0]
                target_codes_tensor = torch.tensor(self.model.meta['target_code'].values, dtype=torch.long, device=self.model.device)

                target_per_guide_tensor = torch.empty(G, dtype=torch.long, device=self.model.device)
                for g in range(G):
                    idx = (guides_tensor == g)
                    guide_targets = torch.unique(target_codes_tensor[idx])
                    if guide_targets.shape[0] != 1:
                        raise ValueError(f"Guide {g} maps to multiple targets: {guide_targets}. independent_mu_sigma=True requires unambiguous target assignment per guide.")
                    target_per_guide_tensor[g] = guide_targets[0]
        else:
            target_per_guide_tensor = None
        primary_mod = self.model.get_modality(self.model.primary_modality)
        sum_factor_tensor = torch.tensor(
            primary_mod.sum_factors.loc[self.model.meta['cell'].values, sum_factor_col].values,
            dtype=torch.float32, device=self.model.device
        )

        # Compute guide means (adjusting for alpha_x if provided)
        x_obs_factored = x_obs_tensor / sum_factor_tensor
        
        if self.model.alpha_x_prefit is not None:
            # alpha_x_prefit is always a [C] point estimate (median already taken at fit_ntc time)
            alpha_x_full = self.model.alpha_x_prefit.flatten()
            # Select the correct alpha_x for each observation (expand for broadcasting)
            alpha_x_used = alpha_x_full[groups_tensor]  # groups_tensor indexes into (C,)

            # Adjust x_obs_factored by alpha_x_used
            x_obs_factored /= alpha_x_used  # Ensure correct shape for division
        
        # --- Extract o_x from the technical fit (cis modality posterior) ---
        cis_modality = self.model.get_modality('cis')
        if (cis_modality.posterior_samples_ntc is None
                or 'o_x' not in cis_modality.posterior_samples_ntc):
            raise ValueError(
                "Cis gene overdispersion not found. Run fit_ntc() before fit_cis().\n"
                "fit_ntc() estimates o_x for the cis gene from NTC cells, which is required "
                "to set a data-driven overdispersion prior instead of the generic Gamma(9,3)."
            )
        # Median matches the point-estimate convention used everywhere else in the
        # pipeline (alpha_x_prefit/alpha_y_prefit/x_true all use median at fit
        # time); also a no-op on a lean-loaded (1-sample) posterior_samples_ntc.
        o_x_ntc = float(torch.quantile(
            cis_modality.posterior_samples_ntc['o_x'].float().flatten(), 0.5).item())
        print(f"[INFO] fit_cis: using technical o_x = {o_x_ntc:.4f} (phi = {1/o_x_ntc**2:.2f})")

        # Compute guide-level means and MADs.
        # Uses numpy on CPU for per-guide median computation (faster than G GPU boolean masks).
        x_np = x_obs_factored.detach().cpu().numpy().astype(np.float32)

        if self.model.is_high_moi:
            # High MOI: vectorized means via matrix-vector multiply, MADs via sorted numpy slices
            assign_np = self.model.guide_assignment  # [N, G] numpy array
            cell_counts_np = assign_np.sum(axis=0).clip(min=1)  # [G]
            guide_sums_np = x_np @ assign_np  # [G]
            guide_means_np = np.log2(np.clip(guide_sums_np / cell_counts_np, epsilon, None))

            log2_x_np = np.log2(np.clip(x_np, epsilon, None))
            guide_mads_np = np.zeros(G, dtype=np.float32)
            for g in range(G):
                cell_mask = assign_np[:, g].astype(bool)
                if cell_mask.sum() > 0:
                    vals = log2_x_np[cell_mask]
                    med = np.median(vals)
                    guide_mads_np[g] = float(np.median(np.abs(vals - med)))

            guide_means = torch.from_numpy(guide_means_np).to(self.model.device)
            guide_mads_tensor = torch.from_numpy(guide_mads_np).to(self.model.device)
        else:
            # Single-guide mode: vectorized means via bincount, MADs via sorted numpy slices
            guides_np = guides_tensor.cpu().numpy()
            n_per_guide_np = np.bincount(guides_np, minlength=G).clip(min=1).astype(np.float32)
            sum_per_guide_np = np.bincount(guides_np, weights=x_np, minlength=G).astype(np.float32)
            guide_means_np = np.log2(np.clip(sum_per_guide_np / n_per_guide_np, epsilon, None))

            log2_x_np = np.log2(np.clip(x_np, epsilon, None))
            sort_idx = np.argsort(guides_np, kind='stable')
            guides_sorted = guides_np[sort_idx]
            log2_x_sorted = log2_x_np[sort_idx]
            boundaries = np.searchsorted(guides_sorted, np.arange(G + 1))

            guide_mads_np = np.zeros(G, dtype=np.float32)
            for g in range(G):
                s, e = boundaries[g], boundaries[g + 1]
                if e > s:
                    vals = log2_x_sorted[s:e]
                    med = np.median(vals)
                    guide_mads_np[g] = float(np.median(np.abs(vals - med)))

            guide_means = torch.from_numpy(guide_means_np).to(self.model.device)
            guide_mads_tensor = torch.from_numpy(guide_mads_np).to(self.model.device)

        mu_x_mean_tensor = torch.mean(guide_means)
        mu_x_sd_tensor = torch.std(guide_means)

        guide_mads_tensor = guide_mads_tensor * 1.4826  # Gaussian-equivalent spread
        sigma_eff_mean_tensor = torch.mean(guide_mads_tensor)#.to(self.model.device)
        sigma_eff_sd_tensor = torch.std(guide_mads_tensor)#.to(self.model.device)

        def init_loc_fn(site):
            name = site["name"]
        
            if name == "mu":
                return mu_x_mean_tensor
            elif name == "sigma":
                return mu_x_sd_tensor.clamp(min=1e-3)
        
            elif name == "sigma_eff_alpha":
                # Corresponds to (mean^2 / var)
                sigma_eff_mean_tensor_tmp = sigma_eff_mean_tensor.clamp(min=1e-2)
                sigma_eff_sd_tensor_tmp = sigma_eff_sd_tensor.clamp(min=1e-2)
                return ((sigma_eff_mean_tensor_tmp ** 2) / (sigma_eff_sd_tensor_tmp ** 2))
            elif name == "sigma_eff_beta":
                # Corresponds to (mean / var)
                sigma_eff_mean_tensor_tmp = sigma_eff_mean_tensor.clamp(min=1e-2)
                sigma_eff_sd_tensor_tmp = sigma_eff_sd_tensor.clamp(min=1e-2)
                return (sigma_eff_mean_tensor_tmp / (sigma_eff_sd_tensor_tmp ** 2))
        
            elif name == "sigma_eff":
                # Assume you’re sampling per-guide (e.g. G = 37)
                return sigma_eff_mean_tensor.clamp(min=1e-2).expand(G)
        
            return pyro.infer.autoguide.initialization.init_to_median(site)

        # Pre-compute NTC guide mask for high MOI mode (avoids pandas access inside Pyro model)
        if self.model.is_high_moi:
            _ntc_variants = {'ntc', 'NTC', 'non-targeting', 'non-targeting-control', 'Non-Targeting'}
            if 'target' in self.model.guide_meta.columns:
                ntc_flags = [self.model.guide_meta.iloc[g]['target'] in _ntc_variants
                             for g in range(G)]
            elif hasattr(self.model, 'guide_targets_dict') and self.model.guide_targets_dict:
                ntc_flags = [
                    any(t in _ntc_variants for t in self.model.guide_targets_dict.get(row['guide'], []))
                    for _, row in self.model.guide_meta.iterrows()
                ]
            else:
                raise ValueError(
                    "High MOI mode requires either guide_meta['target'] or guide_targets_dict "
                    "to identify NTC guides."
                )
            self.model._ntc_guide_mask = torch.tensor(ntc_flags, dtype=torch.bool,
                                                       device=self.model.device)

        # Precompute model invariants once to avoid repeated computation in every SVI step
        phi_x_precomputed = torch.tensor(1.0 / (o_x_ntc ** 2), dtype=torch.float32, device=self.model.device)

        if cis_model == 'marginal':
            log_offset = torch.log(sum_factor_tensor)
            if self.model.alpha_x_prefit is not None:
                log_offset = log_offset + torch.log(alpha_x_used)
            self._fit_cis_marginal(
                N, G, guides_tensor, x_obs_tensor, log_offset, guide_means, phi_x_precomputed,
                lr=1e-2 if lr is None else lr, niters=niters, nsamples=nsamples,
                n_quadrature=n_quadrature, refit_phi=refit_phi, tolerance=tolerance,
                pool_scale=pool_scale,
            )
            print("Finished fit_cis.")
            return
        if lr is None:
            lr = 1e-3
        scale_2_tensor = self._t(2.0)
        _seff_mean = sigma_eff_mean_tensor.clamp(min=1e-2)
        _seff_sd = sigma_eff_sd_tensor.clamp(min=1e-2)
        if (_seff_mean >= 0.01) and (_seff_sd >= 0.01):
            rate_alpha_precomputed = (_seff_sd ** 2) / (_seff_mean ** 2)
            rate_beta_precomputed = (_seff_sd ** 2) / _seff_mean
        else:
            rate_alpha_precomputed = None
            rate_beta_precomputed = None

        guide_x = pyro.infer.autoguide.AutoNormalMessenger(self._model_x, init_loc_fn=init_loc_fn)
        guide_x.to(self.model.device)
        optimizer = pyro.optim.Adam({"lr": lr})
        svi = pyro.infer.SVI(self._model_x, guide_x, optimizer,
                             loss=pyro.infer.Trace_ELBO())

        original_device = self.model.device
        losses = []
        smoothed_loss = None
        alpha_x_sample_loop = self.model.alpha_x_prefit

        for step in range(niters):
            loss = svi.step(
                N,
                G,
                guides_tensor,
                x_obs_tensor,
                sum_factor_tensor,
                mu_x_mean_tensor,
                mu_x_sd_tensor,
                _seff_mean,
                _seff_sd,
                C=C,
                groups_tensor=groups_tensor,
                alpha_x_sample=alpha_x_sample_loop,
                target_per_guide_tensor=target_per_guide_tensor,
                independent_mu_sigma=independent_mu_sigma,
                phi_x_precomputed=phi_x_precomputed,
                rate_alpha_precomputed=rate_alpha_precomputed,
                rate_beta_precomputed=rate_beta_precomputed,
                scale_2_tensor=scale_2_tensor,
            )
            losses.append(loss)
            if step % 1000 == 0:
                print(f"Step {step} : loss = {loss:.5e}, device: {mu_x_mean_tensor.device}")
            if smoothed_loss is None:
                smoothed_loss = loss
            else:
                smoothed_loss = alpha_ewma * loss + (1 - alpha_ewma) * smoothed_loss

        # Move to CPU if using too much GPU memory for Predictive
        run_on_cpu = predictive_on_cpu and self.model.device.type != "cpu"

        if run_on_cpu:
            print(f"[INFO] SVI completed on {original_device}. Running Predictive on CPU to reduce GPU memory pressure...")
            guide_x.to("cpu")
            self.model.device = torch.device("cpu")
            # Move high-MOI tensors that are accessed directly in _model_x
            if self.model.is_high_moi:
                self.model.guide_assignment_tensor = self.model.guide_assignment_tensor.cpu()
                self.model._ntc_guide_mask = self.model._ntc_guide_mask.cpu()

            model_inputs = {
                "N": N,
                "G": G,
                "guides_tensor": self._to_cpu(guides_tensor),
                "x_obs_tensor": self._to_cpu(x_obs_tensor),
                "sum_factor_tensor": self._to_cpu(sum_factor_tensor),
                "mu_x_mean_tensor": self._to_cpu(mu_x_mean_tensor),
                "mu_x_sd_tensor": self._to_cpu(mu_x_sd_tensor),
                "sigma_eff_mean_tensor": self._to_cpu(_seff_mean),
                "sigma_eff_sd_tensor": self._to_cpu(_seff_sd),
                "C": C,
                "groups_tensor": self._to_cpu(groups_tensor),
                "alpha_x_sample": self._to_cpu(self.model.alpha_x_prefit),
                "target_per_guide_tensor": self._to_cpu(target_per_guide_tensor),
                "independent_mu_sigma": independent_mu_sigma,
                "phi_x_precomputed": self._to_cpu(phi_x_precomputed),
                "rate_alpha_precomputed": self._to_cpu(rate_alpha_precomputed),
                "rate_beta_precomputed": self._to_cpu(rate_beta_precomputed),
                "scale_2_tensor": self._to_cpu(scale_2_tensor),
            }
        else:
            model_inputs = {
                "N": N,
                "G": G,
                "guides_tensor": guides_tensor,
                "x_obs_tensor": x_obs_tensor,
                "sum_factor_tensor": sum_factor_tensor,
                "mu_x_mean_tensor": mu_x_mean_tensor,
                "mu_x_sd_tensor": mu_x_sd_tensor,
                "sigma_eff_mean_tensor": _seff_mean,
                "sigma_eff_sd_tensor": _seff_sd,
                "C": C,
                "groups_tensor": groups_tensor,
                "alpha_x_sample": self.model.alpha_x_prefit,
                "target_per_guide_tensor": target_per_guide_tensor,
                "independent_mu_sigma": independent_mu_sigma,
                "phi_x_precomputed": phi_x_precomputed,
                "rate_alpha_precomputed": rate_alpha_precomputed,
                "rate_beta_precomputed": rate_beta_precomputed,
                "scale_2_tensor": scale_2_tensor,
            }

        if self.model.device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        max_samples = nsamples
        keep_sites = kwargs.get("keep_sites", lambda name, site: site["value"].ndim <= 2 or name != "x_obs")

        if minibatch_size is not None:
            from collections import defaultdict
            print(f"[INFO] Running Predictive in minibatches of {minibatch_size}...")
            predictive_x = pyro.infer.Predictive(
                self._model_x,
                guide=guide_x,
                num_samples=minibatch_size,
                parallel=True
            )
            all_samples = defaultdict(list)
            with torch.no_grad():
                for i in range(0, max_samples, minibatch_size):
                    samples = predictive_x(**model_inputs)
                    for k, v in samples.items():
                        if keep_sites(k, {"value": v}):
                            all_samples[k].append(self._to_cpu(v))
                    if self.model.device.type == "cuda":
                        torch.cuda.empty_cache()
                    gc.collect()
            posterior_samples_x = {k: torch.cat(v, dim=0) for k, v in all_samples.items()}
        else:
            predictive_x = pyro.infer.Predictive(
                self._model_x,
                guide=guide_x,
                num_samples=nsamples
            )
            with torch.no_grad():
                posterior_samples_x = predictive_x(**model_inputs)
                if self.model.device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

        if run_on_cpu:
            self.model.device = original_device
            print("[INFO] Reset self.model.device to:", self.model.device)
            if self.model.is_high_moi:
                self.model.guide_assignment_tensor = self.model.guide_assignment_tensor.to(original_device)
                self.model._ntc_guide_mask = self.model._ntc_guide_mask.to(original_device)

        for k, v in posterior_samples_x.items():
            posterior_samples_x[k] = self._to_cpu(v)

        self.model.loss_x = losses
        # Store full posterior on model (not just on CisFitter)
        self.model.posterior_samples_cis = posterior_samples_x
        self.model.x_true = posterior_samples_x['x_true'].median(dim=0).values
        self.model.log2_x_true = posterior_samples_x['log_x_true'].median(dim=0).values

        if self.model.device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        pyro.clear_param_store()

        print("Finished fit_cis.")
