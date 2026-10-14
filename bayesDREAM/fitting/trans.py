"""
Trans effects fitting for bayesDREAM.

This module contains the trans model and fitting logic.
"""

import os
import warnings
import numpy as np
import pandas as pd
import torch
# Pre-import torch._dynamo to avoid circular import issues with OneCycleLR scheduler
# This is a workaround for a known PyTorch issue in some versions/builds (especially ROCm)
try:
    import torch._dynamo
except (ImportError, AttributeError):
    pass  # Older PyTorch versions may not have _dynamo
import pyro
import pyro.distributions as dist
from pyro.distributions.transforms import iterated, affine_autoregressive
import pyro.optim as optim
import pyro.infer as infer
import pyro.poutine as poutine
import pyro.distributions as dist

from ..utils import find_beta, Hill_based_positive, Hill_based_positive_logK, Polynomial_function, check_tensor


class _FeatureBlockGuide:
    """
    Variational family for the per-feature latent variables of the logistic_hill model.

    For every feature t, the unconstrained values of all per-feature sites are stacked
    into one vector eta_t of length d and given a full-rank Gaussian,

        eta_t ~ MVN(loc_t, L_t L_t^T),

    independently across features. This is block-diagonal over features, so the cost is
    O(T d^2) and the correlations within a feature (amplitude vs. midpoint vs. slope) are
    kept. Each site is mapped to its support with biject_to and the log-Jacobian is
    attached through a Delta site, as in pyro's AutoContinuous. Global scalar sites get
    independent Gaussians in unconstrained space.

    Parameters
    ----------
    T : int
        Number of features.
    J : int
        Number of logistic components.
    site_specs : list of (name, per_component, constraint)
        per_component=False -> site has shape [T]; True -> shape [J, T].
    init_loc : torch.Tensor
        Initial location in unconstrained space, shape [T, d].
    init_scale : torch.Tensor
        Initial standard deviations (diagonal of L_t), shape [T, d].
    global_specs : list of (name, constraint, init_loc, init_scale)
        Scalar sites shared by all features, in unconstrained space.
    """

    def __init__(self, T, J, site_specs, init_loc, init_scale, global_specs=()):
        self.T = T
        self.J = J
        self.site_specs = site_specs
        self.global_specs = list(global_specs)
        self.init_loc = init_loc.detach().clone()
        self.init_scale = init_scale.detach().clone()
        self.d = init_loc.shape[-1]

    def __call__(self, *args, **kwargs):
        from torch.distributions import biject_to, constraints

        device = self.init_loc.device
        for name, constraint, init_loc, init_scale in self.global_specs:
            g_loc = pyro.param(f"logistic_{name}_loc", lambda: torch.as_tensor(init_loc, device=device).clone())
            g_scale = pyro.param(f"logistic_{name}_scale", lambda: torch.as_tensor(init_scale, device=device).clone(),
                                 constraint=constraints.positive)
            pyro.sample(name, dist.TransformedDistribution(dist.Normal(g_loc, g_scale), biject_to(constraint)))

        loc = pyro.param("logistic_block_loc", lambda: self.init_loc.clone())
        scale_tril = pyro.param(
            "logistic_block_scale_tril",
            lambda: torch.diag_embed(self.init_scale),
            constraint=constraints.lower_cholesky,
        )
        component_plate = pyro.plate("logistic_component_plate", self.J, dim=-2)
        with pyro.plate("trans_plate", self.T, dim=-1):
            eta = pyro.sample(
                "_logistic_block_latent",
                dist.MultivariateNormal(loc, scale_tril=scale_tril),
                infer={"is_auxiliary": True},
            )  # [..., T, d]
            pos = 0
            for name, per_component, constraint in self.site_specs:
                width = self.J if per_component else 1
                unconstrained = eta[..., pos:pos + width]  # [..., T, width]
                pos += width
                transform = biject_to(constraint)
                value = transform(unconstrained)
                log_density = -transform.log_abs_det_jacobian(unconstrained, value)
                if per_component:
                    with component_plate:
                        pyro.sample(name, dist.Delta(value.transpose(-1, -2),
                                                     log_density=log_density.transpose(-1, -2)))
                else:
                    pyro.sample(name, dist.Delta(value[..., 0], log_density=log_density[..., 0]))



class TransFitter:
    """Handles trans effects fitting."""

    def __init__(self, model):
        """
        Initialize trans fitter.

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

    # ----------------------------
    # Debug helpers (Option 1)
    # ----------------------------
    def _tensor_summary(self, x: torch.Tensor):
        x_det = x.detach()
        finite = torch.isfinite(x_det)
        n = x_det.numel()
        n_bad = (~finite).sum().item()
        n_nan = torch.isnan(x_det).sum().item()
        n_inf = torch.isinf(x_det).sum().item()
        if finite.any():
            x_f = x_det[finite]
            mn = x_f.min().item()
            mx = x_f.max().item()
            mean = x_f.mean().item()
            std = x_f.std().item()
        else:
            mn = mx = mean = std = float("nan")
        return dict(n=n, n_bad=n_bad, n_nan=n_nan, n_inf=n_inf, min=mn, max=mx, mean=mean, std=std)

    def _print_tensor(self, name: str, x: torch.Tensor, prefix=""):
        s = self._tensor_summary(x)
        print(
            f"{prefix}{name}: shape={tuple(x.shape)} dtype={x.dtype} device={x.device} "
            f"bad={s['n_bad']}/{s['n']} (nan={s['n_nan']}, inf={s['n_inf']}) "
            f"min={s['min']:.3g} max={s['max']:.3g} mean={s['mean']:.3g} std={s['std']:.3g}"
        )

    def _check_param_store_all(self, step: int, svi=None, where="", prev_finite=None, verbose_on_change=True):
        store = pyro.get_param_store()
        if prev_finite is None:
            prev_finite = {}

        changed = []
        bad_vals = []
        bad_grads = []

        for name, p in store.items():
            if not isinstance(p, torch.Tensor):
                continue

            now_finite = torch.isfinite(p).all().item()
            was_finite = prev_finite.get(name, True)
            prev_finite[name] = now_finite

            if was_finite and (not now_finite):
                changed.append(name)

            if not now_finite:
                bad_vals.append(name)

            if p.grad is not None and (not torch.isfinite(p.grad).all().item()):
                bad_grads.append(name)

        if changed and verbose_on_change:
            print(f"\n[FATAL] Param(s) flipped finite→nonfinite at step {step} {where}: {changed}")
            for n in changed:
                self._print_tensor(n, store[n], prefix="  ")
                if store[n].grad is not None:
                    self._print_tensor(n + ".grad", store[n].grad, prefix="  ")

        if bad_vals and verbose_on_change and not changed:
            print(f"\n[FATAL] Non-finite param value(s) present at step {step} {where}: "
                  f"{bad_vals[:10]}{'...' if len(bad_vals)>10 else ''}")
            for n in bad_vals[:10]:
                self._print_tensor(n, store[n], prefix="  ")

        if bad_grads and verbose_on_change:
            print(f"\n[FATAL] Non-finite gradient(s) present at step {step} {where}: "
                  f"{bad_grads[:10]}{'...' if len(bad_grads)>10 else ''}")
            for n in bad_grads[:10]:
                if store[n].grad is not None:
                    self._print_tensor(n + ".grad", store[n].grad, prefix="  ")

        # Best-effort optimizer-state check (optional)
        if svi is not None and hasattr(svi, "optim"):
            opt = svi.optim
            optim_objs = getattr(opt, "optim_objs", None)
            if isinstance(optim_objs, dict):
                bad_state = []
                for pname, torch_opt in optim_objs.items():
                    base_opt = getattr(torch_opt, "optimizer", torch_opt)
                    if not hasattr(base_opt, "state"):
                        continue
                    for group in base_opt.param_groups:
                        for param in group.get("params", []):
                            st = base_opt.state.get(param, {})
                            for key, val in st.items():
                                if torch.is_tensor(val) and (not torch.isfinite(val).all().item()):
                                    bad_state.append((pname, key))
                if bad_state and verbose_on_change:
                    print(f"\n[FATAL] Non-finite optimizer state at step {step} {where}: "
                          f"{bad_state[:10]}{'...' if len(bad_state)>10 else ''}")

        any_bad = bool(bad_vals or bad_grads)
        return prev_finite, any_bad

    def _collect_dist_tensors(self, fn):
        tensors = []
        if isinstance(fn, dist.TransformedDistribution):
            tensors += self._collect_dist_tensors(fn.base_dist)
            return tensors

        for attr in [
            "loc", "scale", "total_count", "logits", "probs", "rate",
            "concentration", "concentration0", "concentration1", "df"
        ]:
            if hasattr(fn, attr):
                v = getattr(fn, attr)
                if torch.is_tensor(v):
                    tensors.append((attr, v))
        return tensors

    def diagnose_nonfinite_sites(self, model, guide, *args, **kwargs):
        print("\n[DIAG] Running traced guide+model to locate non-finite site...")

        guide_trace = poutine.trace(guide).get_trace(*args, **kwargs)
        replayed_model = poutine.replay(model, trace=guide_trace)
        model_trace = poutine.trace(replayed_model).get_trace(*args, **kwargs)

        def scan_trace(tr, label):
            print(f"[DIAG] Scanning {label} trace...")
            for name, node in tr.nodes.items():
                if node.get("type") != "sample":
                    continue

                fn = node["fn"]
                val = node["value"]

                if torch.is_tensor(val) and (not torch.isfinite(val).all().item()):
                    print(f"  [BAD VALUE] site={name}")
                    self._print_tensor(f"{label}:{name}.value", val, prefix="    ")
                    return name

                for attr, ten in self._collect_dist_tensors(fn):
                    if not torch.isfinite(ten).all().item():
                        print(f"  [BAD DIST PARAM] site={name} param={attr}")
                        self._print_tensor(f"{label}:{name}.fn.{attr}", ten, prefix="    ")
                        return name

                if "log_prob" in node:
                    lp = node["log_prob"]
                    if torch.is_tensor(lp) and (not torch.isfinite(lp).all().item()):
                        print(f"  [BAD LOG_PROB] site={name}")
                        self._print_tensor(f"{label}:{name}.log_prob", lp, prefix="    ")
                        return name

            print(f"[DIAG] No non-finite sample sites found in {label} trace.")
            return None

        bad_guide = scan_trace(guide_trace, "guide")
        bad_model = scan_trace(model_trace, "model")
        print(f"[DIAG] First bad site guide={bad_guide} model={bad_model}\n")

    def _debug_svi_step(self, svi, step, prev_finite, *args, **kwargs):
        """
        A 'manual' SVI step that lets us check:
          - pre-step params
          - post-grads, pre-update params/grads
          - post-update params
        and run trace diagnosis on first failure.
        """
        # Pre-step check
        prev_finite, any_bad = self._check_param_store_all(
            step, svi=svi, where="(pre-step)", prev_finite=prev_finite
        )
        if any_bad:
            store = pyro.get_param_store()
            if "locs.log_K_a" in store and store["locs.log_K_a"].grad is not None:
                g = store["locs.log_K_a"].grad
                bad_idx = torch.nonzero(torch.isnan(g), as_tuple=False).squeeze(-1)
                if bad_idx.numel() > 0:
                    self._diagnose_log_Ka_nan_grad(svi, bad_idx, *args, **kwargs)
        
            self.diagnose_nonfinite_sites(svi.model, svi.guide, *args, **kwargs)
            raise FloatingPointError("Non-finite grads/params after loss_and_grads (before update).")


        # Compute loss + grads (no update yet)
        with poutine.trace(param_only=True) as param_capture:
            loss = svi.loss_and_grads(svi.model, svi.guide, *args, **kwargs)

        # Post-grads check (this often catches the real cause)
        prev_finite, any_bad = self._check_param_store_all(
            step, svi=svi, where="(post-grads, pre-update)", prev_finite=prev_finite
        )
        if any_bad:
            store = pyro.get_param_store()
            if "locs.log_K_a" in store and store["locs.log_K_a"].grad is not None:
                g = store["locs.log_K_a"].grad
                bad_idx = torch.nonzero(torch.isnan(g), as_tuple=False).squeeze(-1)
                if bad_idx.numel() > 0:
                    self._diagnose_log_Ka_nan_grad(svi, bad_idx, *args, **kwargs)
        
            self.diagnose_nonfinite_sites(svi.model, svi.guide, *args, **kwargs)
            raise FloatingPointError("Non-finite grads/params after loss_and_grads (before update).")
        if any_bad:
            self.diagnose_nonfinite_sites(svi.model, svi.guide, *args, **kwargs)
            raise FloatingPointError("Non-finite grads/params after loss_and_grads (before update).")

        # Apply optimizer update
        params = set(site["value"].unconstrained() for site in param_capture.trace.nodes.values())
        svi.optim(params)

        # Post-update check
        prev_finite, any_bad = self._check_param_store_all(
            step, svi=svi, where="(post-update)", prev_finite=prev_finite
        )
        if any_bad:
            store = pyro.get_param_store()
            if "locs.log_K_a" in store and store["locs.log_K_a"].grad is not None:
                g = store["locs.log_K_a"].grad
                bad_idx = torch.nonzero(torch.isnan(g), as_tuple=False).squeeze(-1)
                if bad_idx.numel() > 0:
                    self._diagnose_log_Ka_nan_grad(svi, bad_idx, *args, **kwargs)
        
            self.diagnose_nonfinite_sites(svi.model, svi.guide, *args, **kwargs)
            raise FloatingPointError("Non-finite grads/params after loss_and_grads (before update).")
        if any_bad:
            self.diagnose_nonfinite_sites(svi.model, svi.guide, *args, **kwargs)
            raise FloatingPointError("Optimizer update produced non-finite params.")

        return loss, prev_finite

    def _diagnose_log_Ka_nan_grad(self, svi, bad_idx, *args, **kwargs):
        """
        bad_idx: 1D LongTensor of indices where locs.log_K_a.grad is NaN
        Prints model/guide site values for those indices and derived quantities
        (phi, Hill logit range, NB logits range).
        """
        import pyro.poutine as poutine
        import torch
    
        print(f"\n[DIAG+] NaN grad indices for log_K_a: {bad_idx.tolist()}")
    
        guide_trace = poutine.trace(svi.guide).get_trace(*args, **kwargs)
        replayed_model = poutine.replay(svi.model, trace=guide_trace)
        model_trace = poutine.trace(replayed_model).get_trace(*args, **kwargs)
    
        def get_site(tr, name):
            return tr.nodes[name]["value"] if name in tr.nodes else None
    
        # --- pull site values (all should be finite, but may be extreme) ---
        log_K_a = get_site(model_trace, "log_K_a") or get_site(guide_trace, "log_K_a")
        log_Vmax_a = get_site(model_trace, "log_Vmax_a") or get_site(guide_trace, "log_Vmax_a")
        Vmax_a = get_site(model_trace, "Vmax_a")
        K_a = get_site(model_trace, "K_a")
        n_a = get_site(model_trace, "n_a")
        A = get_site(model_trace, "A")
        o_y = get_site(model_trace, "o_y")  # sampled per feature
        alpha = get_site(model_trace, "alpha")  # relaxed bernoulli
    
        # Some of these might not exist depending on branch / naming
        for name, tensor in [
            ("log_K_a", log_K_a),
            ("K_a", K_a),
            ("log_Vmax_a", log_Vmax_a),
            ("Vmax_a", Vmax_a),
            ("n_a", n_a),
            ("A", A),
            ("o_y", o_y),
            ("alpha", alpha),
        ]:
            if tensor is None:
                continue
            sel = tensor[bad_idx]
            self._print_tensor(f"{name}[bad_idx]", sel, prefix="  ")
    
        # --- derived phi and ranges (very informative) ---
        if o_y is not None:
            phi = 1.0 / (o_y ** 2)
            self._print_tensor("phi_y[bad_idx]", phi[bad_idx], prefix="  ")
    
        # --- derived Hill logit range (this catches x**n instabilities) ---
        # kwargs must include x_true_sample, and we prefer to use log_K_a directly
        x_true = kwargs.get("x_true_sample", None)
        if x_true is not None and log_K_a is not None and n_a is not None:
            tiny = torch.finfo(x_true.dtype).tiny
            log_x = torch.log(x_true.clamp_min(tiny)).unsqueeze(-1)  # [N,1]
            # logit for each bad feature: [N, |bad_idx|]
            logit = (n_a[bad_idx].unsqueeze(0) * (log_x - log_K_a[bad_idx].unsqueeze(0)))
            # print range across cells (per bad feature)
            logit_min = logit.min(dim=0).values
            logit_max = logit.max(dim=0).values
            self._print_tensor("hill_logit_min[bad_idx]", logit_min, prefix="  ")
            self._print_tensor("hill_logit_max[bad_idx]", logit_max, prefix="  ")
    
            # approximate Hill output range
            if Vmax_a is not None:
                H = Vmax_a[bad_idx].unsqueeze(0) * torch.sigmoid(logit)
                self._print_tensor("Hilla_min[bad_idx]", H.min(dim=0).values, prefix="  ")
                self._print_tensor("Hilla_max[bad_idx]", H.max(dim=0).values, prefix="  ")
    
        # --- derived NB logits range for bad features (if you pass mu_y pieces) ---
        # We can’t see mu_final unless you expose it; but we can at least compute the part driven by Hill:
        # If you want full mu_final, expose it via pyro.deterministic (recommended).
        print("[DIAG+] If hill_logit range is extreme (e.g. > ~80 in magnitude), Hill math/backward is a prime suspect.\n")

    
    #########################################
    ## Step 3: Fit trans effects (model_y) ##
    #########################################
    def _model_y(
        self,
        N,
        T,
        y_obs_tensor,
        sum_factor_tensor,
        beta_o_alpha_tensor,
        beta_o_beta_tensor,
        alpha_alpha_mu_tensor,
        K_max_tensor,
        K_alpha_tensor,
        Vmax_mean_tensor,
        Vmax_alpha_tensor,
        n_mu_tensor,
        Amean_tensor,
        p_n_logits_tensor,
        epsilon_tensor,
        x_true_sample,
        log2_x_true_sample,
        nmin,
        nmax,
        alpha_y_sample=None,
        C=None,
        groups_tensor=None,
        temperature=1.0,
        use_straight_through=False,
        function_type='single_hill',
        polynomial_degree=6,
        use_alpha=True,
        distribution='negbinom',
        denominator_tensor=None,
        K=None,
        D=None,
        mean_within_guide_var=None,
        x_true_CV=None,
        use_data_driven_priors=True,
        use_lognormal_priors=True,
        use_epsilon=True,
    ):

        if use_alpha:
            alpha_dist = dist.RelaxedBernoulliStraightThrough if use_straight_through else dist.RelaxedBernoulli

        trans_plate = pyro.plate("trans_plate", T, dim=-1)
        
        ##########
        ## x_true (Now Fully Observed) ##
        ##########
        #x_true = pyro.deterministic("x_true", x_true_sample)
        x_true = x_true_sample
    
        ####################
        ## alpha_y ##
        ####################
        alpha_y = None
        if alpha_y_sample is not None:
            alpha_y = alpha_y_sample
        elif groups_tensor is not None:
            with pyro.plate("technical_groups_plate", C-1, dim=-2):  # **Now correctly uses C-1**
                with trans_plate:
                    alpha_alpha = pyro.sample("alpha_alpha", dist.Exponential(1 / alpha_alpha_mu_tensor))  # shape = [C-1, T]
                    alpha_mu = pyro.sample("alpha_mu", dist.Gamma(1, 1))  # shape = [C-1, T]
                    alpha_y = pyro.sample("alpha_y", dist.Gamma(alpha_alpha, alpha_alpha / alpha_mu))  # shape = [C-1, T]
    
        ####################
        ## Overdispersion ##
        ####################
        beta_o = pyro.sample("beta_o", dist.Gamma(beta_o_alpha_tensor, beta_o_beta_tensor))
        with trans_plate:
            o_y = pyro.sample("o_y", dist.Exponential(beta_o))
            phi_y = 1 / (o_y**2)
        phi_y_used = phi_y.unsqueeze(-2)

        # Degrees of freedom for Student-t distribution (nu_y)
        # Only needed if using studentt distribution
        nu_y = None
        if distribution == 'studentt':
            # Two options (must match fit_technical choice):
            # Option 1: Fixed value (simpler, faster) - COMMENTED OUT FOR NOW
            # nu_y = self._t(3.0)
            # Option 2: Sample per-feature (more flexible, slower) - ACTIVE
            with trans_plate:
                nu_y = pyro.sample("nu_y", dist.Gamma(self._t(10.0), self._t(2.0)))  # mean~5, ensures df>2
    
        #################
        ## Hill-based: ##
        #################
        if function_type in ['single_hill', 'additive_hill', 'nested_hill']:
            # ----------------------------------------------------
            # 1) Define global hyperparameters for n_a
            # ----------------------------------------------------
            sigma_n_a = pyro.sample("sigma_n_a", dist.Exponential(self._t(1/5))) #   -> controls how variable n_a can be across genes
            if function_type in ['additive_hill', 'nested_hill']:
                sigma_n_b = pyro.sample("sigma_n_b", dist.Exponential(self._t(1/5))) #   -> controls how variable n_a can be across genes
        if function_type in ['polynomial']:
            #sigma_coeff = pyro.sample("sigma_coeff", dist.Exponential(100)) #   -> controls how variable n_a can be across genes
            sigma_coeff = pyro.sample("sigma_coeff", dist.HalfCauchy(scale=self._t(1.0)))
        
        # Now enter the trans_plate (T dimension)
        with trans_plate:

            weight = o_y / (o_y + (beta_o_beta_tensor / beta_o_alpha_tensor)).clamp_min(epsilon_tensor)

            # For multinomial, Amean_tensor and Vmax_mean_tensor are already [T, K] shaped
            # For binomial, they are [T] shaped
            # We keep them in their native shape for per-category priors in multinomial

            # Baseline parameter A depends on distribution:
            # - normal/studentt: can be negative (natural value space)
            # - negbinom: positive count
            # - binomial/multinomial: probability in [0,1], using NEW reparameterization
            if distribution in ['normal', 'studentt']:
                # For continuous distributions: average of min and max
                Amean_for_A = Amean_tensor  # [T]
                Vmax_for_A = Vmax_mean_tensor  # [T]
                Amean_adjusted = ((1 - weight) * Amean_for_A) + (weight * Vmax_for_A) + epsilon_tensor
                # Use Normal distribution to allow negative baseline values
                A = pyro.sample("A", dist.Normal(Amean_adjusted, Amean_adjusted.abs()))

            elif distribution in ['binomial', 'multinomial']:
                # SIMPLIFIED Beta/Dirichlet priors with weak regularization
                # For binomial:
                #   A ~ Beta with α=1 (pushes toward 0, mean = A_mean)
                #   upper_limit ~ Beta with β=1 (pushes toward 1, mean = Vmax_mean)
                # For multinomial:
                #   A ~ Dirichlet with concentration = mean * K (weak prior)
                #   upper_limit ~ Dirichlet with concentration = mean * K

                # Amean_tensor: [T] for binomial, [T, K] for multinomial
                # Vmax_mean_tensor: [T] for binomial, [T, K] for multinomial

                # Sample A and upper_limit
                if distribution == 'multinomial' and Amean_tensor.ndim > 1:
                    # For multinomial: Use Dirichlet with weak concentration
                    # concentration = mean_normalized * K (where K = number of categories)
                    # This gives each category concentration ≈ 1 on average (weak prior)
                    K_dim = Amean_tensor.shape[-1]

                    # For multinomial: use Dirichlet priors over K categories
                    if use_data_driven_priors:
                        # Data-driven Dirichlet: concentration proportional to mean probabilities
                        # Normalize means to sum to 1
                        A_mean_clamped = Amean_tensor.clamp(min=epsilon_tensor, max=1.0 - epsilon_tensor)
                        A_mean_normalized = A_mean_clamped / A_mean_clamped.sum(dim=-1, keepdim=True)  # [T, K]

                        Vmax_clamped = Vmax_mean_tensor.clamp(min=epsilon_tensor, max=1.0 - epsilon_tensor)
                        upper_mean_normalized = Vmax_clamped / Vmax_clamped.sum(dim=-1, keepdim=True)  # [T, K]

                        # Weak concentration: mean_normalized * K gives ~1 per category
                        concentration_A = A_mean_normalized * K_dim  # [T, K]
                        concentration_upper = upper_mean_normalized * K_dim  # [T, K]

                        #print(f"[INFO] Dirichlet data-driven concentration: A={concentration_A.mean().item():.2f}, upper={concentration_upper.mean().item():.2f}")
                    else:
                        # Uniform Dirichlet: all categories have equal concentration=1
                        concentration_A = self._t(1.0).expand([T, K_dim])  # [T, K] with all 1s
                        concentration_upper = self._t(1.0).expand([T, K_dim])  # [T, K] with all 1s

                    # Sample K-dimensional probability vectors from Dirichlet
                    # Each row sums to 1
                    A = pyro.sample("A", dist.Dirichlet(concentration_A))  # [T, K]
                    upper_limit = pyro.sample("upper_limit", dist.Dirichlet(concentration_upper))  # [T, K]

                else:
                    # For binomial: Beta priors
                    if use_data_driven_priors:
                        # Data-driven Beta priors with α=1 or β=1
                        # A ~ Beta(α=1, β) with mean = A_mean
                        #   mean = α/(α+β) = 1/(1+β) = A_mean
                        #   β = (1-A_mean)/A_mean
                        beta_A = (1.0 - Amean_tensor) / Amean_tensor  # [T]
                        alpha_A = self._t(1.0)  # [scalar] or could be torch.ones_like(beta_A)

                        # upper_limit ~ Beta(α, β=1) with mean = Vmax_mean
                        #   mean = α/(α+1) = Vmax_mean
                        #   α = Vmax_mean/(1-Vmax_mean)
                        alpha_upper = Vmax_mean_tensor / (1.0 - Vmax_mean_tensor)  # [T]
                        beta_upper = self._t(1.0)  # [scalar]

                        #print(f"[INFO] Beta data-driven priors: A (α=1, β̄={beta_A.mean().item():.2f}), upper (ᾱ={alpha_upper.mean().item():.2f}, β=1)")
                    else:
                        # Uniform priors: Beta(1, 1) for both A and upper_limit
                        alpha_A = self._t(1.0)
                        beta_A = self._t(1.0)
                        alpha_upper = self._t(1.0)
                        beta_upper = self._t(1.0)

                    # Sample per-feature [T]
                    # Note: When using uniform priors, these become scalars broadcast to [T]
                    if use_data_driven_priors:
                        A = pyro.sample("A", dist.Beta(alpha_A, beta_A))  # [T]
                        upper_limit = pyro.sample("upper_limit", dist.Beta(alpha_upper, beta_upper))  # [T]
                    else:
                        # Uniform priors - need to expand to match shape [T]
                        A = pyro.sample("A", dist.Beta(alpha_A, beta_A).expand([T]))  # [T]
                        upper_limit = pyro.sample("upper_limit", dist.Beta(alpha_upper, beta_upper).expand([T]))  # [T]

            else:
                # For negbinom: A must be positive count
                Amean_for_A = Amean_tensor  # [T]
                Vmax_for_A = Vmax_mean_tensor  # [T]
                Amean_adjusted = ((1 - weight) * Amean_for_A) + (weight * Vmax_for_A) + epsilon_tensor
                A = pyro.sample("A", dist.Exponential(1 / Amean_adjusted))

            if use_alpha:
                # Relaxed Bernoulli: alpha ~ (0,1), becomes more discrete as temperature -> 0
                alpha = pyro.sample("alpha", alpha_dist(temperature=temperature, logits=p_n_logits_tensor))
            else:
                alpha = torch.ones((T,), device=self.model.device)
            
            if function_type in ['single_hill', 'additive_hill', 'nested_hill']:

                #####################################
                ## function priors (depend on o_y) ##
                #####################################
                # Gamma and delta depend on T dimension
                # Reduce over group dimension if necessary
                K_sigma = (K_max_tensor / (self._t(2) * torch.sqrt(K_alpha_tensor))) + epsilon_tensor

                # For multinomial, use reduced Vmax_for_A (without category dimension) for priors
                # For other distributions, use Vmax_mean_tensor directly
                if distribution == 'multinomial' and Vmax_mean_tensor.ndim > 1:
                    Vmax_prior_mean = Vmax_for_A  # [T] - already reduced
                else:
                    Vmax_prior_mean = Vmax_mean_tensor  # [T]

                Vmax_sigma = (Vmax_prior_mean / torch.sqrt(Vmax_alpha_tensor)) + epsilon_tensor

                # For multinomial, we need per-category parameters (K-1 categories, Kth is residual)
                if distribution == 'multinomial' and K is not None:
                    K_minus_1 = K - 1
                    # Sample parameters for K-1 categories
                    # Each category gets its own Hill function parameters
                    with pyro.plate("category_plate_n_a", K_minus_1, dim=-1):
                        n_a_raw = pyro.sample("n_a_raw", dist.Normal(n_mu_tensor, sigma_n_a))  # [T, K-1]
                        BOX_LOW  = self._t(-20.0)
                        BOX_HIGH = self._t( 20.0)
                        low  = torch.maximum(nmin, BOX_LOW)
                        high = torch.minimum(nmax, BOX_HIGH)
                        n_a = pyro.deterministic(
                            "n_a",
                            (alpha.unsqueeze(-1) * n_a_raw).clamp(min=low.item(), max=high.item())
                        )  # [T, 1] * [T, K-1] -> [T, K-1]
                else:
                    # For non-multinomial: single set of parameters per feature
                    n_a_raw = pyro.sample("n_a_raw", dist.Normal(n_mu_tensor, sigma_n_a))
                    BOX_LOW  = self._t(-20.0)
                    BOX_HIGH = self._t( 20.0)
                    low  = torch.maximum(nmin, BOX_LOW)
                    high = torch.minimum(nmax, BOX_HIGH)
                    n_a = pyro.deterministic(
                        "n_a",
                        (alpha * n_a_raw).clamp(min=low.item(), max=high.item())
                    )
                
                # Scale for Vmax, K is multiplied by alpha
                #eff_Vmax_sigma = alpha * Vmaxa_sigma + epsilon_tensor
                #eff_Ka_sigma   = alpha * Ka_sigma    + epsilon_tensor

                # UNIFIED Vmax and K priors (Log-Normal for all distributions)
                # K uses CV-based std (scale-invariant, works without guides)
                # Vmax uses raw variance (data-driven) for negbinom/normal/studentt

                # K parameterization (UNIFIED for all distributions)
                K_mean_prior = (K_max_tensor / 2.0).clamp_min(epsilon_tensor)  # Half of max cis expression
                if x_true_CV is not None:
                    K_std_prior = K_mean_prior * x_true_CV  # Scale by coefficient of variation
                else:
                    # Fallback to fixed alpha if CV not provided
                    K_std_prior = K_max_tensor / (self._t(2.0) * torch.sqrt(K_alpha_tensor))

                # Log-Normal parameterization for K
                ratio_K = (K_std_prior / K_mean_prior).clamp_min(self._t(1e-6))
                K_log_sigma = torch.sqrt(torch.log1p(ratio_K ** 2))
                K_log_mu = torch.log(K_mean_prior) - 0.5 * K_log_sigma ** 2

                if distribution in ['binomial', 'multinomial']:
                    # For binomial/multinomial: Sample Vmax_a INDEPENDENTLY (like BCD1C4F)
                    # Even for multinomial, the Kth category is residual, so we can have independent Vmax
                    # This gives Vmax_a direct gradient signal, not mediated through alpha
                    # Avoids chicken-and-egg: alpha needs signal → signal needs Vmax → Vmax needs alpha
                    Vmax_mean_clamped = Vmax_mean_tensor.clamp(min=0.01, max=0.99)
                    concentration_vmax = self._t(10.0)
                    alpha_vmax = Vmax_mean_clamped * concentration_vmax
                    beta_vmax = (1 - Vmax_mean_clamped) * concentration_vmax

                    if distribution == 'multinomial' and K is not None:
                        # For multinomial: Sample Vmax_a for K-1 categories (Kth is residual)
                        K_minus_1 = K - 1
                        with pyro.plate("category_plate_Vmax_a", K_minus_1, dim=-1):
                            Vmax_a = pyro.sample("Vmax_a", dist.Beta(alpha_vmax, beta_vmax))  # [T, K-1]

                        # K_a: Log-Normal for K-1 categories
                        with pyro.plate("category_plate_K_a", K_minus_1, dim=-1):
                            log_K_a = pyro.sample("log_K_a", dist.Normal(K_log_mu, K_log_sigma))  # [T, K-1]
                            K_a = pyro.deterministic("K_a", torch.exp(log_K_a))  # [T, K-1]
                    else:
                        # For binomial: per-feature Vmax_a and K_a
                        Vmax_a = pyro.sample("Vmax_a", dist.Beta(alpha_vmax, beta_vmax))  # [T]
                        log_K_a = pyro.sample("log_K_a", dist.Normal(K_log_mu, K_log_sigma))  # [T]
                        K_a = pyro.deterministic("K_a", torch.exp(log_K_a))  # [T]

                else:
                    # For negbinom/normal/studentt: Choose between Gamma and Log-Normal priors
                    K_sigma = (K_max_tensor / (self._t(2.0) * torch.sqrt(K_alpha_tensor))) + epsilon_tensor
                    Vmax_sigma = (Vmax_prior_mean / torch.sqrt(Vmax_alpha_tensor)) + epsilon_tensor

                    if use_lognormal_priors:
                        # Log-Normal priors (experimental) - use for comparison
                        Vmax_log_mu = torch.log(Vmax_prior_mean) - 0.5 * torch.log1p((Vmax_sigma / Vmax_prior_mean) ** 2)
                        Vmax_log_sigma = torch.sqrt(torch.log1p((Vmax_sigma / Vmax_prior_mean) ** 2))
                        log_Vmax_a = pyro.sample("log_Vmax_a", dist.Normal(Vmax_log_mu, Vmax_log_sigma))
                        Vmax_a = pyro.deterministic("Vmax_a", torch.exp(log_Vmax_a))

                        log_K_a = pyro.sample("log_K_a", dist.Normal(K_log_mu, K_log_sigma))
                        K_a = pyro.deterministic("K_a", torch.exp(log_K_a))
                    else:
                        # Gamma priors (default, matching archive) - better convergence and sparsity
                        # Gamma parameterization: shape = mean^2/var, rate = mean/var
                        Vmax_a = pyro.sample("Vmax_a", dist.Gamma(
                            (Vmax_prior_mean ** 2) / (Vmax_sigma ** 2),
                            Vmax_prior_mean / (Vmax_sigma ** 2)
                        ))
                        K_a = pyro.sample("K_a", dist.Gamma(
                            ((K_max_tensor / 2) ** 2) / (K_sigma ** 2),
                            (K_max_tensor / 2) / (K_sigma ** 2)
                        ))

                # Sample all required parameters (additive_hill and nested_hill need second set)
                if function_type in ['additive_hill', 'nested_hill']:
                    beta = pyro.sample("beta", alpha_dist(temperature=temperature, logits=p_n_logits_tensor))

                    # n_b: per-category for multinomial, single for others
                    if distribution == 'multinomial' and K is not None:
                        K_minus_1 = K - 1
                        with pyro.plate("category_plate_n_b", K_minus_1, dim=-1):
                            n_b_raw = pyro.sample("n_b_raw", dist.Normal(n_mu_tensor, sigma_n_b))  # [T, K-1]
                            BOX_LOW  = self._t(-20.0)
                            BOX_HIGH = self._t( 20.0)
                            low  = torch.maximum(nmin, BOX_LOW)
                            high = torch.minimum(nmax, BOX_HIGH)
                            n_b = pyro.deterministic(
                                "n_b",
                                (beta.unsqueeze(-1) * n_b_raw).clamp(min=low.item(), max=high.item())
                            )  # [T, 1] * [T, K-1] -> [T, K-1]
                    else:
                        n_b_raw = pyro.sample("n_b_raw", dist.Normal(n_mu_tensor, sigma_n_b))
                        BOX_LOW  = self._t(-20.0)
                        BOX_HIGH = self._t( 20.0)
                        low  = torch.maximum(nmin, BOX_LOW)
                        high = torch.minimum(nmax, BOX_HIGH)
                        n_b = pyro.deterministic(
                            "n_b",
                            (beta * n_b_raw).clamp(min=low.item(), max=high.item())
                        )

                    # Vmax_b and K_b: same structure as Vmax_a and K_a
                    if distribution in ['binomial', 'multinomial']:
                        if distribution == 'multinomial' and K is not None:
                            # For multinomial: Sample Vmax_b for K-1 categories (Kth is residual)
                            K_minus_1 = K - 1
                            with pyro.plate("category_plate_Vmax_b", K_minus_1, dim=-1):
                                Vmax_b = pyro.sample("Vmax_b", dist.Beta(alpha_vmax, beta_vmax))  # [T, K-1]

                            # K_b: Log-Normal for K-1 categories
                            with pyro.plate("category_plate_K_b", K_minus_1, dim=-1):
                                log_K_b = pyro.sample("log_K_b", dist.Normal(K_log_mu, K_log_sigma))  # [T, K-1]
                                K_b = pyro.deterministic("K_b", torch.exp(log_K_b))  # [T, K-1]
                        else:
                            # For binomial: per-feature Vmax_b and K_b
                            Vmax_b = pyro.sample("Vmax_b", dist.Beta(alpha_vmax, beta_vmax))  # [T]
                            log_K_b = pyro.sample("log_K_b", dist.Normal(K_log_mu, K_log_sigma))  # [T]
                            K_b = pyro.deterministic("K_b", torch.exp(log_K_b))  # [T]

                    else:
                        # For negbinom/normal/studentt: Choose between Gamma and Log-Normal priors
                        # K_sigma and Vmax_sigma were computed above in the Vmax_a/K_a section
                        if use_lognormal_priors:
                            # Log-Normal priors (experimental) - Vmax_log_mu/sigma computed in Vmax_a section
                            log_Vmax_b = pyro.sample("log_Vmax_b", dist.Normal(Vmax_log_mu, Vmax_log_sigma))
                            Vmax_b = pyro.deterministic("Vmax_b", torch.exp(log_Vmax_b))

                            log_K_b = pyro.sample("log_K_b", dist.Normal(K_log_mu, K_log_sigma))
                            K_b = pyro.deterministic("K_b", torch.exp(log_K_b))
                        else:
                            # Gamma priors (default, matching archive) - better convergence and sparsity
                            Vmax_b = pyro.sample("Vmax_b", dist.Gamma(
                                (Vmax_prior_mean ** 2) / (Vmax_sigma ** 2),
                                Vmax_prior_mean / (Vmax_sigma ** 2)
                            ))
                            K_b = pyro.sample("K_b", dist.Gamma(
                                ((K_max_tensor / 2) ** 2) / (K_sigma ** 2),
                                (K_max_tensor / 2) / (K_sigma ** 2)
                            ))
                
                # Compute Hill function(s)
                # Hill_based_positive returns values in [0, Vmax]
                # We compute: y = A + alpha * Hill + beta * Hill (for additive)

                if distribution == 'multinomial' and K is not None:
                    # NEW FORMULATION for multinomial (matching binomial structure):
                    # Sample A from Dirichlet (sum to 1)
                    # Sample Vmax_a and Vmax_b INDEPENDENTLY for K-1 categories (Kth is residual)
                    # For each category k in K-1:
                    #   y_k = A_k + (alpha * Hill_a_k(Vmax=Vmax_a_k)) + (beta * Hill_b_k(Vmax=Vmax_b_k))
                    # Then: y_K = 1 - sum(y_1, ..., y_{K-1})
                    # This ensures probabilities sum to 1, and Kth category gets whatever is left

                    K_dim = K
                    K_minus_1 = K - 1
                    x_expanded = x_true.unsqueeze(-1)  # [N, 1]

                    # A is [T, K] from Dirichlet
                    # Extract K-1 for fitting Hills (Kth doesn't get a Hill function)
                    A_kminus1 = A[..., :K_minus_1]  # [T, K-1]

                    # Expand for broadcasting
                    A_kminus1_expanded = A_kminus1.unsqueeze(0)  # [1, T, K-1]

                    if function_type == 'single_hill':
                        # Compute Hills for K-1 categories using Vmax_a directly
                        Hilla_list = []
                        for k in range(K_minus_1):
                            hill_k = Hill_based_positive(x_expanded, Vmax=Vmax_a[:, k], A=0,
                                                        K=K_a[:, k], n=n_a[:, k],  # [T]
                                                        epsilon=epsilon_tensor)  # [N, T]
                            Hilla_list.append(hill_k.unsqueeze(-1))  # [N, T, 1]
                        Hilla_kminus1 = torch.cat(Hilla_list, dim=-1)  # [N, T, K-1]

                        # Combine: y = A + alpha * Hill_a(Vmax=Vmax_a)
                        combined_hill = alpha.unsqueeze(0).unsqueeze(-1) * Hilla_kminus1  # [N, T, K-1]
                        y_kminus1 = A_kminus1_expanded + combined_hill  # [N, T, K-1]

                    elif function_type == 'additive_hill':
                        # Compute Hills for K-1 categories using Vmax_a and Vmax_b directly
                        Hilla_list = []
                        Hillb_list = []
                        for k in range(K_minus_1):
                            hill_a_k = Hill_based_positive(x_expanded, Vmax=Vmax_a[:, k], A=0,
                                                          K=K_a[:, k], n=n_a[:, k],  # [T]
                                                          epsilon=epsilon_tensor)  # [N, T]
                            hill_b_k = Hill_based_positive(x_expanded, Vmax=Vmax_b[:, k], A=0,
                                                          K=K_b[:, k], n=n_b[:, k],  # [T]
                                                          epsilon=epsilon_tensor)  # [N, T]
                            Hilla_list.append(hill_a_k.unsqueeze(-1))  # [N, T, 1]
                            Hillb_list.append(hill_b_k.unsqueeze(-1))  # [N, T, 1]
                        Hilla_kminus1 = torch.cat(Hilla_list, dim=-1)  # [N, T, K-1]
                        Hillb_kminus1 = torch.cat(Hillb_list, dim=-1)  # [N, T, K-1]

                        # Combine: y = A + (alpha * Hill_a) + (beta * Hill_b)
                        combined_hill = (alpha.unsqueeze(0).unsqueeze(-1) * Hilla_kminus1 +
                                       beta.unsqueeze(0).unsqueeze(-1) * Hillb_kminus1)  # [N, T, K-1]
                        y_kminus1 = A_kminus1_expanded + combined_hill  # [N, T, K-1]

                    elif function_type == 'nested_hill':
                        # Compute nested Hills for K-1 categories using Vmax_a and Vmax_b directly
                        Hillb_list = []
                        for k in range(K_minus_1):
                            hill_a_k = Hill_based_positive(x_expanded, Vmax=Vmax_a[:, k], A=0,
                                                          K=K_a[:, k], n=n_a[:, k],
                                                          epsilon=epsilon_tensor)  # [N, T]
                            hill_b_k = Hill_based_positive(hill_a_k.unsqueeze(-1), Vmax=Vmax_b[:, k], A=0,
                                                          K=K_b[:, k], n=n_b[:, k],
                                                          epsilon=epsilon_tensor)  # [N, T]
                            Hillb_list.append(hill_b_k.unsqueeze(-1))  # [N, T, 1]
                        Hillb_kminus1 = torch.cat(Hillb_list, dim=-1)  # [N, T, K-1]

                        combined_hill = alpha.unsqueeze(0).unsqueeze(-1) * Hillb_kminus1  # [N, T, K-1]
                        y_kminus1 = A_kminus1_expanded + combined_hill  # [N, T, K-1]

                    # Clamp K-1 probabilities to [epsilon, 1-epsilon] to ensure valid residual
                    y_kminus1 = torch.clamp(y_kminus1, min=epsilon_tensor, max=1.0 - epsilon_tensor)

                    # Ensure sum of K-1 doesn't exceed 1
                    sum_kminus1 = y_kminus1.sum(dim=-1, keepdim=True)  # [N, T, 1]
                    # If sum > (1-epsilon), rescale proportionally
                    y_kminus1 = torch.where(
                        sum_kminus1 > (1.0 - epsilon_tensor),
                        y_kminus1 * (1.0 - epsilon_tensor) / sum_kminus1,
                        y_kminus1
                    )

                    # Compute Kth category as residual: y_K = 1 - sum(y_1, ..., y_{K-1})
                    y_K = 1.0 - y_kminus1.sum(dim=-1, keepdim=True)  # [N, T, 1]
                    y_K = torch.clamp(y_K, min=epsilon_tensor, max=1.0 - epsilon_tensor)

                    # Concatenate to get all K probabilities
                    y_dose_response = torch.cat([y_kminus1, y_K], dim=-1)  # [N, T, K]

                else:
                    # For non-multinomial: standard Hill computation
                    if distribution == 'binomial':
                        # For binomial: Use Vmax_a and Vmax_b directly (sampled independently)
                        # y = A + (alpha * Hill_a(Vmax=Vmax_a)) + (beta * Hill_b(Vmax=Vmax_b))
                        # Note: Vmax_a + Vmax_b CAN exceed 1 - we'll try without clamp first

                        if function_type == 'single_hill':
                            Hilla = Hill_based_positive(x_true.unsqueeze(-1), Vmax=Vmax_a, A=0, K=K_a, n=n_a, epsilon=epsilon_tensor)
                            y_dose_response = A + (alpha * Hilla)

                        elif function_type == 'additive_hill':
                            Hilla = Hill_based_positive(x_true.unsqueeze(-1), Vmax=Vmax_a, A=0, K=K_a, n=n_a, epsilon=epsilon_tensor)
                            Hillb = Hill_based_positive(x_true.unsqueeze(-1), Vmax=Vmax_b, A=0, K=K_b, n=n_b, epsilon=epsilon_tensor)
                            y_dose_response = A + (alpha * Hilla) + (beta * Hillb)

                        elif function_type == 'nested_hill':
                            Hilla = Hill_based_positive(x_true.unsqueeze(-1), Vmax=Vmax_a, A=0, K=K_a, n=n_a, epsilon=epsilon_tensor)
                            Hillb = Hill_based_positive(Hilla, Vmax=Vmax_b, A=0, K=K_b, n=n_b, epsilon=epsilon_tensor)
                            y_dose_response = A + (alpha * Hillb)

                        # OPTIONAL CLAMP (commented out for now - try without first):
                        # If Vmax_a + Vmax_b > 1 causes issues, uncomment:
                        # y_dose_response = torch.clamp(y_dose_response, min=epsilon_tensor, max=1.0 - epsilon_tensor)

                    else:
                        # For negbinom/normal/studentt: use standard formulation with learned Vmax_a/Vmax_b
                        if function_type == 'single_hill':
                            if use_lognormal_priors:
                                Hilla = Hill_based_positive_logK(x_true.unsqueeze(-1), Vmax=Vmax_a, A=0, logK=log_K_a, n=n_a)
                            else:
                                Hilla = Hill_based_positive(x_true.unsqueeze(-1), Vmax=Vmax_a, A=0, K=K_a, n=n_a, epsilon=epsilon_tensor)
                            y_dose_response = A + (alpha * Hilla)
                        elif function_type == 'additive_hill':
                            if use_lognormal_priors:
                                Hilla = Hill_based_positive_logK(x_true.unsqueeze(-1), Vmax=Vmax_a, A=0, logK=log_K_a, n=n_a)
                                Hillb = Hill_based_positive_logK(x_true.unsqueeze(-1), Vmax=Vmax_b, A=0, logK=log_K_b, n=n_b)
                            else:
                                Hilla = Hill_based_positive(x_true.unsqueeze(-1), Vmax=Vmax_a, A=0, K=K_a, n=n_a, epsilon=epsilon_tensor)
                                Hillb = Hill_based_positive(x_true.unsqueeze(-1), Vmax=Vmax_b, A=0, K=K_b, n=n_b, epsilon=epsilon_tensor)
                            y_dose_response = A + (alpha * Hilla) + (beta * Hillb)
                        elif function_type == 'nested_hill':
                            if use_lognormal_priors:
                                Hilla = Hill_based_positive_logK(x_true.unsqueeze(-1), Vmax=Vmax_a, A=0, logK=log_K_a, n=n_a)
                                Hillb = Hill_based_positive_logK(Hilla, Vmax=Vmax_b, A=0, logK=log_K_b, n=n_b)
                            else:
                                Hilla = Hill_based_positive(x_true.unsqueeze(-1), Vmax=Vmax_a, A=0, K=K_a, n=n_a, epsilon=epsilon_tensor)
                                Hillb = Hill_based_positive(Hilla, Vmax=Vmax_b, A=0, K=K_b, n=n_b, epsilon=epsilon_tensor)
                            y_dose_response = A + (alpha * Hillb)

            elif function_type == 'polynomial':
                assert polynomial_degree is not None and polynomial_degree >= 1, \
                    "polynomial_degree must be ≥ 1 (no intercept, A is handled separately)"

                if distribution == 'multinomial' and K is not None:
                    # For multinomial: fit K independent polynomials (one per category) in logit space
                    # Then apply softmax to get K probabilities that sum to 1
                    # This is more standard than K-1 with residual for unbounded logit space

                    # Sample per-category polynomial coefficients
                    coeffs_per_category = []
                    for d in range(1, polynomial_degree + 1):
                        with pyro.plate(f"poly_category_plate_deg{d}", K, dim=-2):
                            coeff = pyro.sample(f"poly_coeff_{d}", dist.Normal(0., sigma_coeff))  # [K, T]
                            coeffs_per_category.append(coeff)
                    coeffs = torch.stack(coeffs_per_category, dim=-3)  # [degree, K, T]

                    # Compute polynomial for each category
                    # coeffs: [degree, K, T]
                    # Need to permute to [degree, T, K] for Polynomial_function to work correctly
                    coeffs_permuted = coeffs.permute(0, 2, 1)  # [degree, T, K]
                    poly_val = Polynomial_function(x_true_sample, coeffs_permuted)  # [N, T, K]

                    # A is baseline logit for each category (need K logits)
                    # Sample K baseline logits (unbounded)
                    with pyro.plate("category_plate_A", K, dim=-2):
                        A_clamped = torch.clamp(A.unsqueeze(-2), min=epsilon_tensor, max=1.0 - epsilon_tensor)  # [1, T]
                        logit_A_transpose = torch.log(A_clamped) - torch.log(1 - A_clamped)  # [K, T] logits
                    logit_A = logit_A_transpose.transpose(-1, -2)  # [T, K]

                    # Compute logits for each category
                    logits_K = logit_A.unsqueeze(0) + alpha.unsqueeze(0).unsqueeze(-1) * poly_val  # [N, T, K]

                    # Apply softmax to get probabilities that sum to 1
                    y_dose_response = torch.softmax(logits_K, dim=-1)  # [N, T, K]

                else:
                    # For non-multinomial: single set of coefficients per feature
                    # Stack polynomial coefficients
                    coeffs = []
                    for d in range(1, polynomial_degree + 1):  # start at degree 1, no intercept
                        coeff = pyro.sample(f"poly_coeff_{d}", dist.Normal(0., sigma_coeff))
                        coeffs.append(coeff)
                    coeffs = torch.stack(coeffs, dim=-2)  # [degree, T]
                    if (coeffs.shape[1] == 1) & (coeffs.ndim == 4):
                        coeffs = coeffs.squeeze(1)        # [S, D, T]

                    # Distribution-specific polynomial computation:
                    if distribution in ['normal', 'studentt']:
                        # For continuous distributions: work in natural space (no logs!)
                        # Polynomial is applied to x_true directly: y = A + alpha * poly(x)
                        poly_val = Polynomial_function(x_true_sample, coeffs)  # [N, T]
                        y_dose_response = A + alpha * poly_val  # [N, T] - can be negative!

                    elif distribution in ['negbinom']:
                        # For negbinom: work in log space
                        # Polynomial is applied to log2(x): log2(y) = log2(A) + alpha * poly(log2(x))
                        log2_x_true = torch.log2(x_true_sample)  # [N]
                        poly_val = Polynomial_function(log2_x_true, coeffs)  # [N, T]
                        log2_y_dose_response = torch.log2(A) + alpha * poly_val  # [N, T]
                        y_dose_response = 2 ** log2_y_dose_response  # Convert back to count space

                    elif distribution == 'binomial':
                        # For binomial: work in LOGIT space (unbounded: -inf to +inf)
                        # A is in [0,1] from Beta prior, convert to logit
                        # Polynomial is applied to x_true: logit(p) = logit(A) + alpha * poly(x)
                        A_clamped = torch.clamp(A, min=epsilon_tensor, max=1.0 - epsilon_tensor)
                        logit_A = torch.log(A_clamped) - torch.log(1 - A_clamped)  # logit(A)
                        poly_val = Polynomial_function(x_true_sample, coeffs)  # [N, T]
                        logit_p = logit_A + alpha * poly_val  # [N, T] - logit space (unbounded)
                        # Convert back to probability space for sampler
                        y_dose_response = torch.sigmoid(logit_p)  # [N, T] in [0, 1]

                    else:
                        raise ValueError(f"Unknown distribution for polynomial: {distribution}")
            else:
                raise ValueError(f"Unknown function_type: {function_type}")
            

        ##########################
        ## Cell-level variables ##
        ##########################
        # At this point, y_dose_response contains the dose-response function output.
        # We need to transform it to the format expected by samplers.
        #
        # IMPORTANT: Samplers in distributions.py handle technical group effects themselves!
        # Do NOT apply alpha_y here - pass it to the sampler via alpha_y_full.

        # Prepare alpha_y_full (full C technical groups, including reference)
        # This will be passed to samplers, which apply technical groups themselves
        # Multinomial now supported (additive on logit scale, like binomial)
        if alpha_y is not None and groups_tensor is not None:
            # CRITICAL: Check if alpha_y already includes reference group
            # If alpha_y.shape[0 or 1] == C, it already has reference, don't add it!

            if alpha_y.dim() == 4:  # Predictive multinomial: Could be (S, C-1, T, K) or (S, C, T, K)
                if alpha_y.shape[1] == C:
                    # Already includes reference
                    alpha_y_full = alpha_y
                else:
                    # Need to add reference (zeros for additive on logit scale)
                    baseline_shape = (alpha_y.shape[0], 1, alpha_y.shape[2], alpha_y.shape[3])
                    baseline = torch.zeros(baseline_shape, device=self.model.device)
                    alpha_y_full = torch.cat([baseline, alpha_y], dim=1)

            elif alpha_y.dim() == 3:
                # Could be Predictive 2D (S, C-1, T) or Training multinomial (C-1, T, K) or (C, T, K)
                # Check if last dimension matches K (multinomial) or T (2D predictive)
                if distribution == 'multinomial' and K is not None and alpha_y.shape[-1] == K:
                    # Training multinomial: (C-1, T, K) or (C, T, K)
                    if alpha_y.shape[0] == C:
                        # Already includes reference
                        alpha_y_full = alpha_y
                    else:
                        # Need to add reference (zeros for additive on logit scale)
                        baseline_shape = (1, alpha_y.shape[1], alpha_y.shape[2])
                        baseline = torch.zeros(baseline_shape, device=self.model.device)
                        alpha_y_full = torch.cat([baseline, alpha_y], dim=0)
                else:
                    # Predictive 2D: (S, C-1, T) or (S, C, T)
                    if alpha_y.shape[1] == C:
                        # Already includes reference
                        alpha_y_full = alpha_y
                    else:
                        # Need to add reference (zeros for additive, ones for multiplicative)
                        ones_shape = (alpha_y.shape[0], 1, T)
                        if distribution == 'negbinom':
                            baseline = torch.ones(ones_shape, device=self.model.device)
                        else:
                            baseline = torch.zeros(ones_shape, device=self.model.device)
                        alpha_y_full = torch.cat([baseline, alpha_y], dim=1)

            elif alpha_y.dim() == 2:  # Training 2D: Could be (C-1, T) or (C, T)
                if alpha_y.shape[0] == C:
                    # Already includes reference
                    alpha_y_full = alpha_y
                else:
                    # Need to add reference (zeros for additive, ones for multiplicative)
                    ones_shape = (1, T)
                    if distribution == 'negbinom':
                        baseline = torch.ones(ones_shape, device=self.model.device)
                    else:
                        baseline = torch.zeros(ones_shape, device=self.model.device)
                    alpha_y_full = torch.cat([baseline, alpha_y], dim=0)
            else:
                raise ValueError(f"Unexpected alpha_y shape: {alpha_y.shape}")
        else:
            alpha_y_full = None

        # Transform dose-response to sampler-expected format
        # (samplers will handle technical groups and normalization)
        if distribution == 'negbinom':
            # Sampler expects: mu_y = dose-response in count space (NO technical groups, NO sum factors)
            # Sampler will apply: mu_final = mu_y * alpha_y * sum_factor
            mu_y = y_dose_response  # [N, T] - just the dose-response

        elif distribution == 'binomial':
            # Sampler expects: mu_y = probability in [0, 1]
            # y_dose_response should already be a probability from Hill/polynomial
            # Sampler will apply technical groups on logit scale
            mu_y = y_dose_response  # [N, T] - already probability

        elif distribution == 'multinomial':
            # Sampler expects: mu_y = baseline probabilities [N, T, K]
            # y_dose_response is already [N, T, K] probabilities that sum to 1
            # Sampler will apply technical groups on logit scale per category
            mu_y = y_dose_response  # [N, T, K] - probabilities per category

        elif distribution in ['normal', 'studentt']:
            # Sampler expects: mu_y = natural value space
            # Sampler will apply technical groups additively
            mu_y = y_dose_response  # [N, T] - can be negative!

        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        # Debug checks (keep for troubleshooting)
        if torch.isnan(mu_y).any() or torch.isinf(mu_y).any():
            check_tensor("mu_y", mu_y)
            check_tensor("y_dose_response", y_dose_response)
            check_tensor("sum_factor_tensor", sum_factor_tensor)
            check_tensor("phi_y_used", phi_y_used)
            check_tensor("A", A)
            if function_type in ['single_hill', 'additive_hill', 'nested_hill']:
                check_tensor("n_a", n_a)
                check_tensor("Vmax_a", Vmax_a)
                check_tensor("K_a", K_a)
            if function_type in ['additive_hill', 'nested_hill']:
                check_tensor("n_b", n_b)
                check_tensor("Vmax_b", Vmax_b)
                check_tensor("K_b", K_b)
            if function_type == 'polynomial':
                check_tensor("coeffs", coeffs)

        # Call distribution-specific observation sampler
        from .distributions import get_observation_sampler
        observation_sampler = get_observation_sampler(distribution, 'trans')

        # Call the appropriate sampler based on distribution
        if distribution == 'negbinom':
            observation_sampler(
                y_obs_tensor=y_obs_tensor,
                mu_y=mu_y,
                phi_y_used=phi_y_used,
                alpha_y_full=alpha_y_full,
                groups_tensor=groups_tensor,
                sum_factor_tensor=sum_factor_tensor,
                N=N,
                T=T,
                C=C,
                use_epsilon=use_epsilon
            )
        elif distribution == 'multinomial':
            # For multinomial, mu_y should be probabilities [N, T, K]
            observation_sampler(
                y_obs_tensor=y_obs_tensor,
                mu_y=mu_y,  # Should be [N, T, K] probabilities
                alpha_y_full=alpha_y_full,  # [C, T, K] or [S, C, T, K] - additive on logit scale
                groups_tensor=groups_tensor,
                N=N,
                T=T,
                K=K,
                C=C
            )
        elif distribution == 'binomial':
            observation_sampler(
                y_obs_tensor=y_obs_tensor,
                denominator_tensor=denominator_tensor,
                mu_y=mu_y,  # Should be probabilities [N, T]
                alpha_y_full=alpha_y_full,
                groups_tensor=groups_tensor,
                N=N,
                T=T,
                C=C
            )
        elif distribution == 'normal':
            # For normal, we need sigma_y (standard deviation)
            sigma_y = 1.0 / torch.sqrt(phi_y)  # Convert from precision to std dev
            observation_sampler(
                y_obs_tensor=y_obs_tensor,
                mu_y=mu_y,
                sigma_y=sigma_y,
                alpha_y_full=alpha_y_full,
                groups_tensor=groups_tensor,
                N=N,
                T=T,
                C=C
            )
        elif distribution == 'studentt':
            # For studentt, we need sigma_y (standard deviation) and nu_y (degrees of freedom)
            sigma_y = 1.0 / torch.sqrt(phi_y)  # Convert from precision to std dev
            observation_sampler(
                y_obs_tensor=y_obs_tensor,
                mu_y=mu_y,
                sigma_y=sigma_y,
                nu_y=nu_y,
                alpha_y_full=alpha_y_full,
                groups_tensor=groups_tensor,
                N=N,
                T=T,
                C=C
            )
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

    ########################################################
    # Step 3b: logistic_hill trans model (model_y_logistic)
    ########################################################
    def _model_y_logistic(
        self,
        N,
        T,
        J,
        y_obs_tensor,
        u_tensor,
        sum_factor_tensor,
        b0_tensor,
        b_scale_tensor,
        data_scale_tensor,
        tau0,
        slab_scale,
        k_scale,
        log_n_loc,
        log_n_scale,
        beta_o_alpha_tensor,
        beta_o_beta_tensor,
        u_center,
        u_scale,
        alpha_y_full=None,
        groups_tensor=None,
        C=None,
        distribution='negbinom',
        denominator_tensor=None,
        use_epsilon=False,
    ):
        """
        Trans model with a sum of logistic curves in standardised log2 x.

        With u = (log2 x - u_center) / u_scale (u = 0 at the NTC level), the linear
        predictor for feature t is

            eta_t(u) = b_t + sum_j V_jt [ sigmoid(n_jt (u - k_jt)) - sigmoid(-n_jt k_jt) ],

        so b_t is the value at the NTC level. The link is log2 (negbinom), logit
        (binomial) or identity (normal, studentt). Each logistic term is a Hill
        function in x with Hill coefficient n / (u_scale ln 2) and log2 K = u_center + u_scale k.

        Priors:
            b_t      ~ Normal(b0_t, b_scale_t)
            V_jt     = s_t tau lambda~_jt z_jt     (regularised horseshoe, signed amplitude)
            z_jt     ~ Normal(0, 1),  lambda_jt ~ HalfCauchy(1),  tau ~ HalfCauchy(tau0)
            lambda~  = c lambda / sqrt(c^2 + tau^2 lambda^2)   (slab scale c)
            log n_jt ~ Normal(log_n_loc, log_n_scale)          (n > 0, sign carried by V)
            k_1t ~ Normal(0, k_scale),  k_jt = k_(j-1)t + softplus(r_jt), r_jt ~ Normal(0, 1)

        s_t is 1 for negbinom/binomial and the per-feature data SD for normal/studentt.
        """
        trans_plate = pyro.plate("trans_plate", T, dim=-1)
        component_plate = pyro.plate("logistic_component_plate", J, dim=-2)

        tau = pyro.sample("tau", dist.HalfCauchy(self._t(tau0)))
        if distribution == 'negbinom':
            beta_o = pyro.sample("beta_o", dist.Gamma(beta_o_alpha_tensor, beta_o_beta_tensor))

        with trans_plate:
            b = pyro.sample("b", dist.Normal(b0_tensor, b_scale_tensor))
            if distribution == 'negbinom':
                o_y = pyro.sample("o_y", dist.Exponential(beta_o))
                phi_y = 1.0 / (o_y ** 2)
            elif distribution in ('normal', 'studentt'):
                sigma_y = pyro.sample("sigma_y", dist.HalfNormal(2.0 * data_scale_tensor))
                if distribution == 'studentt':
                    nu_y = pyro.sample("nu_y", dist.Gamma(self._t(10.0), self._t(2.0)))
            with component_plate:
                z = pyro.sample("z", dist.Normal(self._t(0.0), self._t(1.0)))
                lam = pyro.sample("lam", dist.HalfCauchy(self._t(1.0)))
                log_n = pyro.sample("log_n", dist.Normal(self._t(log_n_loc), self._t(log_n_scale)))
                k_raw = pyro.sample("k_raw", dist.Normal(self._t(0.0), self._t(1.0)))

        c2 = self._t(slab_scale) ** 2
        lam_reg = torch.sqrt(c2 * lam ** 2 / (c2 + tau ** 2 * lam ** 2))
        V = pyro.deterministic("V", data_scale_tensor * tau * lam_reg * z)          # [J, T]
        k = pyro.deterministic("k", torch.cumsum(
            torch.cat([k_scale * k_raw[..., :1, :],
                       torch.nn.functional.softplus(k_raw[..., 1:, :])], dim=-2), dim=-2))
        n = pyro.deterministic("n", torch.exp(log_n))
        # Interpretable Hill parameters on the original x scale
        pyro.deterministic("log2_K", u_center + u_scale * k)
        pyro.deterministic("hill_n", n / (u_scale * np.log(2.0)))

        u = u_tensor.view(-1, 1, 1)                                                  # [N, 1, 1]
        g = (V * (torch.sigmoid(n * (u - k)) - torch.sigmoid(-n * k))).sum(dim=-2)   # [N, T]
        eta = b + g                                                                  # [N, T]

        from .distributions import get_observation_sampler
        observation_sampler = get_observation_sampler(distribution, 'trans')
        if distribution == 'negbinom':
            observation_sampler(
                y_obs_tensor=y_obs_tensor, mu_y=torch.exp2(eta), phi_y_used=phi_y.unsqueeze(-2),
                alpha_y_full=alpha_y_full, groups_tensor=groups_tensor,
                sum_factor_tensor=sum_factor_tensor, N=N, T=T, C=C, use_epsilon=use_epsilon,
            )
        elif distribution == 'binomial':
            observation_sampler(
                y_obs_tensor=y_obs_tensor, denominator_tensor=denominator_tensor,
                mu_y=torch.sigmoid(eta), alpha_y_full=alpha_y_full,
                groups_tensor=groups_tensor, N=N, T=T, C=C,
            )
        elif distribution == 'normal':
            observation_sampler(
                y_obs_tensor=y_obs_tensor, mu_y=eta, sigma_y=sigma_y,
                alpha_y_full=alpha_y_full, groups_tensor=groups_tensor, N=N, T=T, C=C,
            )
        elif distribution == 'studentt':
            observation_sampler(
                y_obs_tensor=y_obs_tensor, mu_y=eta, sigma_y=sigma_y, nu_y=nu_y,
                alpha_y_full=alpha_y_full, groups_tensor=groups_tensor, N=N, T=T, C=C,
            )
        else:
            raise ValueError(f"logistic_hill does not support distribution '{distribution}'")

    def _alpha_y_with_reference(self, alpha_y, C, T, distribution):
        """Return alpha_y with the reference group (row 0) included, shape [C, T]."""
        if alpha_y is None:
            return None
        if alpha_y.shape[0] == C:
            return alpha_y
        ref_value = 1.0 if distribution == 'negbinom' else 0.0
        ref = torch.full((1, T), ref_value, device=alpha_y.device, dtype=alpha_y.dtype)
        return torch.cat([ref, alpha_y], dim=0)

    def _fit_trans_logistic(
        self,
        N,
        T,
        y_obs_tensor,
        y_obs_for_prior,
        sum_factor_tensor,
        denominator_tensor,
        alpha_y_prefit,
        groups_tensor,
        C,
        ntc_mask,
        distribution,
        niters,
        lr,
        nsamples,
        tolerance,
        beta_o_alpha_tensor,
        beta_o_beta_tensor,
        use_epsilon,
        n_components=2,
        logistic_guide='block_mvn',
        horseshoe_tau0=None,
        horseshoe_slab_scale=None,
    ):
        """Fit the logistic_hill trans model with SVI and return posterior samples."""
        from torch.distributions import constraints
        from pyro.infer.autoguide import AutoNormal
        from pyro.infer.autoguide.initialization import init_to_median, init_to_value

        if distribution not in ('negbinom', 'binomial', 'normal', 'studentt'):
            raise NotImplementedError(
                f"function_type='logistic_hill' supports negbinom, binomial, normal and studentt, not '{distribution}'"
            )
        if logistic_guide not in ('block_mvn', 'normal'):
            raise ValueError("logistic_guide must be 'block_mvn' or 'normal'")
        J = int(n_components)
        device = self.model.device
        pyro.clear_param_store()

        # ---------------------------------------------
        # Standardised input u = (log2 x - center) / scale
        # ---------------------------------------------
        log2_x = self.model.log2_x_true.to(device)
        log2_x_point = log2_x.mean(dim=0) if self.model.log2_x_true_type == 'posterior' else log2_x
        ntc_mask_t = torch.tensor(np.asarray(ntc_mask, dtype=bool), device=device)
        if ntc_mask_t.sum() == 0:
            warnings.warn("logistic_hill: no NTC cells found; centring u on the mean over all cells.")
            ntc_mask_t = torch.ones(N, dtype=torch.bool, device=device)
        u_center = log2_x_point[ntc_mask_t].mean().item()
        u_scale = max(log2_x_point.std().item(), 1e-3)

        def to_u(l2x):
            return (l2x - u_center) / u_scale

        # ---------------------------------------------
        # Baseline (link scale) and data scale per feature
        # ---------------------------------------------
        y_ntc = y_obs_for_prior[ntc_mask_t]
        if distribution == 'negbinom':
            y_bar = torch.nanmean(y_ntc, dim=0).clamp_min(1.0 / max(int(ntc_mask_t.sum()), 1))
            b0 = torch.log2(y_bar)
            b_scale = torch.ones(T, device=device)
            data_scale = torch.ones(T, device=device)
        elif distribution == 'binomial':
            p_bar = torch.nanmean(y_ntc, dim=0)
            p_bar = torch.where(torch.isfinite(p_bar), p_bar, torch.full_like(p_bar, 0.5)).clamp(1e-3, 1 - 1e-3)
            b0 = torch.log(p_bar) - torch.log1p(-p_bar)
            b_scale = torch.full((T,), 1.5, device=device)
            data_scale = torch.ones(T, device=device)
        else:
            b0 = torch.nanmean(y_ntc, dim=0)
            centred = y_obs_for_prior - torch.nanmean(y_obs_for_prior, dim=0)
            sd = torch.sqrt(torch.nanmean(centred ** 2, dim=0))
            data_scale = torch.where(torch.isfinite(sd) & (sd > 0), sd, torch.ones_like(sd)).clamp_min(1e-3)
            b0 = torch.where(torch.isfinite(b0), b0, torch.zeros_like(b0))
            b_scale = data_scale
        tau0 = 0.1 if horseshoe_tau0 is None else horseshoe_tau0
        slab_scale = 3.0 if horseshoe_slab_scale is None else horseshoe_slab_scale
        k_scale, log_n_loc, log_n_scale = 1.5, 0.5, 0.75

        # ---------------------------------------------
        # Technical group effects (fixed from fit_technical)
        # ---------------------------------------------
        alpha_is_posterior = alpha_y_prefit is not None and alpha_y_prefit.dim() >= 3
        alpha_point = None
        if alpha_y_prefit is not None and groups_tensor is not None:
            alpha_point = self._alpha_y_with_reference(
                alpha_y_prefit.mean(dim=0) if alpha_is_posterior else alpha_y_prefit, C, T, distribution)

        def model_args(u_tensor, alpha_y_full):
            return dict(
                N=N, T=T, J=J, y_obs_tensor=y_obs_tensor, u_tensor=u_tensor,
                sum_factor_tensor=sum_factor_tensor, b0_tensor=b0, b_scale_tensor=b_scale,
                data_scale_tensor=data_scale, tau0=tau0, slab_scale=slab_scale, k_scale=k_scale,
                log_n_loc=log_n_loc, log_n_scale=log_n_scale,
                beta_o_alpha_tensor=beta_o_alpha_tensor, beta_o_beta_tensor=beta_o_beta_tensor,
                u_center=u_center, u_scale=u_scale, alpha_y_full=alpha_y_full,
                groups_tensor=groups_tensor, C=C, distribution=distribution,
                denominator_tensor=denominator_tensor, use_epsilon=use_epsilon,
            )

        # ---------------------------------------------
        # Variational family
        # ---------------------------------------------
        feature_sites = [("b", False, constraints.real)]
        if distribution == 'negbinom':
            feature_sites.append(("o_y", False, constraints.positive))
        elif distribution in ('normal', 'studentt'):
            feature_sites.append(("sigma_y", False, constraints.positive))
            if distribution == 'studentt':
                feature_sites.append(("nu_y", False, constraints.positive))
        feature_sites += [
            ("z", True, constraints.real),
            ("lam", True, constraints.positive),
            ("log_n", True, constraints.real),
            ("k_raw", True, constraints.real),
        ]
        feature_site_names = [name for name, _, _ in feature_sites]
        global_site_names = ["tau"] + (["beta_o"] if distribution == 'negbinom' else [])

        mean_field = AutoNormal(
            self._model_y_logistic,
            init_loc_fn=init_to_value(values={"b": b0}, fallback=init_to_median),
        )

        def block_from_mean_field():
            """Full-rank per-feature guide, initialised at the current mean-field solution."""
            locs, scales = [], []
            for name, per_component, _ in feature_sites:
                loc = getattr(mean_field.locs, name).detach()
                scale = getattr(mean_field.scales, name).detach()
                if per_component:
                    locs.append(loc.T)                 # [J, T] -> [T, J]
                    scales.append(scale.T)
                else:
                    locs.append(loc.unsqueeze(-1))
                    scales.append(scale.unsqueeze(-1))
            global_specs = [(name, constraints.positive,
                             getattr(mean_field.locs, name).detach(),
                             getattr(mean_field.scales, name).detach()) for name in global_site_names]
            return _FeatureBlockGuide(T, J, feature_sites, torch.cat(locs, dim=-1),
                                      torch.cat(scales, dim=-1), global_specs)

        # The full-rank guide is warm-started from a mean-field fit: started cold it
        # tends to get stuck in poorer optima.
        if logistic_guide == 'block_mvn':
            n_mean_field = niters // 2
            phases = [('mean_field', n_mean_field), ('block_mvn', niters - n_mean_field)]
        else:
            phases = [('mean_field', niters)]

        if lr is None:
            lr = 1e-2
        lrd = 0.1 ** (1.0 / max(niters, 1))  # learning rate decays to 10 % over the run

        u_point = to_u(log2_x_point)
        x_is_posterior = self.model.log2_x_true_type == 'posterior' and log2_x.dim() >= 2

        print(f"[INFO] logistic_hill: J={J} components, guide={logistic_guide}, "
              f"u_center={u_center:.3f}, u_scale={u_scale:.3f}, tau0={tau0}, slab={slab_scale}")

        self.losses_trans = []
        window = 1000
        guide = mean_field
        failed = False
        for phase, phase_steps in phases:
            if failed or phase_steps == 0:
                break
            if phase == 'block_mvn':
                guide = block_from_mean_field()
            optimizer = pyro.optim.ClippedAdam({
                "lr": lr * lrd ** len(self.losses_trans), "clip_norm": 10.0, "lrd": lrd,
            })
            svi = pyro.infer.SVI(self._model_y_logistic, guide, optimizer, loss=pyro.infer.Trace_ELBO())
            phase_losses = []
            for step in range(phase_steps):
                u_step = u_point
                alpha_step = alpha_point
                if x_is_posterior:
                    u_step = to_u(log2_x[torch.randint(log2_x.shape[0], (1,)).item()])
                if alpha_is_posterior and groups_tensor is not None:
                    samp = torch.randint(alpha_y_prefit.shape[0], (1,)).item()
                    alpha_step = self._alpha_y_with_reference(alpha_y_prefit[samp], C, T, distribution)

                loss = svi.step(**model_args(u_step, alpha_step))
                if not np.isfinite(loss):
                    print(f"[WARNING] Non-finite loss at step {len(self.losses_trans)}; stopping optimisation.")
                    failed = True
                    break
                phase_losses.append(loss)
                self.losses_trans.append(loss)
                if step % 1000 == 0:
                    print(f"Step {len(self.losses_trans) - 1} ({phase}) : loss = {loss:.5e}")
                # Stop the phase when the mean loss over the last window improves on the
                # previous window by less than `tolerance` (relative).
                if (step + 1) % window == 0 and step + 1 >= 2 * window:
                    prev = np.mean(phase_losses[-2 * window:-window])
                    cur = np.mean(phase_losses[-window:])
                    if (prev - cur) / abs(prev) < tolerance:
                        print(f"[INFO] {phase} phase converged at step {len(self.losses_trans) - 1}; "
                              f"mean loss over last {window} steps = {cur:.5e}")
                        break

        return_sites = ["b", "tau", "z", "lam", "log_n", "k_raw", "V", "k", "n", "log2_K", "hill_n"]
        return_sites += [name for name in ("o_y", "sigma_y", "nu_y") if name in feature_site_names]
        if distribution == 'negbinom':
            return_sites.append("beta_o")
        predictive = pyro.infer.Predictive(
            self._model_y_logistic, guide=guide, num_samples=nsamples, return_sites=return_sites,
        )
        with torch.no_grad():
            posterior_samples_y = dict(predictive(**model_args(u_point, alpha_point)))
        # Predictive pads sample shapes with plate dimensions; normalise to [S], [S, T], [S, J, T]
        component_sites = {"z", "lam", "log_n", "k_raw", "V", "k", "n", "log2_K", "hill_n"}
        global_sites = {"tau", "beta_o"}
        for name, value in posterior_samples_y.items():
            if name in component_sites:
                posterior_samples_y[name] = value.reshape(nsamples, J, T)
            elif name in global_sites:
                posterior_samples_y[name] = value.reshape(nsamples)
            else:
                posterior_samples_y[name] = value.reshape(nsamples, T)
        if "o_y" in posterior_samples_y:
            posterior_samples_y["phi_y"] = 1.0 / posterior_samples_y["o_y"] ** 2
        posterior_samples_y = {k: self._to_cpu(v) for k, v in posterior_samples_y.items()}
        posterior_samples_y.update({
            "u_center": u_center,
            "u_scale": u_scale,
            "n_components": J,
            "link": {"negbinom": "log2", "binomial": "logit"}.get(distribution, "identity"),
        })
        return posterior_samples_y

    ########################################################
    # Step 3: Fit trans effects (model_y)
    ########################################################
    def fit_trans(
        self,
        sum_factor_col: str = None,
        function_type: str = 'single_hill',  # or 'additive', 'nested'
        use_posterior: bool = False,
        polynomial_degree: int = 6,
        lr: float = None,
        niters: int = None,
        nsamples: int = 1000,
        alpha_ewma: float = 0.05,
        tolerance: float = 1e-4, # recommended to keep based on cell2location
        beta_o_beta: float = 3, # recommended to keep based on cell2location
        beta_o_alpha: float = 9, # recommended to keep based on cell2location
        alpha_alpha_mu: float = 5.8,
        K_alpha: float = 2,
        Vmax_alpha: float = 2,
        n_mu: float = 0,
        p_n: float = 1e-6,
        epsilon: float = 1e-6,
        init_temp: float = 1.0,
        #final_temp: float = 1e-8,
        final_temp: float = 0.1,
        minibatch_size: int = None,
        distribution: str = None,
        denominator: np.ndarray = None,
        modality_name: str = None,
        min_denominator: int = None,
        use_data_driven_priors: bool = True,
        use_lognormal_priors: bool = True,
        correct_priors_for_technical: bool = True,
        use_archive_prior_computation: bool = False,
        use_epsilon: bool = False,
        n_components: int = 2,
        logistic_guide: str = 'block_mvn',
        horseshoe_tau0: float = None,
        horseshoe_slab_scale: float = None,
        **kwargs
    ):
        """
        Fit trans effects using distribution-specific likelihood.

        Parameters
        ----------
        modality_name : str, optional
            Name of modality to fit. If None, uses primary modality.
        distribution : str, optional
            Distribution type: 'negbinom', 'multinomial', 'binomial', 'normal'
            If None, auto-detected from modality.
        sum_factor_col : str, optional
            Sum factor column name. Required for negbinom, ignored for others.
        use_posterior : bool, default=False
            If False (default), converts alpha_y_prefit and x_true from prior fits to
            point estimates (mean over posterior samples) before fitting. This is faster
            and usually sufficient. If True, uses full posterior samples for uncertainty
            propagation (slower but more accurate).
        denominator : np.ndarray, optional
            Denominator array for binomial distribution (e.g., total counts for PSI)
            If None, auto-detected from modality.
        min_denominator : int, optional
            Minimum denominator value for binomial observations. Observations where
            denominator < min_denominator are masked (excluded from fitting).
            Useful for filtering low-coverage splicing junctions. Default: None (no filtering).
        use_data_driven_priors : bool, optional
            If True (default), use Beta priors for A and upper_limit based on data percentiles.
            If False, use uniform priors (Beta(1, 1)). Useful for testing if data-driven
            priors are too strong or causing issues. Default: True.
        use_lognormal_priors : bool, optional
            If True (default), use Log-Normal priors for Vmax and K.
            If False, use direct Gamma priors (archive behavior).
            Only affects negbinom/normal/studentt distributions.
        correct_priors_for_technical : bool, optional
            If True (default), correct data for technical effects before computing priors (Amean, Vmax_mean).
            If False (archive behavior), compute priors from raw sum_factor-normalized data.
            Technical effects are still corrected during model fitting via alpha_y_sample.
        use_archive_prior_computation : bool, optional
            If True, compute Amean and Vmax_mean using archive method:
            - Amean = min(guide_means) per feature
            - Vmax_mean = max(guide_means) per feature
            If False (default), use percentile-based method:
            - Amean = 5th percentile of guide means
            - Vmax_mean = 95th percentile - 5th percentile (range)
        use_epsilon : bool, optional
            If True, add 1e-8 epsilon for numerical stability in NegativeBinomial logits.
            If False (default), use log(mu) - log(phi) directly.
        function_type : str
            Dose-response function: 'single_hill', 'additive_hill', 'polynomial', 'logistic_hill'.
            'logistic_hill' models the link-scale mean (log2 for negbinom, logit for binomial,
            identity for normal/studentt) as a baseline plus n_components logistic curves in
            standardised log2 x, with horseshoe-shrunk signed amplitudes. See
            docs/LOGISTIC_HILL.md. Not available for multinomial.
        n_components : int, default=2
            logistic_hill only: number of logistic components (ordered by midpoint).
        logistic_guide : str, default='block_mvn'
            logistic_hill only: 'block_mvn' (full-covariance Gaussian per feature) or
            'normal' (mean-field AutoNormal).
        horseshoe_tau0 : float, optional
            logistic_hill only: scale of the HalfCauchy prior on the global shrinkage tau
            (default 0.1, in link units; for normal/studentt in units of the feature SD).
        horseshoe_slab_scale : float, optional
            logistic_hill only: slab scale of the regularised horseshoe (default 3.0).
        **kwargs
            Additional parameters for specific distributions

        Notes
        -----
        Each modality stores its own fitting results.
        Primary modality results are also stored at model level for backward compatibility.
        Trans fitting requires that technical fit has been performed for the modality.

        If technical_group_code is set (via set_technical_groups()), it will be used for
        correction. Otherwise, no group correction is applied.

        Examples
        --------
        >>> model.set_technical_groups(['cell_line'])  # Optional, for correction
        >>> model.fit_trans(sum_factor_col='sum_factor', function_type='additive_hill')

        >>> # Test without data-driven priors
        >>> model.fit_trans(sum_factor_col='sum_factor', function_type='additive_hill',
        ...                 use_data_driven_priors=False)
        """

        # Handle use_posterior flag - convert posteriors to point estimates if needed
        # Check for x_true (from cis fit)
        has_x_true = self.model.x_true is not None

        if not has_x_true:
            warnings.warn(
                "x_true not set. You should run fit_cis() before fit_trans(). "
                "Proceeding without cis effects."
            )

        if not use_posterior:
            # Convert x_true to point estimate if it's a posterior
            if has_x_true and getattr(self.model, 'x_true_type', None) == 'posterior':
                print("[INFO] use_posterior=False: Converting x_true to point estimate (mean)")
                self.model.x_true = self.model.x_true.mean(dim=0)
                self.model.x_true_type = 'point'

            # Convert log2_x_true to point estimate if it's a posterior
            if hasattr(self.model, 'log2_x_true') and self.model.log2_x_true is not None:
                if getattr(self.model, 'log2_x_true_type', None) == 'posterior':
                    print("[INFO] use_posterior=False: Converting log2_x_true to point estimate (mean)")
                    self.model.log2_x_true = self.model.log2_x_true.mean(dim=0)
                    self.model.log2_x_true_type = 'point'

        # Determine which modality to use
        if modality_name is None:
            modality_name = self.model.primary_modality
        modality = self.model.get_modality(modality_name)

        # Handle use_posterior flag for modality-specific alpha_y
        if not use_posterior:
            # Convert alpha_y_prefit_mult to point estimate if it's a posterior (negbinom)
            if hasattr(modality, 'alpha_y_prefit_mult') and modality.alpha_y_prefit_mult is not None:
                if getattr(modality, 'alpha_y_type', None) == 'posterior' and modality.alpha_y_prefit_mult.dim() >= 3:
                    print(f"[INFO] use_posterior=False: Converting {modality_name}.alpha_y_prefit_mult to point estimate (mean)")
                    modality.alpha_y_prefit_mult = modality.alpha_y_prefit_mult.mean(dim=0)
                    modality.alpha_y_type = 'point'

            # Convert alpha_y_prefit_add to point estimate if it's a posterior (normal, binomial, etc)
            if hasattr(modality, 'alpha_y_prefit_add') and modality.alpha_y_prefit_add is not None:
                if getattr(modality, 'alpha_y_type', None) == 'posterior' and modality.alpha_y_prefit_add.dim() >= 3:
                    print(f"[INFO] use_posterior=False: Converting {modality_name}.alpha_y_prefit_add to point estimate (mean)")
                    modality.alpha_y_prefit_add = modality.alpha_y_prefit_add.mean(dim=0)
                    modality.alpha_y_type = 'point'

        # Auto-detect distribution from modality
        if distribution is None:
            distribution = modality.distribution

        if function_type == 'logistic_hill' and distribution == 'multinomial':
            raise NotImplementedError("function_type='logistic_hill' is not implemented for multinomial")

        # Auto-detect denominator from modality (for binomial)
        if denominator is None and modality.denominator is not None:
            denominator = modality.denominator

        # ---------------------------
        # Set conditional default for niters
        # ---------------------------
        if niters is None:
            # Default: 100,000 unless multinomial OR polynomial function, then 200,000
            if distribution == 'multinomial':
                niters = 200_000
                print(f"[INFO] Using default niters=200,000 for multivariate distribution '{distribution}'")
            elif function_type == 'polynomial':
                niters = 200_000
                print(f"[INFO] Using default niters=200,000 for polynomial function")
            elif function_type == 'logistic_hill':
                niters = 20_000
                print(f"[INFO] Using default niters=20,000 for logistic_hill function")
            else:
                niters = 100_000
                print(f"[INFO] Using default niters=100,000 for distribution '{distribution}' and function_type '{function_type}'")
        
        #if lr is None:
        #    # Default: 100,000 unless multinomial OR polynomial function, then 200,000
        #    if distribution in ['binomial', 'multinomial']:
        #        lr = 1e-4
        #        print(f"[INFO] Using default niters=200,000 for distribution '{distribution}'")
        #    else:
        #        lr = 1e-3
        #        print(f"[INFO] Using default lr=1e-3 for distribution '{distribution}'")

        # Check that technical fit has been done for this modality
        if modality.alpha_y_prefit is None and 'technical_group_code' in self.model.meta.columns:
            raise ValueError(
                f"Modality '{modality_name}' has not been fit with fit_technical(). "
                f"Please run fit_technical(modality_name='{modality_name}') first."
            )

        # Get counts from modality
        counts_to_fit = modality.counts

        # Get cell names from modality
        if modality.cell_names is not None:
            modality_cells = modality.cell_names
        else:
            # Modality doesn't have cell names - assume same order as model.meta['cell']
            modality_cells = self.model.meta['cell'].values[:counts_to_fit.shape[modality.cells_axis]]

        # Get technical fit results from modality (NOT self.model.alpha_y_prefit!)
        # DEFENSIVE: Use distribution-specific attributes first (like plot_xy_data does)
        # This is more robust than the generic alpha_y_prefit
        alpha_y_prefit = None
        alpha_y_type = None

        if distribution == 'negbinom':
            # For negbinom, use multiplicative correction
            if hasattr(modality, 'alpha_y_prefit_mult') and modality.alpha_y_prefit_mult is not None:
                alpha_y_prefit = modality.alpha_y_prefit_mult
                alpha_y_type = getattr(modality, 'alpha_y_type', 'posterior')
                print(f"[INFO] Using distribution-specific alpha_y_prefit_mult for {distribution}")
        else:
            # For binomial, multinomial, normal, studentt: use additive correction
            if hasattr(modality, 'alpha_y_prefit_add') and modality.alpha_y_prefit_add is not None:
                alpha_y_prefit = modality.alpha_y_prefit_add
                alpha_y_type = getattr(modality, 'alpha_y_type', 'posterior')
                print(f"[INFO] Using distribution-specific alpha_y_prefit_add for {distribution}")

        # Fallback to generic attribute if distribution-specific not available
        if alpha_y_prefit is None and hasattr(modality, 'alpha_y_prefit') and modality.alpha_y_prefit is not None:
            alpha_y_prefit = modality.alpha_y_prefit
            alpha_y_type = getattr(modality, 'alpha_y_type', 'posterior')
            print(f"[WARNING] Distribution-specific alpha_y not found, falling back to generic alpha_y_prefit. "
                  f"This may be incorrect if using old technical fit results. "
                  f"Consider re-running fit_technical with current code.")

        print(f"[INFO] Fitting trans model for modality '{modality_name}' (distribution: {distribution})")

        # Validate min_denominator is specified for binomial/multinomial
        if distribution in ['binomial', 'multinomial'] and min_denominator is None:
            raise ValueError(
                f"min_denominator is required for distribution='{distribution}'. "
                f"Please specify min_denominator (e.g., min_denominator=0 for no filtering, "
                f"or min_denominator=3 for standard quality filtering)."
            )

        # Validate distribution-specific requirements
        from .distributions import requires_sum_factor, requires_denominator

        if requires_sum_factor(distribution) and sum_factor_col is None:
            raise ValueError(f"Distribution '{distribution}' requires sum_factor_col parameter")

        if requires_denominator(distribution) and denominator is None:
            raise ValueError(f"Distribution '{distribution}' requires denominator parameter")

        # convert to gpu for fitting if applicable
        if self.model.x_true is not None and self.model.x_true.device != self.model.device:
            self.model.x_true = self.model.x_true.to(self.model.device)
        if alpha_y_prefit is not None and alpha_y_prefit.device != self.model.device:
            alpha_y_prefit = alpha_y_prefit.to(self.model.device)

        if not hasattr(self.model, "log2_x_true") or self.model.log2_x_true is None:
            if self.model.x_true is not None:
                self.model.log2_x_true = torch.log2(self.model.x_true)
                self.model.log2_x_true_type = self.model.x_true_type

        # Handle cell subsetting
        # CRITICAL: Preserve exact modality cell order when subsetting meta
        # Use .set_index().loc[] to ensure meta_subset has same order as modality_cells
        meta_with_index = self.model.meta.set_index('cell')
        meta_subset = meta_with_index.loc[modality_cells].reset_index()

        # Verify we got all cells (debugging)
        if len(meta_subset) != len(modality_cells):
            missing = set(modality_cells) - set(meta_subset['cell'].values)
            raise ValueError(f"Some modality cells not found in meta: {missing}")

        # Check if technical_group_code exists (for correction)
        if "technical_group_code" in meta_subset.columns:
            C = meta_subset['technical_group_code'].nunique()
            groups_tensor = torch.tensor(meta_subset['technical_group_code'].values, dtype=torch.long, device=self.model.device)
            print(f"[INFO] Using technical_group_code with {C} groups for correction")
        else:
            C = None
            groups_tensor = None
            if alpha_y_prefit is None:
                warnings.warn("no alpha_y_prefit and no technical_group_code, assuming no confounding effect.")

        # All cells are already in correct order, no need for cell_indices
        N = len(modality_cells)

        # Modality counts are already in correct order matching modality_cells
        # Handle both 2D and 3D arrays
        if counts_to_fit.ndim == 2:
            if modality.cells_axis == 1:
                y_obs = counts_to_fit.T  # [T, N] -> [N, T]
            else:
                y_obs = counts_to_fit  # Already [N, T]
            T = y_obs.shape[1]
        elif counts_to_fit.ndim == 3:
            # 3D data: (features, cells, categories/dimensions)
            # Transpose to (cells, features, categories/dimensions)
            y_obs = counts_to_fit.transpose(1, 0, 2)  # [T, N, K] -> [N, T, K]
            T = counts_to_fit.shape[0]
        else:
            raise ValueError(f"Unexpected number of dimensions: {counts_to_fit.ndim}")

        # Handle sum factors for modality cells
        if sum_factor_col is not None:
            sum_factor_tensor = torch.tensor(meta_subset[sum_factor_col].values, dtype=torch.float32, device=self.model.device)
        else:
            sum_factor_tensor = torch.ones(N, dtype=torch.float32, device=self.model.device)

        # Handle denominator for modality cells
        denominator_tensor = None
        if denominator is not None:
            if denominator.ndim == 2:
                if modality.cells_axis == 1:
                    denominator_subset = denominator.T  # [T, N] -> [N, T]
                else:
                    denominator_subset = denominator  # Already [N, T]
                denominator_tensor = torch.tensor(denominator_subset, dtype=torch.float32, device=self.model.device)
            elif denominator.ndim == 3:
                # 3D denominator (shouldn't happen for current distributions, but handle it)
                denominator_subset = denominator.transpose(1, 0, 2)
                denominator_tensor = torch.tensor(denominator_subset, dtype=torch.float32, device=self.model.device)

            # Apply min_denominator filter if specified
            if min_denominator is not None and min_denominator > 0:
                # Create mask for observations where denominator < threshold
                low_coverage_mask = denominator_tensor < min_denominator
                n_masked = low_coverage_mask.sum().item()
                n_total = denominator_tensor.numel()
                pct_masked = 100 * n_masked / n_total if n_total > 0 else 0

                print(f"[INFO] Filtering observations with denominator < {min_denominator}")
                print(f"[INFO] Masked {n_masked}/{n_total} observations ({pct_masked:.1f}%)")

                # For binomial distributions, we'll pass the mask to the model
                # The sampler will need to handle it (for now, set those observations to special value)
                # We'll use a very negative value that the sampler can detect
                # Actually, better approach: modify y_obs to have NaN for masked observations
                # But binomial doesn't support NaN observations...
                # Best approach: set denominator=0 for masked observations, and sampler handles it
                denominator_tensor = torch.where(low_coverage_mask,
                                                 torch.zeros_like(denominator_tensor),
                                                 denominator_tensor)

        # Detect data dimensions (for multinomial)
        from .distributions import is_3d_distribution
        K = None
        D = None
        if is_3d_distribution(distribution):
            if y_obs.ndim == 3:
                if distribution == 'multinomial':
                    K = y_obs.shape[2]  # Number of categories
            else:
                raise ValueError(f"Distribution '{distribution}' requires 3D data but got shape {y_obs.shape}")
        if self.model.x_true_type == 'point':
            x_true_mean = self.model.x_true
        elif self.model.x_true_type == 'posterior':
            x_true_mean = self.model.x_true.mean(dim=0)
        beta_o_alpha_tensor = torch.tensor(beta_o_alpha, dtype=torch.float32, device=self.model.device)
        beta_o_beta_tensor = torch.tensor(beta_o_beta, dtype=torch.float32, device=self.model.device)
        alpha_alpha_mu_tensor = torch.tensor(alpha_alpha_mu, dtype=torch.float32, device=self.model.device)
        K_alpha_tensor = torch.tensor(K_alpha, dtype=torch.float32, device=self.model.device)
        Vmax_alpha_tensor = torch.tensor(Vmax_alpha, dtype=torch.float32, device=self.model.device)
        n_mu_tensor = torch.tensor(n_mu, dtype=torch.float32, device=self.model.device)
        y_obs_tensor = torch.tensor(y_obs, dtype=torch.float32, device=self.model.device)
        epsilon_tensor = torch.tensor(epsilon, dtype=torch.float32, device=self.model.device)
        p_n_tensor = torch.tensor(p_n, dtype=torch.float32, device=self.model.device)
        # Pre-compute logits for RelaxedBernoulli (more numerically stable than probs)
        # logits = log(p / (1-p))
        p_n_logits_tensor = torch.tensor(np.log(p_n / (1 - p_n)), dtype=torch.float32, device=self.model.device)

        # --- robust, finite bounds for n to avoid overflow in x**n ---
        # use the same x_true sample type you use elsewhere
        x_for_bounds = self.model.x_true if self.model.x_true_type == "posterior" else self.model.x_true
        x_min = torch.clamp(x_for_bounds.min(), min=1e-12)  # strictly > 0 to avoid log(0)
        x_max = x_for_bounds.max()

        log_fmax = torch.log(torch.tensor(torch.finfo(torch.float32).max, device=self.model.device))

        # candidates can be inf if denominator ~ 0; we cap them later
        nmin_cand = (-log_fmax / torch.abs(torch.log(x_min))) if (x_min < 1) else torch.tensor(float('-inf'), device=self.model.device)
        nmax_cand = ( log_fmax / torch.abs(torch.log(x_max))) if (x_max > 1) else torch.tensor(float('inf'),  device=self.model.device)

        BOX_LOW  = torch.tensor(-20.0, device=self.model.device)
        BOX_HIGH = torch.tensor( 20.0, device=self.model.device)

        nmin = torch.where(torch.isfinite(nmin_cand), torch.maximum(nmin_cand, BOX_LOW),  BOX_LOW)
        nmax = torch.where(torch.isfinite(nmax_cand), torch.minimum(nmax_cand, BOX_HIGH), BOX_HIGH)
        # ensure proper ordering just in case
        nmin = torch.minimum(nmin, nmax)

        # CRITICAL: Use meta_subset (correctly ordered) instead of full self.model.meta
        guides_tensor = torch.tensor(meta_subset['guide_code'].values, dtype=torch.long, device=self.model.device)

        # Distribution-specific normalization for data-driven priors
        # For building priors later:
        y_obs_for_prior = None

        # Note: min_denominator is now required for binomial/multinomial (validated earlier)
        # For negbinom/normal/studentt, it's not used so None is fine

        if distribution == 'binomial' and denominator_tensor is not None:
            # Full probabilities for the likelihood
            y_obs_factored = y_obs_tensor / denominator_tensor.clamp_min(epsilon_tensor)

            # Store valid_mask for later (will apply AFTER technical correction)
            valid_mask_binomial = (denominator_tensor >= min_denominator)
            print(f"[INFO] Binomial: will use {valid_mask_binomial.float().mean().item()*100:.1f}% of entries with denominator >= {min_denominator} for priors (after technical correction)")

            # Don't create y_obs_for_prior yet - technical correction happens first
            y_obs_for_prior = y_obs_factored

        elif distribution == 'multinomial' and y_obs_tensor.ndim == 3:
            total_counts = y_obs_tensor.sum(dim=-1, keepdim=True).clamp_min(epsilon_tensor)  # [N, T, 1]
            y_obs_factored = y_obs_tensor / total_counts  # [N, T, K]

            # Store valid_mask for later (will apply AFTER technical correction)
            valid_mask_multinomial = (total_counts >= min_denominator)  # [N, T, 1]
            print(f"[INFO] Multinomial: will use {valid_mask_multinomial.float().mean().item()*100:.1f}% of entries with total counts >= {min_denominator} for priors (after technical correction)")

            # Don't create y_obs_for_prior yet - technical correction happens first
            y_obs_for_prior = y_obs_factored

        elif sum_factor_col is not None:
            y_obs_factored = y_obs_tensor / sum_factor_tensor.view(-1, 1)
            y_obs_for_prior = y_obs_factored
        else:
            y_obs_factored = y_obs_tensor
            y_obs_for_prior = y_obs_factored

        # ===================================================================
        # CORRECT FOR TECHNICAL EFFECTS BEFORE COMPUTING PRIORS
        # ===================================================================
        # The observed data includes technical batch effects. To compute unbiased
        # priors for A and Vmax (baseline parameters), we can optionally remove these effects
        # using the inverse transformation.
        # NOTE: Archive code does NOT correct priors - it uses raw sum_factor-normalized data.
        # Technical effects are only applied during model fitting via alpha_y_sample.
        # Setting correct_priors_for_technical=False (default) matches archive behavior.
        if correct_priors_for_technical and alpha_y_prefit is not None and groups_tensor is not None:
            print(f"[INFO] Correcting for technical effects before computing priors (distribution: {distribution})")

            # Handle both point estimates (2D) and posterior samples (3D)
            # Check actual tensor dimensions rather than just alpha_y_type (for robustness)
            # Note: y_obs_for_prior is always [N, T] (cells × features)
            if alpha_y_prefit.dim() >= 3:
                # alpha_y_prefit is 3D/4D: [samples, groups, features] or [samples, groups, features, categories]
                # Index groups dimension and average over samples
                alpha_y_grouped = alpha_y_prefit[:, groups_tensor, ...]  # [S, N, ...] where ... is T or T,K
                alpha_y_expanded = alpha_y_grouped.mean(dim=0)  # [N, ...] where ... is T or T,K
                print(f"[INFO] Using posterior mean for technical correction (averaged over {alpha_y_prefit.shape[0]} samples)")
            else:
                # alpha_y_prefit is 2D: [groups, features] or 3D: [groups, features, categories]
                # Just index groups dimension
                alpha_y_expanded = alpha_y_prefit[groups_tensor, ...]  # [N, ...] where ... is T or T,K
                if alpha_y_type == 'posterior':
                    print(f"[WARNING] alpha_y_type='posterior' but tensor is only {alpha_y_prefit.dim()}D. Treating as point estimate.")

            # alpha_y_expanded is now [N, T] (or [N, T, K] for multinomial), matching y_obs_for_prior
            # NO TRANSPOSE NEEDED - both are [N, T]

            # Diagnostic: Check alpha_y_expanded for issues
            if not torch.isfinite(alpha_y_expanded).all():
                n_invalid_alpha = (~torch.isfinite(alpha_y_expanded)).sum().item()
                print(f"[WARNING] alpha_y_expanded contains {n_invalid_alpha} non-finite values before correction")
            else:
                # Check if values look like multiplicative (around 1.0) vs additive (logit scale)
                alpha_mean = alpha_y_expanded.mean().item()
                alpha_std = alpha_y_expanded.std().item()
                print(f"[DEBUG] alpha_y_expanded stats: mean={alpha_mean:.4f}, std={alpha_std:.4f}")
                if distribution == 'binomial' and abs(alpha_mean - 1.0) < 0.5:
                    print(f"[WARNING] alpha_y_expanded looks like multiplicative correction (mean~1.0). "
                          f"For binomial, expected additive correction on logit scale (mean~0, range~[-5,5]). "
                          f"You may be using technical fit results from old (buggy) code. "
                          f"Re-run fit_technical with the fixed code.")

            if distribution == 'negbinom':
                # Technical effect: multiplicative (mu_corrected = mu * alpha_y_mult)
                # Inverse: divide by alpha_y_mult to get baseline
                # alpha_y_prefit for negbinom is multiplicative (from fit_technical)
                y_obs_for_prior = y_obs_for_prior / alpha_y_expanded.clamp_min(epsilon_tensor)
                print(f"[INFO] negbinom: Applied inverse multiplicative correction (divide by alpha_y_mult)")

            elif distribution in ['normal', 'studentt']:
                # Technical effect: additive (mu_corrected = mu + alpha_y_add)
                # Inverse: subtract alpha_y_add to get baseline
                # alpha_y_prefit for normal/studentt is additive (from fit_technical)
                y_obs_for_prior = y_obs_for_prior - alpha_y_expanded
                print(f"[INFO] {distribution}: Applied inverse additive correction (subtract alpha_y_add)")

            elif distribution == 'binomial':
                # Technical effect: logit scale (logit(p_corrected) = logit(p) + alpha_y_add)
                # Inverse: logit(p_baseline) = logit(p_observed) - alpha_y_add
                # Then: p_baseline = sigmoid(logit(p_baseline))

                # Convert observed proportions to logit scale
                p_obs_clamped = torch.clamp(y_obs_for_prior, min=epsilon_tensor, max=1.0 - epsilon_tensor)
                logit_obs = torch.log(p_obs_clamped) - torch.log(1.0 - p_obs_clamped)

                # Diagnostic: Check logit_obs range
                print(f"[DEBUG] logit_obs range: [{logit_obs.min().item():.2f}, {logit_obs.max().item():.2f}]")
                print(f"[DEBUG] alpha_y_expanded range: [{alpha_y_expanded.min().item():.4f}, {alpha_y_expanded.max().item():.4f}]")

                # Apply inverse correction on logit scale
                logit_baseline = logit_obs - alpha_y_expanded

                # Diagnostic: Check logit_baseline range and finite values
                print(f"[DEBUG] logit_baseline range: [{logit_baseline.min().item():.2f}, {logit_baseline.max().item():.2f}]")
                if not torch.isfinite(logit_baseline).all():
                    n_invalid = (~torch.isfinite(logit_baseline)).sum().item()
                    print(f"[WARNING] logit_baseline contains {n_invalid} non-finite values after correction")

                # Convert back to probability scale
                y_obs_for_prior = torch.sigmoid(logit_baseline)
                print(f"[INFO] binomial: Applied inverse logit correction (subtract alpha_y_add on logit scale)")

                # DIAGNOSTIC: Check correction for specific feature
                sj_check_idx = 1298  # User's specific SJ position
                if sj_check_idx < T:
                    # Get uncorrected values (observed PSI)
                    y_uncorrected = y_obs_factored[:, sj_check_idx]

                    # Get corrected values (baseline PSI)
                    y_corrected = y_obs_for_prior[:, sj_check_idx]

                    # Check by group
                    print(f"[DEBUG] Checking inverse correction for feature {sj_check_idx} (chr6:34236964:34237203):")
                    for grp in range(C):
                        grp_mask = groups_tensor == grp
                        if grp_mask.sum() > 0:
                            uncorr_mean = y_uncorrected[grp_mask].mean().item()
                            corr_mean = y_corrected[grp_mask].mean().item()
                            alpha_val = alpha_y_prefit[grp, sj_check_idx].item() if alpha_y_prefit.ndim == 2 else alpha_y_prefit[0, grp, sj_check_idx].mean().item()
                            print(f"  Group {grp} (n={grp_mask.sum()}): α={alpha_val:.4f}, Observed PSI={uncorr_mean:.4f}, Baseline PSI={corr_mean:.4f}")

            elif distribution == 'multinomial':
                # Technical effect: log scale (log(probs_corrected) = log(probs) + alpha_y_add)
                # Inverse: log(probs_baseline) = log(probs_observed) - alpha_y_add
                # Then: probs_baseline = exp(log_probs_baseline) / sum(exp(...))
                # For multinomial, alpha_y_expanded is already [N, T, K] from above (matches y_obs_for_prior)

                # Convert observed proportions to log scale
                p_obs_clamped = torch.clamp(y_obs_for_prior, min=epsilon_tensor)
                log_probs_obs = torch.log(p_obs_clamped)

                # Apply inverse correction on log scale
                log_probs_baseline = log_probs_obs - alpha_y_expanded

                # Normalize (softmax) to get valid probabilities
                y_obs_for_prior = torch.softmax(log_probs_baseline, dim=-1)
                print(f"[INFO] multinomial: Applied inverse log correction (subtract alpha_y_add on log scale)")

            # Handle any NaN or invalid values that may result from correction
            # (e.g., if correction pushes values outside valid range)
            if not torch.isfinite(y_obs_for_prior).all():
                n_invalid = (~torch.isfinite(y_obs_for_prior)).sum().item()
                n_total = y_obs_for_prior.numel()
                print(f"[WARNING] Technical correction produced {n_invalid}/{n_total} "
                      f"({100*n_invalid/n_total:.2f}%) non-finite values. "
                      f"These will be excluded from prior computation.")
                # Mark invalid values as NaN so they're excluded by nanmean/nanvar
                y_obs_for_prior = torch.where(
                    torch.isfinite(y_obs_for_prior),
                    y_obs_for_prior,
                    torch.full_like(y_obs_for_prior, float('nan'))
                )
        else:
            if alpha_y_prefit is not None:
                print(f"[INFO] alpha_y_prefit provided but groups_tensor is None - skipping technical correction")

        # NOW apply min_denominator masking (AFTER technical correction)
        # This prevents computing priors from low-coverage observations
        if distribution == 'binomial' and 'valid_mask_binomial' in locals():
            # Mask low-coverage entries with NaN
            y_obs_for_prior = torch.where(
                valid_mask_binomial,
                y_obs_for_prior,
                torch.full_like(y_obs_for_prior, float('nan'))
            )
            n_masked = (~valid_mask_binomial).sum().item()
            n_total = valid_mask_binomial.numel()
            print(f"[INFO] Masked {n_masked}/{n_total} ({100*n_masked/n_total:.2f}%) low-coverage observations (denominator < {min_denominator})")

        elif distribution == 'multinomial' and 'valid_mask_multinomial' in locals():
            # Mask low-coverage entries with NaN
            y_obs_for_prior = torch.where(
                valid_mask_multinomial,
                y_obs_for_prior,
                torch.full_like(y_obs_for_prior, float('nan'))
            )
            n_masked = (~valid_mask_multinomial).sum().item()
            n_total = valid_mask_multinomial.numel()
            print(f"[INFO] Masked {n_masked}/{n_total} ({100*n_masked/n_total:.2f}%) low-coverage observations (total counts < {min_denominator})")

        if function_type == 'logistic_hill':
            if self.model.x_true is None:
                raise ValueError("function_type='logistic_hill' requires x_true; run fit_cis() first.")
            ntc_mask = (meta_subset['target'].astype(str).str.lower() == 'ntc').values
            posterior_samples_y = self._fit_trans_logistic(
                N=N, T=T,
                y_obs_tensor=y_obs_tensor,
                y_obs_for_prior=y_obs_for_prior,
                sum_factor_tensor=sum_factor_tensor,
                denominator_tensor=denominator_tensor,
                alpha_y_prefit=alpha_y_prefit,
                groups_tensor=groups_tensor,
                C=C,
                ntc_mask=ntc_mask,
                distribution=distribution,
                niters=niters,
                lr=lr,
                nsamples=nsamples,
                tolerance=tolerance,
                beta_o_alpha_tensor=beta_o_alpha_tensor,
                beta_o_beta_tensor=beta_o_beta_tensor,
                use_epsilon=use_epsilon,
                n_components=n_components,
                logistic_guide=logistic_guide,
                horseshoe_tau0=horseshoe_tau0,
                horseshoe_slab_scale=horseshoe_slab_scale,
            )
            self._store_trans_results(modality, modality_name, posterior_samples_y, groups_tensor)
            return

        unique_guides = torch.unique(guides_tensor)

        # nanmean and nanvar helpers (in case torch.nanmean/nanvar isn't available)
        if hasattr(torch, "nanmean"):
            def nanmean(x, dim):
                return torch.nanmean(x, dim=dim)
        else:
            def nanmean(x, dim):
                mask = ~torch.isnan(x)
                num = torch.where(mask, x, torch.zeros_like(x)).sum(dim=dim)
                den = mask.sum(dim=dim).clamp_min(1)
                return num / den

        if hasattr(torch, "nanvar"):
            def nanvar(x, dim):
                return torch.var(x, dim=dim)  # Note: torch.var ignores NaN in older versions
        else:
            def nanvar(x, dim):
                mask = ~torch.isnan(x)
                n = mask.sum(dim=dim).clamp_min(2)  # Need at least 2 values for variance
                x_mean = nanmean(x, dim=dim)
                # Expand mean to match x shape for broadcasting
                if dim == 0:
                    x_mean_expanded = x_mean.unsqueeze(0)
                else:
                    x_mean_expanded = x_mean
                sq_diff = torch.where(mask, (x - x_mean_expanded) ** 2, torch.zeros_like(x))
                return sq_diff.sum(dim=dim) / (n - 1)  # Unbiased variance

        # nanquantile helper
        def nanquantile(x, q, dim):
            """Compute quantile ignoring NaN values."""
            if x.ndim == 2:  # [G, T]
                result = []
                for t in range(x.shape[1]):
                    vals = x[:, t]
                    valid = vals[~torch.isnan(vals)]
                    if valid.numel() > 0:
                        result.append(torch.quantile(valid, q))
                    else:
                        result.append(torch.tensor(float('nan'), device=x.device))
                return torch.stack(result)  # [T]
            elif x.ndim == 3:  # [G, T, K]
                result = []
                for t in range(x.shape[1]):
                    result_k = []
                    for k in range(x.shape[2]):
                        vals = x[:, t, k]
                        valid = vals[~torch.isnan(vals)]
                        if valid.numel() > 0:
                            result_k.append(torch.quantile(valid, q))
                        else:
                            result_k.append(torch.tensor(float('nan'), device=x.device))
                    result.append(torch.stack(result_k))
                return torch.stack(result)  # [T, K]
            else:
                raise ValueError(f"Unsupported ndim for nanquantile: {x.ndim}")

        # Compute priors: use percentiles from guide means (works with or without guides)
        if 'guide_code' in self.model.meta.columns and len(unique_guides) > 1:
            # WITH GUIDES: Compute guide-level statistics
            guide_means = []
            guide_vars = []
            for g in unique_guides:
                vals_g = y_obs_for_prior[guides_tensor == g, ...]  # [Ng, T] or [Ng, T, K]
                guide_means.append(nanmean(vals_g, dim=0))         # [T] or [T, K]
                guide_vars.append(nanvar(vals_g, dim=0))           # [T] or [T, K]

            guide_means = torch.stack(guide_means, dim=0)  # [G, T] or [G, T, K]
            guide_vars = torch.stack(guide_vars, dim=0)    # [G, T] or [G, T, K]

            if use_archive_prior_computation:
                # ARCHIVE METHOD: Use min/max of guide means (matching archive behavior)
                # This uses the actual min and max observed guide means per feature
                # Handle NaN values by using nanmin/nanmax
                def nanmin(x, dim):
                    """Compute min ignoring NaN values."""
                    # Replace NaN with +inf so they don't affect min
                    x_filled = torch.where(torch.isnan(x), torch.full_like(x, float('inf')), x)
                    result = torch.min(x_filled, dim=dim)[0]
                    # If all values were NaN, result will be inf - replace with nan
                    all_nan = torch.all(torch.isnan(x), dim=dim)
                    return torch.where(all_nan, torch.full_like(result, float('nan')), result)

                def nanmax(x, dim):
                    """Compute max ignoring NaN values."""
                    # Replace NaN with -inf so they don't affect max
                    x_filled = torch.where(torch.isnan(x), torch.full_like(x, float('-inf')), x)
                    result = torch.max(x_filled, dim=dim)[0]
                    # If all values were NaN, result will be -inf - replace with nan
                    all_nan = torch.all(torch.isnan(x), dim=dim)
                    return torch.where(all_nan, torch.full_like(result, float('nan')), result)

                Amean_tensor = nanmin(guide_means, dim=0)  # [T] or [T, K]
                Vmax_mean_tensor = nanmax(guide_means, dim=0)  # [T] or [T, K]

                # Ensure positive
                Amean_tensor = Amean_tensor.clamp_min(self._t(1e-3))
                Vmax_mean_tensor = Vmax_mean_tensor.clamp_min(self._t(1e-3))

                print(f"[INFO] Using guide-based priors (archive method): {len(unique_guides)} guides, min/max")

            else:
                # PERCENTILE METHOD: Use 5th and 95th percentiles of guide means
                # This is more robust to outliers but computes Vmax as range (amplitude)
                Amean_tensor = nanquantile(guide_means, 0.05, dim=0)  # [T] or [T, K]
                upper_quantile = nanquantile(guide_means, 0.95, dim=0)  # [T] or [T, K]

                # Ensure A_mean >= 1e-3
                Amean_tensor = Amean_tensor.clamp_min(self._t(1e-3))

                # Vmax is the RANGE (amplitude), not the absolute value
                # Model: y = A + Vmax_a * Hill_a, so Vmax_a = y_max - A
                Vmax_mean_tensor = upper_quantile - Amean_tensor  # [T] or [T, K]
                Vmax_mean_tensor = Vmax_mean_tensor.clamp_min(self._t(1e-3))  # Ensure positive

                print(f"[INFO] Using guide-based priors (percentile method): {len(unique_guides)} guides, 5th/95th percentiles")

            # For variances: use mean variance across guides (average within-guide variability)
            # This captures how much variability exists around each guide's mean
            mean_within_guide_var = nanmean(guide_vars, dim=0)  # [T] or [T, K]

        else:
            # WITHOUT GUIDES: Use overall percentiles
            print("[INFO] No guides found or only 1 guide. Using overall data percentiles for priors.")

            # Compute 5th and 95th percentiles across all cells for each feature
            if y_obs_for_prior.ndim == 2:  # [N, T]
                Amean_tensor = []
                Vmax_mean_tensor = []
                overall_vars = []
                for t in range(y_obs_for_prior.shape[1]):
                    vals_t = y_obs_for_prior[:, t]
                    valid_t = vals_t[~torch.isnan(vals_t)]
                    if valid_t.numel() > 0:
                        Amean_tensor.append(torch.quantile(valid_t, 0.05))
                        Vmax_mean_tensor.append(torch.quantile(valid_t, 0.95))
                        overall_vars.append(torch.var(valid_t))
                    else:
                        Amean_tensor.append(torch.tensor(float('nan'), device=self.model.device))
                        Vmax_mean_tensor.append(torch.tensor(float('nan'), device=self.model.device))
                        overall_vars.append(torch.tensor(float('nan'), device=self.model.device))

                Amean_tensor = torch.stack(Amean_tensor)  # [T]
                Vmax_mean_tensor = torch.stack(Vmax_mean_tensor)  # [T]
                mean_within_guide_var = torch.stack(overall_vars)  # [T]

            elif y_obs_for_prior.ndim == 3:  # [N, T, K]
                Amean_tensor = []
                Vmax_mean_tensor = []
                overall_vars = []
                for t in range(y_obs_for_prior.shape[1]):
                    Amean_k = []
                    Vmax_k = []
                    var_k = []
                    for k in range(y_obs_for_prior.shape[2]):
                        vals_tk = y_obs_for_prior[:, t, k]
                        valid_tk = vals_tk[~torch.isnan(vals_tk)]
                        if valid_tk.numel() > 0:
                            Amean_k.append(torch.quantile(valid_tk, 0.05))
                            Vmax_k.append(torch.quantile(valid_tk, 0.95))
                            var_k.append(torch.var(valid_tk))
                        else:
                            Amean_k.append(torch.tensor(float('nan'), device=self.model.device))
                            Vmax_k.append(torch.tensor(float('nan'), device=self.model.device))
                            var_k.append(torch.tensor(float('nan'), device=self.model.device))
                    Amean_tensor.append(torch.stack(Amean_k))
                    Vmax_mean_tensor.append(torch.stack(Vmax_k))
                    overall_vars.append(torch.stack(var_k))

                Amean_tensor = torch.stack(Amean_tensor)  # [T, K]
                Vmax_mean_tensor = torch.stack(Vmax_mean_tensor)  # [T, K]
                mean_within_guide_var = torch.stack(overall_vars)  # [T, K]

            # Ensure A_mean >= 1e-3
            Amean_tensor = Amean_tensor.clamp_min(self._t(1e-3))

            # Vmax is the RANGE (amplitude), not the absolute value
            # Model: y = A + Vmax_a * Hill_a, so Vmax_a = y_max - A
            # (upper_quantile is stored in Vmax_mean_tensor before this line)
            Vmax_mean_tensor = Vmax_mean_tensor - Amean_tensor  # [T] or [T, K]
            Vmax_mean_tensor = Vmax_mean_tensor.clamp_min(self._t(1e-3))  # Ensure positive

        # For binomial/multinomial: clamp Vmax_mean to valid Beta range
        if distribution in ['binomial', 'multinomial']:
            Vmax_mean_tensor = Vmax_mean_tensor.clamp(min=self._t(1e-3), max=self._t(1.0 - 1e-6))
            Amean_tensor = Amean_tensor.clamp(min=self._t(1e-3), max=self._t(1.0 - 1e-6))

        # Handle NaN values (features where ALL observations were filtered out)
        # For binomial/multinomial with denominator filtering, some features may have no valid observations
        nan_mask = torch.isnan(Amean_tensor) | torch.isnan(Vmax_mean_tensor)
        if nan_mask.any():
            n_nan = nan_mask.sum().item() if nan_mask.ndim == 1 else nan_mask.any(dim=-1).sum().item()
            print(f"[WARNING] {n_nan} features have all observations filtered (denominator < {min_denominator}). Using fallback values.")

            # Fallback: use global mean/var across all valid features
            if distribution in ['binomial', 'multinomial']:
                # Compute fallback from valid (non-NaN) entries only
                valid_means = guide_means[~torch.isnan(guide_means)]
                valid_vars = guide_vars[~torch.isnan(guide_vars)]

                if valid_means.numel() > 0:
                    # Use median for robustness
                    valid_mean = torch.median(valid_means)
                    valid_var = torch.median(valid_vars) if valid_vars.numel() > 0 else self._t(0.1)
                else:
                    # Last resort: use generic defaults for PSI data
                    print("[WARNING] No valid observations found! Using generic defaults: A=0.1, Vmax=0.5, var=0.1")
                    valid_mean = self._t(0.3)  # Midpoint for PSI
                    valid_var = self._t(0.1)  # Reasonable variance for PSI

                print(f"[INFO] Using fallback: mean={valid_mean.item():.3f}, var={valid_var.item():.3f}")

                # Replace NaN with fallback
                # Assume range from 50% to 150% of typical value
                fallback_A = valid_mean * 0.5  # A = 50% of mean (minimum)
                fallback_max = valid_mean * 1.5  # Upper bound (maximum)
                fallback_Vmax = fallback_max - fallback_A  # Vmax = RANGE (not maximum!)
                fallback_A = torch.clamp(fallback_A, min=0.01, max=0.99)  # Keep in valid Beta range
                fallback_Vmax = torch.clamp(fallback_Vmax, min=0.01, max=0.99)

                Amean_tensor = torch.where(torch.isnan(Amean_tensor),
                                          torch.full_like(Amean_tensor, fallback_A),
                                          Amean_tensor)
                Vmax_mean_tensor = torch.where(torch.isnan(Vmax_mean_tensor),
                                              torch.full_like(Vmax_mean_tensor, fallback_Vmax),
                                              Vmax_mean_tensor)
                mean_within_guide_var = torch.where(torch.isnan(mean_within_guide_var),
                                                   torch.full_like(mean_within_guide_var, valid_var),
                                                   mean_within_guide_var)

            else:  # negbinom, normal, studentt
                # Compute fallback from valid (non-NaN) entries only
                valid_means = guide_means[~torch.isnan(guide_means)]
                valid_vars = guide_vars[~torch.isnan(guide_vars)]

                if valid_means.numel() > 0:
                    # Use median for robustness
                    valid_mean = torch.median(valid_means)
                    valid_var = torch.median(valid_vars) if valid_vars.numel() > 0 else valid_mean * 0.5
                else:
                    # Last resort: use generic defaults for count/continuous data
                    print("[WARNING] No valid observations found! Using generic defaults: A=1.0, Vmax=10.0, var=5.0")
                    valid_mean = self._t(5.0)  # Midpoint for log-space counts
                    valid_var = self._t(5.0)  # Reasonable variance

                print(f"[INFO] Using fallback: mean={valid_mean.item():.3f}, var={valid_var.item():.3f}")

                # Replace NaN with fallback
                # Assume range from 50% to 150% of typical value
                fallback_A = valid_mean * 0.5  # A = 50% of mean (minimum)
                fallback_max = valid_mean * 1.5  # Upper bound (maximum)
                fallback_Vmax = fallback_max - fallback_A  # Vmax = RANGE (not maximum!)
                fallback_A = torch.clamp(fallback_A, min=1e-3)  # Ensure positive
                fallback_Vmax = torch.clamp(fallback_Vmax, min=1e-3)

                Amean_tensor = torch.where(torch.isnan(Amean_tensor),
                                          torch.full_like(Amean_tensor, fallback_A),
                                          Amean_tensor)
                Vmax_mean_tensor = torch.where(torch.isnan(Vmax_mean_tensor),
                                              torch.full_like(Vmax_mean_tensor, fallback_Vmax),
                                              Vmax_mean_tensor)
                mean_within_guide_var = torch.where(torch.isnan(mean_within_guide_var),
                                                   torch.full_like(mean_within_guide_var, valid_var),
                                                   mean_within_guide_var)

        # For K: use CV (coefficient of variation) of x_true (works with or without guides)
        # CV = std(x_true) / mean(x_true) - scale-invariant measure of variability
        x_true_mean_global = x_true_mean.mean()
        x_true_std_global = x_true_mean.std()
        x_true_CV = x_true_std_global / x_true_mean_global.clamp_min(epsilon_tensor)

        # K_max: max of x_true (or max of guide means if guides exist)
        if 'guide_code' in self.model.meta.columns and len(unique_guides) > 1:
            guide_x_means = torch.stack([x_true_mean[guides_tensor == g].mean() for g in torch.unique(guides_tensor)])
            K_max_tensor = guide_x_means.max()
        else:
            K_max_tensor = x_true_mean.max()

        print("[DEBUG] Amean:", Amean_tensor.min().item(), Amean_tensor.max().item())
        print("[DEBUG] Vmax_mean:", Vmax_mean_tensor.min().item(), Vmax_mean_tensor.max().item())
        print("[DEBUG] Mean within-guide variance:", mean_within_guide_var.min().item(), mean_within_guide_var.max().item())
        print("[DEBUG] x_true CV:", x_true_CV.item(), "K_max:", K_max_tensor.item())

        # Diagnostic: Verify alpha_y_prefit is correctly structured
        if alpha_y_prefit is not None and groups_tensor is not None:
            print(f"[INFO] Technical correction setup: alpha_y_prefit.shape={alpha_y_prefit.shape}, C={C}")
            if alpha_y_prefit.shape[0] == C if alpha_y_prefit.ndim == 2 else alpha_y_prefit.shape[1] == C:
                print(f"[INFO] alpha_y_prefit already includes reference group (correct!)")
            else:
                print(f"[WARNING] alpha_y_prefit shape mismatch - may need to add reference group")

        # Ensure x_true is on the correct device (may have been loaded from CPU)
        # Always move to device to avoid device comparison issues (cuda vs cuda:0)
        self.model.x_true = self.model.x_true.to(self.model.device)
        if alpha_y_prefit is not None:
            alpha_y_prefit = alpha_y_prefit.to(self.model.device)

        def init_loc_fn(site):
            name = site["name"]
        
            if "poly_coeff" in name:
                return torch.zeros(T)
        
            return pyro.infer.autoguide.initialization.init_to_median(site)
        
        from torch.optim.lr_scheduler import OneCycleLR
        if function_type == "polynomial":
            guide_y = pyro.infer.autoguide.AutoMultivariateNormal(self._model_y, init_loc_fn=init_loc_fn)

            guide_y(
                N,
                T,
                y_obs_tensor,
                sum_factor_tensor,
                beta_o_alpha_tensor,
                beta_o_beta_tensor,
                alpha_alpha_mu_tensor,
                K_max_tensor,
                K_alpha_tensor,
                Vmax_mean_tensor,
                Vmax_alpha_tensor,
                n_mu_tensor,
                Amean_tensor,
                p_n_logits_tensor,
                epsilon_tensor,
                x_true_sample = self.model.x_true.mean(dim=0) if self.model.x_true_type == "posterior" else self.model.x_true,
                log2_x_true_sample = self.model.log2_x_true.mean(dim=0) if self.model.log2_x_true_type == "posterior" else self.model.log2_x_true,
                nmin = nmin,
                nmax = nmax,
                alpha_y_sample = (
                    alpha_y_prefit.mean(dim=0) if alpha_y_prefit.dim() >= 3 else alpha_y_prefit
                ) if alpha_y_prefit is not None else None,
                C = C,
                groups_tensor=groups_tensor,
                temperature=torch.tensor(init_temp, dtype=torch.float32, device=self.model.device),
                use_straight_through=False,
                function_type=function_type,
                polynomial_degree=polynomial_degree,
                use_alpha=True,
                distribution=distribution,
                denominator_tensor=denominator_tensor,
                K=K,
                D=D,
                mean_within_guide_var=mean_within_guide_var,
                x_true_CV=x_true_CV,
                use_data_driven_priors=use_data_driven_priors,
                use_lognormal_priors=use_lognormal_priors,
                use_epsilon=use_epsilon,
            )
            # OneCycleLR for polynomial only
            base_lr = 1e-3 if lr is None else lr

            optimizer = pyro.optim.PyroLRScheduler(
                scheduler_constructor=OneCycleLR,
                optim_args={
                    # underlying torch optimizer
                    "optimizer": torch.optim.Adam,
                    "optim_args": {
                        "lr": base_lr,      # initial lr (OneCycle will move it)
                        "betas": (0.9, 0.999),
                    },
                    # OneCycleLR hyperparameters
                    "max_lr":          base_lr * 10,  # was 1e-2 when base_lr=1e-3
                    "total_steps":     niters,
                    "pct_start":       0.1,
                    "div_factor":      25.0,
                    "final_div_factor": 1e4,
                },
                clip_args={"clip_norm": 5},
            )

            svi = pyro.infer.SVI(
                self._model_y,
                guide_y,
                optimizer,
                loss=pyro.infer.Trace_ELBO()
            )
        else:
            # Simple Adam for Hill-based function types (single_hill, additive_hill, nested_hill)
            guide_y = pyro.infer.autoguide.AutoNormalMessenger(self._model_y)
            hill_lr = 1e-3 if lr is None else lr
            optimizer = pyro.optim.ClippedAdam({"lr": hill_lr, "clip_norm": 10.0})
            svi = pyro.infer.SVI(
                self._model_y,
                guide_y,
                optimizer,
                loss=pyro.infer.Trace_ELBO()
            )

        
        for name, value in pyro.get_param_store().items():
            if "poly_coeff" in name and "loc" in name:
                print(name, value.shape, value.min().item(), value.max().item())

        self.losses_trans = []
        smoothed_loss = None
        for step in range(niters):
            # a simple linear schedule from init_temp down to final_temp:
            fraction_done = step / float(niters)
            if function_type in ['single_hill', 'additive_hill', 'nested_hill']:
                current_temp = init_temp + (final_temp - init_temp) * fraction_done
            elif function_type == 'polynomial':
                current_temp = init_temp + (final_temp - init_temp) * (2*fraction_done-1)
                #current_temp = init_temp + (final_temp - init_temp) * fraction_done
            else:
                raise ValueError(f"Unknown function_type: {function_type}")
                
            #if step < 0.7 * niters:
            #    # First 70% of training: linearly decrease from 1.0 to 0.1
            #    current_temp = 1.0 - (0.9 * (step / (0.7 * niters)))
            #else:
            #    # Last 30% of training: exponentially cool down to 0.0005
            #    current_temp = 0.1 * (final_temp/0.1) ** ((step - 0.7 * niters) / (0.3 * niters))


            # Sample from posterior - use alpha_y_prefit shape if available, otherwise x_true
            if alpha_y_prefit is not None and alpha_y_type == "posterior":
                samp = torch.randint(high=alpha_y_prefit.shape[0], size=(1,)).item()
            elif self.model.x_true_type == "posterior":
                samp = torch.randint(high=self.model.x_true.shape[0], size=(1,)).item()
            else:
                samp = 0  # No sampling needed if both are point estimates

            # Sample from posterior
            x_true_sample = (
                (self.model.x_true[samp] if samp < self.model.x_true.shape[0] else self.model.x_true.mean(dim=0))
                if self.model.x_true_type == "posterior" else self.model.x_true
            )
            log2_x_true_sample = (
                (self.model.log2_x_true[samp] if samp < self.model.log2_x_true.shape[0] else self.model.log2_x_true.mean(dim=0))
                if self.model.log2_x_true_type == "posterior" else self.model.log2_x_true
            )
            # For alpha_y_sample: check actual dimensions instead of just alpha_y_type
            if alpha_y_prefit is not None:
                if alpha_y_prefit.dim() >= 3:
                    # Posterior samples: [S, C-1, T] - sample or average
                    alpha_y_sample = (
                        alpha_y_prefit[samp] if samp < alpha_y_prefit.shape[0] else alpha_y_prefit.mean(dim=0)
                    )
                else:
                    # Point estimate: [C-1, T] - use as-is
                    alpha_y_sample = alpha_y_prefit
            else:
                alpha_y_sample = None

            #use_straight_through = step >= int(0.7 * niters)
            use_straight_through = False

            '''
            loss = svi.step(
                N,
                T,
                y_obs_tensor,
                sum_factor_tensor,
                beta_o_alpha_tensor,
                beta_o_beta_tensor,
                alpha_alpha_mu_tensor,
                K_max_tensor,
                K_alpha_tensor,
                Vmax_mean_tensor,
                Vmax_alpha_tensor,
                n_mu_tensor,
                Amean_tensor,
                p_n_logits_tensor,
                epsilon_tensor,
                x_true_sample = x_true_sample,
                log2_x_true_sample = log2_x_true_sample,
                nmin = nmin,
                nmax = nmax,
                alpha_y_sample = alpha_y_sample,
                C = C,
                groups_tensor=groups_tensor,
                temperature=torch.tensor(current_temp, dtype=torch.float32, device=self.model.device),
                use_straight_through=use_straight_through,
                function_type=function_type,
                polynomial_degree=polynomial_degree,
                use_alpha=True if function_type != 'polynomial' else True if fraction_done>=0.5 else False,
                distribution=distribution,
                denominator_tensor=denominator_tensor,
                K=K,
                D=D,
                mean_within_guide_var=mean_within_guide_var,
                x_true_CV=x_true_CV,
                use_data_driven_priors=use_data_driven_priors,
                use_lognormal_priors=use_lognormal_priors,
                use_epsilon=use_epsilon,
            )
            '''

            if step == 0:
                prev_finite = None  # initialize tracker
            try:
                loss, prev_finite = self._debug_svi_step(
                    svi, step, prev_finite,
                    N, T, y_obs_tensor, sum_factor_tensor, beta_o_alpha_tensor, beta_o_beta_tensor,
                    alpha_alpha_mu_tensor, K_max_tensor, K_alpha_tensor, Vmax_mean_tensor, Vmax_alpha_tensor,
                    n_mu_tensor, Amean_tensor, p_n_logits_tensor, epsilon_tensor,
                    x_true_sample=x_true_sample,
                    log2_x_true_sample=log2_x_true_sample,
                    nmin=nmin,
                    nmax=nmax,
                    alpha_y_sample=alpha_y_sample,
                    C=C,
                    groups_tensor=groups_tensor,
                    temperature=torch.tensor(current_temp, dtype=torch.float32, device=self.model.device),
                    use_straight_through=use_straight_through,
                    function_type=function_type,
                    polynomial_degree=polynomial_degree,
                    use_alpha=True if function_type != 'polynomial' else True if fraction_done>=0.5 else False,
                    distribution=distribution,
                    denominator_tensor=denominator_tensor,
                    K=K,
                    D=D,
                    mean_within_guide_var=mean_within_guide_var,
                    x_true_CV=x_true_CV,
                    use_data_driven_priors=use_data_driven_priors,
                    use_lognormal_priors=use_lognormal_priors,
                    use_epsilon=use_epsilon,
                )
            except FloatingPointError as e:
                print(f"[STOP] {e} at step {step}")
                break


            # NaN detection and early stopping
            if np.isnan(loss) or np.isinf(loss):
                print(f"[WARNING] NaN/Inf detected in loss at step {step}! Stopping optimization.")
                print(f"  - temperature: {current_temp}")
                print(f"  - p_n_tensor: {p_n_tensor.item()}")
                # Check parameters for NaN
                for name, value in pyro.get_param_store().items():
                    if torch.isnan(value).any():
                        nan_count = torch.isnan(value).sum().item()
                        print(f"  - Parameter '{name}' has {nan_count} NaN values")
                break

            self.losses_trans.append(loss)
            if step % 1000 == 0:
                print(f"Step {step} : loss = {loss:.5e}, device: {Vmax_mean_tensor.device}")
            if smoothed_loss is None:
                smoothed_loss = loss
            else:
                if abs(alpha_ewma * (loss - smoothed_loss)) < tolerance:
                    print(f"Converged at step {step}! Loss = {loss:.5e}")
                    break
                smoothed_loss = alpha_ewma * loss + (1 - alpha_ewma) * smoothed_loss

        # Move to CPU if using too much GPU memory for Predictive
        run_on_cpu = self.model.device.type != "cpu"
        if run_on_cpu:
            print("[INFO] Running Predictive on CPU to reduce GPU memory pressure...")
            guide_y.to("cpu")
            self.model.device = torch.device("cpu")
        
            model_inputs = {
                "N": N,
                "T": T,
                "y_obs_tensor": self._to_cpu(y_obs_tensor),
                "sum_factor_tensor": self._to_cpu(sum_factor_tensor),
                "beta_o_alpha_tensor": self._to_cpu(beta_o_alpha_tensor),
                "beta_o_beta_tensor": self._to_cpu(beta_o_beta_tensor),
                "alpha_alpha_mu_tensor": self._to_cpu(alpha_alpha_mu_tensor),
                "K_max_tensor": self._to_cpu(K_max_tensor),
                "K_alpha_tensor": self._to_cpu(K_alpha_tensor),
                "Vmax_mean_tensor": self._to_cpu(Vmax_mean_tensor),
                "Vmax_alpha_tensor": self._to_cpu(Vmax_alpha_tensor),
                "n_mu_tensor": self._to_cpu(n_mu_tensor),
                "Amean_tensor": self._to_cpu(Amean_tensor),
                "p_n_logits_tensor": self._to_cpu(p_n_logits_tensor),
                "epsilon_tensor": self._to_cpu(epsilon_tensor),
                "x_true_sample": self._to_cpu(self.model.x_true.mean(dim=0) if self.model.x_true_type == "posterior" else self.model.x_true),
                "log2_x_true_sample": self._to_cpu(self.model.log2_x_true.mean(dim=0) if self.model.log2_x_true_type == "posterior" else self.model.log2_x_true),
                "nmin": self._to_cpu(nmin),
                "nmax": self._to_cpu(nmax),
                # Only move if not None:
                "alpha_y_sample": self._to_cpu(
                    alpha_y_prefit.mean(dim=0) if alpha_y_prefit.dim() >= 3 else alpha_y_prefit
                ) if alpha_y_prefit is not None else None,
                "C": C,
                "groups_tensor": self._to_cpu(groups_tensor) if groups_tensor is not None else None,
                # create on CPU explicitly since we just set self.model.device="cpu"
                "temperature": torch.tensor(final_temp, dtype=torch.float32, device=torch.device("cpu")),
                "use_straight_through": True,
                "function_type": function_type,
                "polynomial_degree": polynomial_degree,
                "use_alpha": True,
                "distribution": distribution,
                "denominator_tensor": self._to_cpu(denominator_tensor) if denominator_tensor is not None else None,
                "K": K,
                "D": D,
                "mean_within_guide_var": self._to_cpu(mean_within_guide_var) if mean_within_guide_var is not None else None,
                "x_true_CV": self._to_cpu(x_true_CV) if x_true_CV is not None else None,
                "use_data_driven_priors": use_data_driven_priors,
                "use_lognormal_priors": use_lognormal_priors,
                "use_epsilon": use_epsilon,
            }
        else:
            model_inputs = {
                "N": N,
                "T": T,
                "y_obs_tensor": y_obs_tensor,
                "sum_factor_tensor": sum_factor_tensor,
                "beta_o_alpha_tensor": beta_o_alpha_tensor,
                "beta_o_beta_tensor": beta_o_beta_tensor,
                "alpha_alpha_mu_tensor": alpha_alpha_mu_tensor,
                "K_max_tensor": K_max_tensor,
                "K_alpha_tensor": K_alpha_tensor,
                "Vmax_mean_tensor": Vmax_mean_tensor,
                "Vmax_alpha_tensor": Vmax_alpha_tensor,
                "n_mu_tensor": n_mu_tensor,
                "Amean_tensor": Amean_tensor,
                "p_n_logits_tensor": p_n_logits_tensor,
                "epsilon_tensor": epsilon_tensor,
                "x_true_sample": self.model.x_true.mean(dim=0) if self.model.x_true_type == "posterior" else self.model.x_true,
                "log2_x_true_sample": self.model.log2_x_true.mean(dim=0) if self.model.log2_x_true_type == "posterior" else self.model.log2_x_true,
                "nmin": nmin,
                "nmax": nmax,
                "alpha_y_sample": (
                    alpha_y_prefit.mean(dim=0) if alpha_y_prefit.dim() >= 3 else alpha_y_prefit
                ) if alpha_y_prefit is not None else None,
                "C": C,
                "groups_tensor": groups_tensor if groups_tensor is not None else None,
                "temperature": torch.tensor(final_temp, dtype=torch.float32, device=self.model.device),
                "use_straight_through": True,
                "function_type": function_type,
                "polynomial_degree": polynomial_degree,
                "use_alpha": True,
                "distribution": distribution,
                "denominator_tensor": denominator_tensor if denominator_tensor is not None else None,
                "K": K,
                "D": D,
                "mean_within_guide_var": mean_within_guide_var,
                "x_true_CV": x_true_CV,
                "use_data_driven_priors": use_data_driven_priors,
                "use_lognormal_priors": use_lognormal_priors,
                "use_epsilon": use_epsilon,
            }

        if self.model.device.type == "cuda":
            torch.cuda.empty_cache()
        import gc
        gc.collect()

        max_samples = nsamples
        keep_sites = kwargs.get("keep_sites", lambda name, site: site["value"].ndim <= 2 or name != "y_obs")

        if minibatch_size is not None:
            from collections import defaultdict

            print(f"[INFO] Running Predictive in minibatches of {minibatch_size}...")
            predictive_y = pyro.infer.Predictive(
                self._model_y,
                guide=guide_y,
                num_samples=minibatch_size,
                parallel=True
            )
            all_samples = defaultdict(list)
            with torch.no_grad():
                for i in range(0, max_samples, minibatch_size):
                    samples = predictive_y(**model_inputs)
                    for k, v in samples.items():
                        if keep_sites(k, {"value": v}):
                            all_samples[k].append(self._to_cpu(v))
                    if self.model.device.type == "cuda":
                        torch.cuda.empty_cache()
                    import gc
                    gc.collect()

            posterior_samples_y = {k: torch.cat(v, dim=0) for k, v in all_samples.items()}

        else:
            predictive_y = pyro.infer.Predictive(
                self._model_y,
                guide=guide_y,
                num_samples=nsamples#,
                #parallel=True
            )
            with torch.no_grad():
                posterior_samples_y = predictive_y(**model_inputs)
                if self.model.device.type == "cuda":
                    torch.cuda.empty_cache()
                import gc
                gc.collect()

        if run_on_cpu:
            self.model.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print("[INFO] Reset self.model.device to:", self.model.device)

        for k, v in posterior_samples_y.items():
            posterior_samples_y[k] = self._to_cpu(v)

        # ----------------------------------------
        # Store nmin/nmax and check boundaries
        # ----------------------------------------
        posterior_samples_y['nmin'] = self._to_cpu(nmin)
        posterior_samples_y['nmax'] = self._to_cpu(nmax)

        # Warn if fitted n_a parameters are close to boundaries
        if 'n_a' in posterior_samples_y and function_type in ['single_hill', 'additive_hill', 'nested_hill']:
            n_a_samples = posterior_samples_y['n_a']  # Shape: [S, T]
            nmin_val = nmin.item()
            nmax_val = nmax.item()

            # Define "close to boundary" threshold (e.g., within 10% of range)
            boundary_threshold = 0.1 * (nmax_val - nmin_val)

            # Check how many samples are close to boundaries
            close_to_min = (n_a_samples < (nmin_val + boundary_threshold)).float().mean().item()
            close_to_max = (n_a_samples > (nmax_val - boundary_threshold)).float().mean().item()

            # Warn if >10% of samples are at boundaries
            if close_to_min > 0.1:
                warnings.warn(
                    f"[WARNING] {close_to_min*100:.1f}% of n_a samples are close to lower boundary (nmin={nmin_val:.2f}). "
                    f"Consider: (1) checking if x_true range is appropriate, or (2) relaxing nmin constraint.",
                    UserWarning
                )
            if close_to_max > 0.1:
                warnings.warn(
                    f"[WARNING] {close_to_max*100:.1f}% of n_a samples are close to upper boundary (nmax={nmax_val:.2f}). "
                    f"Consider: (1) checking if x_true range is appropriate, or (2) relaxing nmax constraint.",
                    UserWarning
                )

            # Summary statistics
            print(f"[INFO] n_a boundary check: nmin={nmin_val:.2f}, nmax={nmax_val:.2f}")
            print(f"[INFO]   {close_to_min*100:.1f}% of samples near lower bound, {close_to_max*100:.1f}% near upper bound")

        self._store_trans_results(modality, modality_name, posterior_samples_y, groups_tensor)

    def _store_trans_results(self, modality, modality_name, posterior_samples_y, groups_tensor):
        """Store trans posterior samples and losses in the modality (and model, if primary)."""
        # Store results
        # Store in modality
        modality.posterior_samples_trans = posterior_samples_y
        modality.losses_trans = self.losses_trans  # Store loss history

        # Update alpha_y_prefit in modality if it was None and alpha_y was sampled
        if modality.alpha_y_prefit is None and groups_tensor is not None and "alpha_y" in posterior_samples_y:
            modality.alpha_y_prefit = posterior_samples_y["alpha_y"].mean(dim=0)
            modality.alpha_y_type = 'point'  # Mean of posterior -> point estimate

        # If primary modality, also store at model level (backward compatibility)
        if modality_name == self.model.primary_modality:
            self.model.posterior_samples_trans = posterior_samples_y
            self.model.losses_trans = self.losses_trans  # Store loss history at model level
            # NOTE: alpha_y_prefit is stored per-modality (already done above), not at model level
            print(f"[INFO] Stored results in modality '{modality_name}' and at model level (primary modality)")
        else:
            print(f"[INFO] Stored results in modality '{modality_name}'")

        if self.model.device.type == "cuda":
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        pyro.clear_param_store()

        print("Finished fit_trans.")
