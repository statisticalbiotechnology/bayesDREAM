"""
Benchmark the cis fit (fit_cis / _model_x) against a reparameterised alternative.

Simulates a CRISPRi screen with NTC guides and targeting guides of known log2
fold-change. Per-cell true cis expression is lognormal around the guide level and
counts are negative binomial (the generative process assumed by _model_x).
fit_ntc is run once; both cis fits then use the same o_x and alpha_x_prefit.

Methods:
  current   bayesDREAM.fit_cis as on main (AutoNormalMessenger, fixed niters).
  proposed  per-guide log2 level l_g ~ N(m0, 3), per-guide extra SD s_g ~ HalfNormal(1),
            per-cell latent integrated out by Gauss-Hermite quadrature, full-rank
            Gaussian guide over the 2G guide parameters, windowed stopping rule.
            Per-cell x_true posterior is computed afterwards on a grid, given draws
            of (l_g, s_g).

Metrics: RMSE and 90 % interval coverage for guide log2 x and per-cell log2 x_true,
and fit runtime.

Usage:
    python examples/diagnostics/benchmark_cis_fit.py [seed] [n_cells] [current_niters]
"""

import sys
import time

import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
from pyro.infer.autoguide import AutoMultivariateNormal, AutoNormal
from pyro.infer.autoguide.initialization import init_to_value

from bayesDREAM import bayesDREAM

NTC_ITERS = 20000
LN2 = float(np.log(2.0))


def simulate(seed=0, n_cells=3000, sigma_eff=0.5, phi=5.0, x_ntc=16.0, n_filler=10):
    rng = np.random.default_rng(seed)
    l2fc = np.concatenate([np.zeros(4), [-4, -3.5, -3, -2.5, -2, -1.5, -1, -0.75, -0.5, -0.25, -0.1, -3.0]])
    guides = [f"NTC_{i}" for i in range(4)] + [f"g{i}" for i in range(len(l2fc) - 4)]
    targets = ['ntc'] * 4 + ['CIS'] * (len(l2fc) - 4)
    # uneven guide sizes, last guide small
    p = np.ones(len(guides)); p[:4] = 2.0; p[-1] = 0.15
    gi = rng.choice(len(guides), n_cells, p=p / p.sum())
    cell_line = rng.choice(['A', 'B'], n_cells)
    alpha_cl = np.where(cell_line == 'B', 1.3, 1.0)
    sf = 2.0 ** rng.normal(0, 0.5, n_cells)
    l2x_g = np.log2(x_ntc) + l2fc
    l2x = rng.normal(l2x_g[gi], sigma_eff)
    mu = 2.0 ** l2x * sf * alpha_cl
    x_counts = rng.negative_binomial(phi, phi / (phi + mu))

    base = 2.0 ** rng.uniform(1, 6, n_filler)
    mu_f = base[:, None] * sf[None, :] * np.where(cell_line == 'B', 1.2, 1.0)[None, :]
    filler = rng.negative_binomial(5, 5 / (5 + mu_f))

    cells = [f"c{i}" for i in range(n_cells)]
    counts = pd.DataFrame(np.vstack([x_counts[None, :], filler]), index=['CIS'] + [f"f{i}" for i in range(n_filler)],
                          columns=cells)
    meta = pd.DataFrame({'cell': cells, 'guide': np.array(guides)[gi], 'target': np.array(targets)[gi],
                         'cell_line': cell_line, 'sum_factor': sf})
    truth_guide = pd.Series(l2x_g, index=guides)
    truth_cell = pd.Series(l2x, index=cells)
    return meta, counts, truth_guide, truth_cell


# ----------------------------------------------------------------------------------
# Proposed model
# ----------------------------------------------------------------------------------
def gh_nodes(K):
    z, w = np.polynomial.hermite_e.hermegauss(K)
    return torch.tensor(z, dtype=torch.float32), torch.tensor(np.log(w / np.sqrt(2 * np.pi)), dtype=torch.float32)


def proposed_model(y, log_offset, g, G, phi, m0, z, logw, free_phi=False, shared_s=False):
    if free_phi:  # NB dispersion re-estimated, prior centred on the NTC estimate
        phi = pyro.sample("phi", dist.LogNormal(torch.log(phi), 1.0))
    if shared_s:
        s = pyro.sample("s", dist.HalfNormal(1.0)).expand(G)  # one cell-level SD of log2 x
    with pyro.plate("guides", G):
        l = pyro.sample("l", dist.Normal(m0, 3.0))            # guide log2 level
        if not shared_s:
            s = pyro.sample("s", dist.HalfNormal(1.0))        # per-guide cell-level SD of log2 x
    # log mean per cell and quadrature node: [N, K]
    log_mu = LN2 * (l[g].unsqueeze(-1) + s[g].unsqueeze(-1) * z) + log_offset.unsqueeze(-1)
    lp = dist.NegativeBinomial(total_count=phi, logits=log_mu - torch.log(phi),
                               validate_args=False).log_prob(y.unsqueeze(-1))
    pyro.factor("obs", torch.logsumexp(lp + logw, dim=-1).sum())


def windowed_svi(svi, args, max_iters, window=500, tol=1e-4):
    losses = []
    for step in range(max_iters):
        losses.append(svi.step(*args))
        if (step + 1) % window == 0 and step + 1 >= 2 * window:
            prev, cur = np.mean(losses[-2 * window:-window]), np.mean(losses[-window:])
            if (prev - cur) / abs(prev) < tol:
                break
    return losses


def fit_proposed(y, sf, alpha_cell, g, G, phi, nsamples=1000, K=16, lr=0.01, max_iters=20000, warm=2000,
                 free_phi=False, shared_s=False):
    y = torch.as_tensor(np.asarray(y, dtype=np.float32))
    log_offset = torch.log(torch.as_tensor(sf * alpha_cell, dtype=torch.float32))
    g = torch.as_tensor(g, dtype=torch.long)
    phi = torch.tensor(float(phi))
    xn = (y / torch.exp(log_offset)).numpy()
    gn = g.numpy()
    l_init = np.array([np.log2(max(xn[gn == k].mean(), 1e-3)) for k in range(G)], dtype=np.float32)
    m0 = torch.tensor(float(np.median(l_init)))
    z, logw = gh_nodes(K)
    args = (y, log_offset, g, G, phi, m0, z, logw, free_phi, shared_s)

    pyro.clear_param_store()
    init = init_to_value(values={'l': torch.tensor(l_init), 's': torch.tensor(0.3) if shared_s else torch.full((G,), 0.3),
                                 'phi': phi})
    # Stage 1: mean-field warm start.  Stage 2: full-rank Gaussian started from it.
    mf = AutoNormal(proposed_model, init_loc_fn=init, init_scale=0.05)
    svi = pyro.infer.SVI(proposed_model, mf, pyro.optim.Adam({"lr": lr}), pyro.infer.Trace_ELBO())
    losses = windowed_svi(svi, args, warm, window=250)
    med = {k: v.detach() for k, v in mf.median().items()}
    init2 = init_to_value(values=med)
    pyro.clear_param_store()
    mvn = AutoMultivariateNormal(proposed_model, init_loc_fn=init2, init_scale=0.02)
    svi = pyro.infer.SVI(proposed_model, mvn, pyro.optim.Adam({"lr": lr / 2}), pyro.infer.Trace_ELBO())
    losses += windowed_svi(svi, args, max_iters)
    with torch.no_grad():
        draws = [mvn() for _ in range(nsamples)]
    l_s = torch.stack([d['l'] for d in draws])  # [S, G]
    s_s = torch.stack([d['s'].expand(G) for d in draws])
    if free_phi:
        phi = torch.stack([d['phi'] for d in draws]).median()
    return l_s, s_s, len(losses), (y, log_offset, g, phi)


def cell_posterior_quantiles(l_s, s_s, y, log_offset, g, phi, qs=(0.05, 0.5, 0.95), step=0.02, chunk=20):
    """Posterior quantiles of per-cell log2 x, mixing over draws of (l_g, s_g), on a grid."""
    lo = float(l_s.min() - 4 * s_s.max() - 3)
    hi = float(l_s.max() + 4 * s_s.max() + 3)
    v = torch.arange(lo, hi, step)  # [V]
    N = y.shape[0]
    dens = torch.zeros(N, v.shape[0])
    lik = dist.NegativeBinomial(total_count=phi, logits=LN2 * v + log_offset.unsqueeze(-1) - torch.log(phi),
                                validate_args=False).log_prob(y.unsqueeze(-1))  # [N, V]
    for i in range(0, l_s.shape[0], chunk):
        ls, ss = l_s[i:i + chunk, g], s_s[i:i + chunk, g].clamp(min=step)  # [c, N]
        lp = dist.Normal(ls.unsqueeze(-1), ss.unsqueeze(-1)).log_prob(v) + lik  # [c, N, V]
        dens += torch.softmax(lp, dim=-1).sum(0)
    cdf = torch.cumsum(dens / dens.sum(-1, keepdim=True), -1)
    out = [v[torch.searchsorted(cdf, torch.full((N, 1), q)).clamp(max=v.shape[0] - 1).squeeze(-1)] for q in qs]
    return [o.numpy() for o in out]


# ----------------------------------------------------------------------------------
def summarise(name, truth_g, q_g, truth_c, q_c, secs, iters):
    lo, med, hi = q_g
    r = dict(method=name,
             guide_rmse=float(np.sqrt(np.mean((med - truth_g) ** 2))),
             guide_cov90=float(np.mean((truth_g >= lo) & (truth_g <= hi))),
             guide_width90=float(np.mean(hi - lo)))
    lo, med, hi = q_c
    r.update(cell_rmse=float(np.sqrt(np.mean((med - truth_c) ** 2))),
             cell_cov90=float(np.mean((truth_c >= lo) & (truth_c <= hi))),
             cell_width90=float(np.mean(hi - lo)),
             seconds=round(secs, 1), iters=iters)
    return r


def main(seed=0, n_cells=3000, current_niters=20000):
    pyro.set_rng_seed(seed)
    meta, counts, truth_guide, truth_cell = simulate(seed, n_cells)
    model = bayesDREAM(meta=meta, counts=counts, cis_gene='CIS', label=f'bench{seed}',
                       output_dir='/tmp/bench_cis', device='cpu')
    model.set_technical_groups(['cell_line'])
    model.fit_ntc(sum_factor_col='sum_factor', niters=NTC_ITERS, nsamples=200)
    o_x = float(model.get_modality('cis').posterior_samples_ntc['o_x'].float().median())
    phi_hat = 1.0 / o_x ** 2

    guide_names = model.meta.groupby('guide_code')['guide'].first().sort_index().values
    truth_g = truth_guide.loc[guide_names].values
    truth_c = truth_cell.loc[model.meta['cell'].values].values
    g = model.meta['guide_code'].values
    G = len(guide_names)
    sf = model.get_modality('gene').sum_factors.loc[model.meta['cell'].values, 'sum_factor'].values
    alpha = model.alpha_x_prefit.flatten().cpu().numpy()[model.meta['technical_group_code'].values]

    results = []
    # --- current ---
    t0 = time.time()
    model.fit_cis(sum_factor_col='sum_factor', niters=current_niters, nsamples=1000)
    secs = time.time() - t0
    ps = model.posterior_samples_cis
    l2g = torch.log2(ps['x_eff_g'].reshape(-1, G)).numpy()
    l2c = ps['log_x_true'].reshape(-1, len(g)).numpy()
    q = lambda a: [np.quantile(a, p, axis=0) for p in (0.05, 0.5, 0.95)]
    results.append(summarise('current', truth_g, q(l2g), truth_c, q(l2c), secs, current_niters))
    sig = ps['sigma_eff'].reshape(-1, G).median(0).values.numpy()

    # --- fit_cis(cis_model='marginal'), the package version of proposed_phi_shared_s ---
    pyro.set_rng_seed(seed)
    t0 = time.time()
    model.fit_cis(sum_factor_col='sum_factor', cis_model='marginal', nsamples=1000)
    secs = time.time() - t0
    ps = model.posterior_samples_cis
    results.append(summarise('fit_cis_marginal', truth_g, q(ps['log2_x_eff_g'].numpy()), truth_c,
                             q(ps['log_x_true'].numpy()), secs, len(model.loss_x)))

    # --- proposed (phi fixed from fit_ntc, and phi re-estimated) ---
    y = counts.loc['CIS', model.meta['cell'].values].values
    extra = []
    for name, free_phi, shared_s in (('proposed', False, False), ('proposed_phi', True, False),
                                     ('proposed_phi_shared_s', True, True)):
        pyro.set_rng_seed(seed)
        t0 = time.time()
        l_s, s_s, iters, data = fit_proposed(y, sf, alpha, g, G, phi_hat, free_phi=free_phi, shared_s=shared_s)
        q_c = cell_posterior_quantiles(l_s, s_s, *data)
        secs = time.time() - t0
        results.append(summarise(name, truth_g, q(l_s.numpy()), truth_c, q_c, secs, iters))
        extra.append((name, float(data[3]), np.round(s_s.median(0).values.numpy(), 2)))

    df = pd.DataFrame(results)
    df.insert(0, 'seed', seed)
    print(f"o_x from fit_ntc = {o_x:.3f} (phi {phi_hat:.2f}; true phi 5, true sigma_eff 0.5)")
    print("current sigma_eff medians:", np.round(sig, 2))
    for name, ph, sg in extra:
        print(f"{name}: phi {ph:.2f}, s_g medians {sg}")
    print(df.to_string(index=False))
    df.to_csv(f'/tmp/bench_cis/results_seed{seed}_n{n_cells}_it{current_niters}.csv', index=False)
    return df


if __name__ == '__main__':
    a = sys.argv[1:]
    main(int(a[0]) if a else 0, int(a[1]) if len(a) > 1 else 3000, int(a[2]) if len(a) > 2 else 20000)
