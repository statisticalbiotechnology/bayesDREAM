"""
Benchmark the hierarchical cis model against fit_cis(cis_model='marginal').

Simulates a CRISPRi screen with NTC guides and targeting guides of known log2
fold-change. Per-cell true cis expression is lognormal around the guide level and
counts are negative binomial (the generative process assumed by _model_x).
fit_ntc is run once; all cis fits then use the same o_x and alpha_x_prefit.

Methods:
  current               fit_cis(cis_model='hierarchical'), fixed number of steps.
  marginal_nopool       fit_cis(cis_model='marginal', pool_scale=None)
  marginal_pool_learn   fit_cis(cis_model='marginal', pool_scale='learn')
  marginal_pool_1       fit_cis(cis_model='marginal', pool_scale=1.0)
See docs/CIS_MARGINAL_MODEL.md.

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
import torch

from bayesDREAM import bayesDREAM

NTC_ITERS = 20000


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

    # --- fit_cis(cis_model='marginal') with different guide pooling ---
    for name, pool in (('marginal_nopool', None), ('marginal_pool_learn', 'learn'), ('marginal_pool_1', 1.0)):
        pyro.set_rng_seed(seed)
        t0 = time.time()
        model.fit_cis(sum_factor_col='sum_factor', cis_model='marginal', pool_scale=pool, nsamples=1000)
        secs = time.time() - t0
        ps = model.posterior_samples_cis
        results.append(summarise(name, truth_g, q(ps['log2_x_eff_g'].numpy()), truth_c,
                                 q(ps['log_x_true'].numpy()), secs, len(model.loss_x)))
        err = np.abs(ps['log2_x_eff_g'].median(0).values.numpy() - truth_g)
        print(f"{name}: phi {float(ps['phi_x'].median()):.2f}, s {float(ps['sigma_x_cell'].median()):.2f}, "
              f"largest guide error {err.max():.2f} ({guide_names[err.argmax()]})")

    df = pd.DataFrame(results)
    df.insert(0, 'seed', seed)
    print(f"o_x from fit_ntc = {o_x:.3f} (phi {phi_hat:.2f}; true phi 5, true sigma_eff 0.5)")
    print("current sigma_eff medians:", np.round(sig, 2))
    print(df.to_string(index=False))
    df.to_csv(f'/tmp/bench_cis/results_seed{seed}_n{n_cells}_it{current_niters}.csv', index=False)
    return df


if __name__ == '__main__':
    a = sys.argv[1:]
    main(int(a[0]) if a else 0, int(a[1]) if len(a) > 1 else 3000, int(a[2]) if len(a) > 2 else 20000)
