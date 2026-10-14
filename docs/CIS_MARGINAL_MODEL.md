# Marginal cis model (`fit_cis(cis_model='marginal')`)

An opt-in alternative to the original cis model (`_model_x`), built for stochastic
variational inference. The default remains `cis_model='hierarchical'`.

## Model

Single-guide mode, guide g, cell i in guide g(i):

    l_g          ~ N(m0, 3)                 guide log2 level, m0 = median empirical guide level
    s            ~ HalfNormal(1)            cell-level SD of log2 x, shared by all guides
    phi          ~ LogNormal(log phi_ntc, 1)   (fixed at phi_ntc if refit_phi=False)
    log2 x_i     ~ N(l_g(i), s)
    y_i          ~ NB(phi, alpha_c(i) * sf_i * x_i)

High-MOI mode (cells carry several guides, assignment matrix A):

    b            ~ N(m0, 3)                 NTC log2 level
    d_g          ~ N(0, 3) for targeting guides, d_g = 0 for NTC guides
    log2 x_i     ~ N(b + sum_g A_ig d_g, s)

The per-cell latent log2 x_i is integrated out with Gauss-Hermite quadrature
(`n_quadrature` nodes, default 16), so the only latent variables are the guide
parameters, s and phi. The guide is a mean-field AutoNormal for a warm start,
followed by a full-rank Gaussian (AutoMultivariateNormal) over all parameters.
Each phase stops when the mean loss over the last window (250 steps mean-field,
500 full-rank) improves on the previous window by less than `tolerance`.

After the fit, per-cell draws of log2 x_i are taken from their exact conditional
posterior given each draw of the guide parameters, on a 64-point grid.

Stored in `posterior_samples_cis`: `x_true`, `log_x_true` ([S, N]), `x_eff_g`,
`log2_x_eff_g`, `sigma_eff` ([S, G], s repeated per guide), `sigma_x_cell`, `phi_x`
and in high-MOI mode `log2_x_ntc`. `model.x_true` is the per-cell median, as before.

## Why

Problems in the hierarchical model under SVI:

- Non-centred guide effects with a HalfCauchy scale and StudentT(3) noise form a
  funnel; with many cells per guide the posterior of (sigma, eps_g) is strongly
  curved and a mean-field guide cannot represent it.
- One mu is shared by NTC and knockdown guides, which form separate clusters.
- The Gamma prior on sigma_eff with Exponential hyperpriors is barely identified.
- The NTC fit is pure NB, so phi_ntc already contains cell-level variation; the cis
  model then adds lognormal variation on top (double counting).
- N per-cell latents in a mean-field guide decouple the cell and guide posteriors,
  which makes the guide-level intervals far too narrow.
- Fixed 100k steps at lr 1e-3, no stopping rule.

## Benchmark

`examples/diagnostics/benchmark_cis_fit.py`: 4 NTC and 12 targeting guides
(log2FC 0 to -4, one small guide), two cell lines, true cell SD 0.5, true phi 5.
Means over seeds 1 to 3; hierarchical model run for 30k steps.

| cells | method       | guide RMSE | guide 90% cov. | cell RMSE | cell 90% cov. | seconds |
|-------|--------------|-----------:|---------------:|----------:|--------------:|--------:|
| 3000  | hierarchical | 0.124      | 0.42           | 0.466     | 0.76          | 294     |
| 3000  | marginal     | 0.119      | 0.88           | 0.442     | 0.91          | 66      |
| 600   | hierarchical | 0.294      | 0.56           | 0.466     | 0.85          | 273     |
| 600   | marginal     | 0.411      | 0.83           | 0.466     | 0.88          | 25      |

The marginal model gives near-nominal coverage and is 4 to 11 times faster. With few
cells per guide its point estimates are worse than the hierarchical model's, because it
does not pool information across guides; the hierarchical model shrinks small guides
towards the common mean. A light, centred pooling prior per target would address this.

Not yet tested: real data, minibatching, more than one cis feature.
