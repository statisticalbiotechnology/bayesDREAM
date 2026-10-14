"""
Tests for fit_cis(cis_model='marginal'): runs in single-guide and high-MOI mode,
stores the same posterior keys as the hierarchical model, and recovers simulated
guide-level log2 expression.
"""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip('torch')
pyro = pytest.importorskip('pyro')

from bayesDREAM import bayesDREAM  # noqa: E402

X_NTC = 16.0
L2FC = np.array([0.0, 0.0, -3.0, -2.0, -1.0, -0.5])
NTC_ITERS = 2000


def _filler(rng, sf, n=8):
    return rng.negative_binomial(5, 5 / (5 + 2.0 ** rng.uniform(1, 5, n)[:, None] * sf[None, :]))


def _single_guide_data(seed=0, n_cells=1200):
    rng = np.random.default_rng(seed)
    guides = ['NTC_0', 'NTC_1', 'g0', 'g1', 'g2', 'g3']
    targets = ['ntc', 'ntc', 'CIS', 'CIS', 'CIS', 'CIS']
    gi = rng.integers(0, len(guides), n_cells)
    sf = 2.0 ** rng.normal(0, 0.4, n_cells)
    x = 2.0 ** rng.normal(np.log2(X_NTC) + L2FC[gi], 0.4)
    y = rng.negative_binomial(5, 5 / (5 + x * sf))
    cells = [f'c{i}' for i in range(n_cells)]
    counts = pd.DataFrame(np.vstack([y[None, :], _filler(rng, sf)]),
                          index=['CIS'] + [f'f{i}' for i in range(8)], columns=cells)
    meta = pd.DataFrame({'cell': cells, 'guide': np.array(guides)[gi], 'target': np.array(targets)[gi],
                         'cell_line': rng.choice(['A', 'B'], n_cells), 'sum_factor': sf})
    truth = pd.Series(np.log2(X_NTC) + L2FC, index=guides)
    return meta, counts, truth


def _high_moi_data(seed=0, n_cells=1500):
    rng = np.random.default_rng(seed)
    guides = ['ntc_0', 'ntc_1', 'g0', 'g1', 'g2', 'g3']
    G = len(guides)
    assign = np.zeros((n_cells, G), dtype=int)
    for i in range(n_cells):  # one or two guides per cell
        assign[i, rng.choice(G, rng.integers(1, 3), replace=False)] = 1
    sf = 2.0 ** rng.normal(0, 0.4, n_cells)
    x = 2.0 ** rng.normal(np.log2(X_NTC) + assign @ L2FC, 0.4)
    y = rng.negative_binomial(5, 5 / (5 + x * sf))
    cells = [f'c{i}' for i in range(n_cells)]
    counts = pd.DataFrame(np.vstack([y[None, :], _filler(rng, sf)]),
                          index=['CIS'] + [f'f{i}' for i in range(8)], columns=cells)
    meta = pd.DataFrame({'cell': cells, 'cell_line': rng.choice(['A', 'B'], n_cells), 'sum_factor': sf})
    guide_meta = pd.DataFrame({'guide': guides, 'target': ['ntc', 'ntc', 'CIS', 'CIS', 'CIS', 'CIS']},
                              index=guides)
    return meta, counts, assign, guide_meta


def _fit(model):
    pyro.set_rng_seed(0)
    model.set_technical_groups(['cell_line'])
    model.fit_ntc(sum_factor_col='sum_factor', niters=NTC_ITERS, nsamples=50)
    model.fit_cis(sum_factor_col='sum_factor', cis_model='marginal', niters=5000, nsamples=200)
    return model


@pytest.fixture(scope='module')
def single_guide_fit(tmp_path_factory):
    meta, counts, truth = _single_guide_data()
    model = bayesDREAM(meta=meta, counts=counts, cis_gene='CIS', label='marg_sg',
                       output_dir=str(tmp_path_factory.mktemp('sg')), device='cpu')
    return _fit(model), truth


@pytest.fixture(scope='module')
def high_moi_fit(tmp_path_factory):
    meta, counts, assign, guide_meta = _high_moi_data()
    model = bayesDREAM(meta=meta, counts=counts, guide_assignment=assign, guide_meta=guide_meta,
                       cis_gene='CIS', label='marg_hm', output_dir=str(tmp_path_factory.mktemp('hm')),
                       device='cpu')
    return _fit(model)


def test_posterior_keys_and_shapes(single_guide_fit):
    model, truth = single_guide_fit
    ps = model.posterior_samples_cis
    N, G = len(model.meta), len(truth)
    for k in ('x_true', 'log_x_true', 'x_eff_g', 'log2_x_eff_g', 'sigma_eff', 'phi_x'):
        assert k in ps
    assert ps['x_true'].shape == (200, N)
    assert ps['x_eff_g'].shape == (200, G)
    assert model.x_true.shape == (N,)
    assert torch.isfinite(ps['log_x_true']).all()


def test_single_guide_recovery(single_guide_fit):
    model, truth = single_guide_fit
    names = model.meta.groupby('guide_code')['guide'].first().sort_index().values
    est = model.posterior_samples_cis['log2_x_eff_g'].median(0).values.numpy()
    assert np.sqrt(np.mean((est - truth.loc[names].values) ** 2)) < 0.3


def test_high_moi_recovery(high_moi_fit):
    ps = high_moi_fit.posterior_samples_cis
    est = ps['log2_x_eff_g'].median(0).values.numpy()
    assert ps['x_true'].shape[1] == len(high_moi_fit.meta)
    assert np.sqrt(np.mean((est - (np.log2(X_NTC) + L2FC)) ** 2)) < 0.3


def test_invalid_cis_model(single_guide_fit):
    model, _ = single_guide_fit
    with pytest.raises(ValueError):
        model.fit_cis(sum_factor_col='sum_factor', cis_model='nope')
