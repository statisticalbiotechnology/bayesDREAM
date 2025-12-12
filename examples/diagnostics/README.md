# Diagnostic Tools

This directory contains diagnostic scripts for analyzing bayesDREAM model fits.

## Scripts

### plot_prior_vs_posterior.py

**Purpose**: Compare prior vs posterior distributions for specific features to diagnose trans fitting issues.

**Usage**:
```python
from examples.diagnostics.plot_prior_vs_posterior import plot_sj_prior_vs_posterior
from bayesDREAM import bayesDREAM

# Load fitted model
model = bayesDREAM.load('path/to/saved/model.h5')

# Plot for specific splice junction
fig = plot_sj_prior_vs_posterior(
    model,
    sj_id='chr1:12345:67890:+',  # Feature ID
    modality_name='splicing_sj',
    save_path='diagnostic.png'
)
```

**Output**: 4-panel figure showing:
1. A (baseline) - prior vs posterior distributions
2. upper_limit - prior vs posterior distributions
3. Vmax_sum (amplitude) - posterior distribution
4. Guide-level data distribution with fit overlays

**Use Cases**:
- Debugging flat fits (check if posteriors are stuck at priors)
- Verifying priors are reasonable (based on data percentiles)
- Checking if Vmax_sum is stuck at floor (< 1e-4)
- Understanding how well model captures data structure

**Created**: December 2025, during binomial trans fitting debugging

---

## See Also

- `tests/archive/` - Archived diagnostic scripts from earlier development
- `docs/PLOTTING_GUIDE.md` - Comprehensive plotting documentation
