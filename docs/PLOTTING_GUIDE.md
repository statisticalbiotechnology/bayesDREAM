# Plotting Guide

bayesDREAM provides comprehensive plotting functions for visualizing model parameters and results across the three-step fitting pipeline.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Prior vs Posterior Plots](#prior-vs-posterior-plots)
3. [X-Y Data Plots](#x-y-data-plots)
4. [Advanced Features](#advanced-features)
5. [Troubleshooting](#troubleshooting)

---

## Quick Start

```python
from bayesDREAM import bayesDREAM
import pandas as pd

# Load data and create model
meta = pd.read_csv('meta.csv')
gene_counts = pd.read_csv('gene_counts.csv', index_col=0)
model = bayesDREAM(meta=meta, counts=gene_counts, cis_gene='GFI1B')

# Set technical groups and run pipeline
model.set_technical_groups(['cell_line'])
model.fit_technical()
model.fit_cis(sum_factor_col='sum_factor')
model.fit_trans(sum_factor_col='sum_factor_adj', function_type='additive_hill')

# Plot technical fit parameters
fig = model.plot_technical_fit('beta_o')
fig = model.plot_technical_fit('alpha_y', technical_group_index=0)

# Plot cis fit parameters
fig = model.plot_cis_fit('x_true')

# Plot trans fit parameters
fig = model.plot_trans_fit('theta', function_type='additive_hill')

# Plot raw x-y data
fig = model.plot_xy_data('TET2', window=100, show_hill_function=True)
```

---

## Prior vs Posterior Plots

These functions visualize the goodness-of-fit by comparing prior distributions to posterior samples from variational inference.

### Technical Fit Plots

Plot parameters from `fit_technical()`:

```python
# Plot beta_o (scalar parameter for NB overdispersion prior)
fig = model.plot_technical_fit('beta_o')

# Plot alpha_y for a specific technical group
fig = model.plot_technical_fit('alpha_y', technical_group_index=0)

# Plot alpha_y for all technical groups (2D heatmap)
fig = model.plot_technical_fit('alpha_y')

# Plot for specific modality
fig = model.plot_technical_fit('alpha_y', modality_name='splicing_sj')

# Customize ordering and subset features
fig = model.plot_technical_fit(
    'alpha_y',
    technical_group_index=0,
    order_by='mean',  # or 'difference', 'alphabetical'
    subset_features=['GFI1B', 'TET2', 'MYB']
)
```

**Parameters**:
- `param`: Parameter to plot (`'beta_o'`, `'alpha_x'`, `'alpha_y'`, `'mu_ntc'`, `'o_y'`)
- `modality_name`: Modality name (default: primary modality)
- `technical_group_index`: Index of technical group to plot (if None, plots all groups)
- `order_by`: Feature ordering (`'mean'`, `'difference'`, `'alphabetical'`, `'input'`)
- `subset_features`: List of features to plot
- `plot_type`: `'auto'`, `'violin'`, or `'scatter'`

### Cis Fit Plots

Plot parameters from `fit_cis()`:

```python
# Plot x_true (cis gene expression per guide)
fig = model.plot_cis_fit('x_true')

# Customize ordering
fig = model.plot_cis_fit('x_true', order_by='mean')
```

**Parameters**:
- `param`: Parameter to plot (`'x_true'`, `'mu_x'`, `'log2_x_eff'`)
- `order_by`: Guide ordering (`'mean'`, `'difference'`, `'alphabetical'`, `'input'`)

### Trans Fit Plots

Plot parameters from `fit_trans()`:

```python
# Plot theta (Hill function parameters)
fig = model.plot_trans_fit('theta', function_type='additive_hill')

# Plot for polynomial function
fig = model.plot_trans_fit('theta', function_type='polynomial')

# Subset to specific genes
fig = model.plot_trans_fit(
    'theta',
    subset_features=['GFI1B', 'TET2'],
    function_type='additive_hill'
)
```

**Parameters**:
- `param`: Parameter to plot (`'theta'`, `'gamma'`, `'mu_y'`)
- `modality_name`: Modality name (default: primary modality)
- `function_type`: Function used in fit (`'additive_hill'`, `'single_hill'`, `'polynomial'`)
- `subset_features`: List of features to plot
- `order_by`: Feature ordering
- `plot_type`: `'auto'`, `'violin'`, or `'scatter'`

---

## X-Y Data Plots

Visualize the relationship between cis gene expression (`x_true`) and trans modality values with k-NN smoothing.

### Basic Usage

```python
# Plot gene expression vs cis gene
fig = model.plot_xy_data('TET2', window=100)

# Plot splice junction PSI vs cis gene
fig = model.plot_xy_data(
    'chr1:999788:999865',
    modality_name='splicing_sj',
    window=100
)
```

### Technical Correction

Show uncorrected data, corrected data, or both side-by-side:

```python
# Uncorrected only
fig = model.plot_xy_data('TET2', show_correction='uncorrected')

# Corrected only (default)
fig = model.plot_xy_data('TET2', show_correction='corrected')

# Both side-by-side
fig = model.plot_xy_data('TET2', show_correction='both')
```

### Trans Function Overlay

Overlay the fitted dose-response function from `fit_trans()`:

```python
# With Hill function overlay (default: True)
fig = model.plot_xy_data('TET2', show_hill_function=True)

# Without overlay
fig = model.plot_xy_data('TET2', show_hill_function=False)
```

Works with all function types (`additive_hill`, `single_hill`, `polynomial`) and all distributions.

### NTC Gradient Coloring

Color lines by the proportion of NTC cells in each k-NN window:

```python
# Enable NTC gradient (only on uncorrected plots)
fig = model.plot_xy_data(
    'TET2',
    show_correction='uncorrected',
    show_ntc_gradient=True
)
```

- Darker colors = fewer NTC cells (more perturbed)
- Lighter colors = more NTC cells
- Adds colorbar: "1 - Proportion NTC (darker = fewer NTCs)"
- Currently implemented for: `negbinom`, `binomial`, `normal`

### Custom Colors

Customize technical group colors:

```python
fig = model.plot_xy_data(
    'TET2',
    color_palette={
        'CRISPRa': 'crimson',
        'CRISPRi': 'dodgerblue',
        'Control': 'gray'
    }
)
```

### Distribution-Specific Examples

#### Gene Counts (Negative Binomial)

```python
fig = model.plot_xy_data(
    feature='TET2',
    modality_name='gene',
    window=100,
    show_correction='both',
    show_hill_function=True
)
```

**Y-axis**: `log2(expression)` - log-scale normalized counts

#### Splice Junction PSI (Binomial)

```python
fig = model.plot_xy_data(
    feature='chr1:999788:999865',
    modality_name='splicing_sj',
    window=100,
    min_counts=3,  # minimum denominator filter
    show_correction='uncorrected'
)
```

**Y-axis**: PSI (percent spliced in) - probability scale [0, 1]

#### Donor/Acceptor Usage (Multinomial)

```python
fig = model.plot_xy_data(
    feature='chr1:12345',
    modality_name='splicing_donor',
    window=100
)
```

**Layout**: One subplot per category (acceptor) in a single row

#### SpliZ Scores (Normal)

```python
fig = model.plot_xy_data(
    feature='GFI1B',
    modality_name='spliz',
    window=100,
    show_correction='both'
)
```

**Y-axis**: Raw SpliZ score (continuous measurement)

#### SpliZVD (Multivariate Normal)

```python
fig = model.plot_xy_data(
    feature='GFI1B',
    modality_name='splizvd',
    window=100
)
```

**Layout**: 3 subplots in a row (one per dimension: z0, z1, z2)

### Complete API Reference

```python
model.plot_xy_data(
    feature: str,                          # Feature name (gene, junction, etc.)
    modality_name: str = None,             # Modality (default: primary)
    window: int = 100,                     # k-NN window size
    show_correction: str = 'corrected',    # 'uncorrected', 'corrected', 'both'
    min_counts: int = 3,                   # Min denominator (binomial) or total (multinomial)
    color_palette: dict = None,            # Custom colors for technical groups
    show_hill_function: bool = True,       # Overlay trans function
    show_ntc_gradient: bool = False,       # Color by NTC proportion
    xlabel: str = "log2(x_true)",          # X-axis label
    figsize: tuple = None,                 # Figure size (auto if None)
    **kwargs
) -> plt.Figure
```

---

## Advanced Features

### k-NN Smoothing

The `window` parameter controls the smoothing:

```python
# Light smoothing (50 cells)
fig = model.plot_xy_data('TET2', window=50)

# Heavy smoothing (200 cells)
fig = model.plot_xy_data('TET2', window=200)

# Proportional smoothing (10% of cells)
fig = model.plot_xy_data('TET2', window=0.1)
```

### Filtering Options

Control minimum thresholds for binomial and multinomial distributions:

```python
# Binomial: minimum denominator
fig = model.plot_xy_data(
    'chr1:999788:999865',
    modality_name='splicing_sj',
    min_counts=5  # only plot where denominator >= 5
)

# Multinomial: minimum total counts
fig = model.plot_xy_data(
    'chr1:12345',
    modality_name='splicing_donor',
    min_counts=10  # only plot where total >= 10
)
```

### Multiple Modalities

Plot the same feature across different modalities:

```python
# Gene expression
fig1 = model.plot_xy_data('GFI1B', modality_name='gene')

# SpliZ score
fig2 = model.plot_xy_data('GFI1B', modality_name='spliz')

# SpliZVD
fig3 = model.plot_xy_data('GFI1B', modality_name='splizvd')
```

---

## Troubleshooting

### Error: "x_true not set"

You must run `fit_cis()` before plotting x-y data:

```python
model.fit_cis(sum_factor_col='sum_factor')
fig = model.plot_xy_data('TET2')
```

### Error: "technical_group_code not set"

Set technical groups before fitting:

```python
model.set_technical_groups(['cell_line'])
model.fit_technical()
```

### Warning: "fit_technical not run, plotting uncorrected only"

This is expected if you haven't run `fit_technical()` for a modality. Either:

1. Run `fit_technical()` for that modality:
   ```python
   model.fit_technical(modality_name='splicing_sj')
   ```

2. Or explicitly request uncorrected plots:
   ```python
   fig = model.plot_xy_data('chr1:999788:999865', show_correction='uncorrected')
   ```

### Warning: "KDE failed, using histogram instead"

This occurs when data has very low variance (e.g., many identical values). The plot will automatically fall back to a histogram. This is expected behavior for:
- Features with zero variance in NTC cells
- Sparse datasets with many zeros

### Error: "Shape mismatch: alpha_y has X features, but modality has Y features"

Specify the correct modality explicitly:

```python
# Wrong - using primary modality alpha_y for different modality
fig = model.plot_technical_fit('alpha_y')  # default: primary modality

# Correct - specify the modality
fig = model.plot_technical_fit('alpha_y', modality_name='splicing_sj')
```

### Performance Issues with Large Datasets

For datasets with >10k cells, consider:

1. Reduce window size:
   ```python
   fig = model.plot_xy_data('TET2', window=50)  # instead of 100
   ```

2. Use proportional smoothing:
   ```python
   fig = model.plot_xy_data('TET2', window=0.05)  # 5% of cells
   ```

---

## See Also

- **[API_REFERENCE.md](API_REFERENCE.md)** - Complete API documentation
- **[QUICKSTART_MULTIMODAL.md](QUICKSTART_MULTIMODAL.md)** - Getting started guide
- **[DATA_ACCESS.md](DATA_ACCESS.md)** - Accessing fitted parameters
