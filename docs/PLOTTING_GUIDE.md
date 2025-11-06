# Plotting Guide

bayesDREAM provides comprehensive plotting functions organized by the three-step fitting pipeline: **Technical → Cis → Trans**.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Step 1: Technical Fit Plots](#step-1-technical-fit-plots)
3. [Step 2: Cis Fit Plots](#step-2-cis-fit-plots)
4. [Step 3: Trans Fit Plots](#step-3-trans-fit-plots)
5. [Color Scheme Management](#color-scheme-management)
6. [Advanced Features](#advanced-features)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

```python
from bayesDREAM import bayesDREAM
from bayesDREAM.plotting import ColorScheme, scatter_by_guide
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

# Step 1: Technical fit plots
fig = model.plot_technical_fit('alpha_y', technical_group_index=1)

# Step 2: Cis fit plots
fig = model.plot_cis_fit('x_true')
fig = scatter_by_guide(model, 'GFI1B', log2=True)

# Step 3: Trans fit plots
fig = model.plot_trans_fit('theta', function_type='additive_hill')
fig = model.plot_xy_data('TET2', window=100, show_hill_function=True)
```

---

## Step 1: Technical Fit Plots

Visualize results from `fit_technical()`, which estimates baseline overdispersion parameters from non-targeting control (NTC) cells.

### Prior vs Posterior: `model.plot_technical_fit()`

Compare prior and posterior distributions for technical parameters.

```python
# Plot beta_o (scalar NB overdispersion prior)
fig = model.plot_technical_fit('beta_o')

# Plot alpha_y for a specific technical group (e.g., K562)
fig = model.plot_technical_fit('alpha_y', technical_group_index=1)

# Plot alpha_y for all technical groups (2D heatmap)
fig = model.plot_technical_fit('alpha_y')

# Plot for specific modality
fig = model.plot_technical_fit('alpha_y', modality_name='splicing_sj')

# Customize ordering and subset features
fig = model.plot_technical_fit(
    'alpha_y',
    technical_group_index=1,
    order_by='mean',  # or 'difference', 'alphabetical', 'input'
    subset_features=['GFI1B', 'TET2', 'MYB']
)
```

**Parameters**:
- `param`: Parameter to plot (`'beta_o'`, `'alpha_y'`, `'mu_ntc'`, `'o_y'`)
- `modality_name`: Modality name (default: primary modality)
- `technical_group_index`: Technical group to plot (1, 2, ...). **Cannot be 0** (baseline group).
- `order_by`: Feature ordering (`'mean'`, `'difference'`, `'alphabetical'`, `'input'`)
- `subset_features`: List of features to plot
- `plot_type`: `'auto'`, `'violin'`, or `'scatter'`
- `metric`: Comparison metric (`'overlap'`, `'kl_divergence'`, `'posterior_coverage'`)

**Important Notes**:
- `technical_group_index=0` is the baseline/reference group with no variation (all values fixed at 1.0 for multiplicative, 0.0 for additive). Must specify index ≥1.
- For `negbinom`, alpha_y is plotted in log2 space: `log2(alpha_y_mult)`
- For additive distributions (`normal`, `binomial`), alpha_y is in additive (log2) space

### Sum Factor Diagnostics: `plot_sum_factor_comparison()`

Compare different sum factor normalization methods.

```python
from bayesDREAM.plotting import plot_sum_factor_comparison

# Compare two sum factor methods
fig = plot_sum_factor_comparison(
    model,
    cis_genes=['GFI1B', 'TET2'],
    sf_col1='clustered.sum.factor',
    sf_col2='sum_factor',
    label1='Clustered',
    label2='Basic'
)
```

---

## Step 2: Cis Fit Plots

Visualize results from `fit_cis()`, which estimates cis gene expression (`x_true`) for each guide.

### Prior vs Posterior: `model.plot_cis_fit()`

```python
# Plot x_true (cis gene expression per guide)
fig = model.plot_cis_fit('x_true')

# Customize ordering
fig = model.plot_cis_fit('x_true', order_by='mean')
```

**Parameters**:
- `param`: Parameter to plot (`'x_true'`, `'mu_x'`, `'log2_x_eff'`)
- `order_by`: Guide ordering (`'mean'`, `'difference'`, `'alphabetical'`, `'input'`)

### Scatter Plots

#### Mean vs Std: `scatter_by_guide()`

Raw data scatter plot colored by guide.

```python
from bayesDREAM.plotting import scatter_by_guide, ColorScheme

# Basic scatter plot
fig = scatter_by_guide(model, 'GFI1B', log2=False)

# Log2 scale
fig = scatter_by_guide(model, 'GFI1B', log2=True)

# With custom color scheme
cs = ColorScheme()
fig = scatter_by_guide(model, 'GFI1B', log2=True, color_scheme=cs)
```

Shows per-cell mean vs std of x_true samples.

#### Mean vs 95% CI: `scatter_ci95_by_guide()`

Uncertainty visualization.

```python
from bayesDREAM.plotting import scatter_ci95_by_guide

# With error bars (half-width CI)
fig = scatter_ci95_by_guide(model, 'GFI1B', log2=True)

# Full width CI
fig = scatter_ci95_by_guide(model, 'GFI1B', log2=True, full_width=True)
```

Shows per-cell mean vs 95% credible interval width.

### Distribution Plots

#### Violin Plot: `violin_by_guide_log2()`

Full posterior distribution by guide.

```python
from bayesDREAM.plotting import violin_by_guide_log2

# Violin plot (log2 scale, colored by target)
fig = violin_by_guide_log2(model, 'GFI1B')

# With custom colors
cs = ColorScheme()
fig = violin_by_guide_log2(model, 'GFI1B', color_scheme=cs)
```

Shows full posterior as violin plots in log2 space, colored by target.

#### Filled Density Plot: `filled_density_by_guide_log2()`

Smooth KDE density curves.

```python
from bayesDREAM.plotting import filled_density_by_guide_log2

# Density curves with fill
fig = filled_density_by_guide_log2(model, 'GFI1B')

# Custom bandwidth
fig = filled_density_by_guide_log2(model, 'GFI1B', bw=0.2)
```

Shows KDE densities colored by guide.

#### Per-Cell Density Lines: `plot_xtrue_density_by_guide()`

Vertical density lines for each cell, with guide color bar.

```python
from bayesDREAM.plotting import plot_xtrue_density_by_guide

# Cells grouped by guide
fig = plot_xtrue_density_by_guide(
    model,
    'GFI1B',
    log2=True,
    group_by_guide=True,
    color_scheme=cs
)

# Cells ordered by median only
fig = plot_xtrue_density_by_guide(
    model,
    'GFI1B',
    log2=True,
    group_by_guide=False
)
```

Shows one vertical density line per cell, with:
- Colored median ticks (guide-coded)
- Guide color bar between title and axes
- Legend mapping colors to guides

---

## Step 3: Trans Fit Plots

Visualize results from `fit_trans()`, which models downstream effects as functions of cis expression.

### Prior vs Posterior: `model.plot_trans_fit()`

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
- `modality_name`: Modality name (default: primary)
- `function_type`: Function used (`'additive_hill'`, `'single_hill'`, `'polynomial'`)
- `subset_features`: Features to plot
- `order_by`: Feature ordering
- `plot_type`: `'auto'`, `'violin'`, or `'scatter'`

### X-Y Relationship Plots: `model.plot_xy_data()`

Visualize relationship between cis expression (`x_true`) and trans modality values.

#### Basic Usage

```python
# Plot gene expression vs cis gene
fig = model.plot_xy_data('TET2', window=100)

# Plot splice junction PSI vs cis gene
fig = model.plot_xy_data(
    'chr1:999788:999865',
    modality_name='splicing_sj',
    window=100
)

# Plot SpliZ scores
fig = model.plot_xy_data('GFI1B', modality_name='spliz', window=100)
```

#### Technical Correction

Show uncorrected, corrected, or both:

```python
# Uncorrected only
fig = model.plot_xy_data('TET2', show_correction='uncorrected')

# Corrected only (default)
fig = model.plot_xy_data('TET2', show_correction='corrected')

# Both side-by-side
fig = model.plot_xy_data('TET2', show_correction='both')
```

#### Trans Function Overlay

Overlay fitted dose-response function:

```python
# With Hill function (default)
fig = model.plot_xy_data('TET2', show_hill_function=True)

# Without overlay
fig = model.plot_xy_data('TET2', show_hill_function=False)
```

Works with all function types (`additive_hill`, `single_hill`, `polynomial`).

#### NTC Gradient Coloring

Color by NTC proportion in k-NN window:

```python
# Enable NTC gradient (uncorrected only)
fig = model.plot_xy_data(
    'TET2',
    show_correction='uncorrected',
    show_ntc_gradient=True
)
```

Darker colors = fewer NTC cells (more perturbed).

#### Cell Subsetting

```python
# Plot only NTC cells
fig = model.plot_xy_data('TET2', subset_meta={'target': 'ntc'})

# Plot only CRISPRi cells
fig = model.plot_xy_data('GFI1B', subset_meta={'cell_line': 'CRISPRi'})
```

#### Distribution-Specific Examples

**Gene Counts (Negative Binomial)**:
```python
fig = model.plot_xy_data(
    'TET2',
    modality_name='gene',
    window=100,
    show_correction='both'
)
```
Y-axis: `log2(expression)`

**Splice Junction PSI (Binomial)**:
```python
fig = model.plot_xy_data(
    'chr1:999788:999865',
    modality_name='splicing_sj',
    window=100,
    min_counts=3  # minimum denominator
)
```
Y-axis: PSI [0, 1]

**Donor/Acceptor Usage (Multinomial)**:
```python
fig = model.plot_xy_data(
    'chr1:12345',
    modality_name='splicing_donor',
    window=100
)
```
Layout: One subplot per category

**SpliZ Scores (Normal)**:
```python
fig = model.plot_xy_data(
    'GFI1B',
    modality_name='spliz',
    window=100
)
```
Y-axis: Raw score

**SpliZVD (Multivariate Normal)**:
```python
fig = model.plot_xy_data(
    'GFI1B',
    modality_name='splizvd',
    window=100
)
```
Layout: 3 subplots (z0, z1, z2)

### Trans Parameter Analysis

Advanced analysis of Hill function parameters.

#### Helper Functions

```python
from bayesDREAM.plotting import (
    hill_xinf_samples,      # Compute inflection points
    dependency_mask_from_n, # Check if CI excludes 0
    abs_n_gt_tol_mask,      # Check if |median(n)| > threshold
    log2_pos,               # Log2 with NaN for ≤0
    hill_y                  # Vectorized Hill function
)

# Extract Hill parameters from posteriors
K_samps = model['GFI1B'].posterior_samples_trans['K_a'][:, 0, :].detach().cpu().numpy()
n_samps = model['GFI1B'].posterior_samples_trans['n_a'][:, 0, :].detach().cpu().numpy()

# Compute inflection points
xinf_samps = hill_xinf_samples(K_samps, n_samps, tol_n=0.2)

# Get masks for filtering
dep_mask = dependency_mask_from_n(n_samps, ci=95.0)  # CI excludes 0
steep_mask = abs_n_gt_tol_mask(n_samps, tol=1.0)     # |median(n)| > 2

# Convert to log2
log2_xinf = log2_pos(xinf_samps)

# Compute NaN fraction
frac_nan = np.mean(np.isnan(xinf_samps), axis=0)
```

#### Density Plot with X_true Context: `plot_parameter_density_with_xtrue()`

Two-panel plot: parameter density (left) + x_true distribution (right).

```python
from bayesDREAM.plotting import plot_parameter_density_with_xtrue

# Plot x_infl with x_true on right
fig = plot_parameter_density_with_xtrue(
    xinf_samps,
    model,
    'GFI1B',
    param_name='x_infl',
    subset_mask=dep_mask,
    log2=True,
    show_xtrue=True  # Show x_true panel
)

# Plot K_a with x_true
fig = plot_parameter_density_with_xtrue(
    K_samps,
    model,
    'GFI1B',
    param_name='K_a',
    subset_mask=dep_mask,
    log2=True
)

# Plot n without x_true panel
fig = plot_parameter_density_with_xtrue(
    n_samps,
    model,
    'GFI1B',
    param_name='n',
    subset_mask=dep_mask,
    log2=False,
    show_xtrue=False
)
```

**Parameters**:
- `param_samps`: Parameter samples (n_samples, n_features)
- `model`: bayesDREAM model
- `cis_gene`: Cis gene name
- `param_name`: Parameter name for labels
- `subset_mask`: Boolean mask (e.g., dependent genes)
- `log2`: Plot on log2 scale
- `show_xtrue`: Show x_true panel on right (default: True)
- `color_scheme`: ColorScheme for target colors

**Features**:
- Left: Vertical density lines for parameter
- Right: X_true density by target (colored)
- Shared y-axis for easy comparison
- Global 95% CI reference lines

#### Mean vs CI Scatter: `scatter_param_mean_vs_ci()`

Visualize parameter uncertainty vs magnitude.

```python
from bayesDREAM.plotting import scatter_param_mean_vs_ci
import numpy as np

# Example 1: x_infl with NaN fraction coloring
xinf_samps = hill_xinf_samples(K_samps, n_samps, tol_n=0.2)
dep_mask = dependency_mask_from_n(n_samps)
frac_nan = np.mean(np.isnan(xinf_samps), axis=0)

fig = scatter_param_mean_vs_ci(
    xinf_samps,
    param_name='x_infl',
    subset_mask=dep_mask,
    color_by=frac_nan,
    color_label='fraction NaN (lighter = more NaN)',
    cmap='Blues_r',  # darker = fewer NaNs
    log2=True,
    title='GFI1B — inflection point uncertainty'
)

# Example 2: Hill coefficient n with dependency coloring
fig = scatter_param_mean_vs_ci(
    n_samps,
    param_name='n (Hill coefficient)',
    subset_mask=dep_mask,  # Grey=not dependent, Blue=dependent
    log2=False,
    title='GFI1B — Hill coefficient uncertainty'
)

# Example 3: K_a without subsetting
fig = scatter_param_mean_vs_ci(
    K_samps,
    param_name='K_a',
    log2=True
)
```

**Parameters**:
- `param_samps`: Parameter samples (n_samples, n_features)
- `param_name`: Parameter name for labels
- `subset_mask`: Boolean mask (plots grey + colored groups)
- `color_by`: Values to color points by (e.g., NaN fraction)
- `color_label`: Colorbar label
- `cmap`: Colormap name
- `vmin`, `vmax`: Color scale limits
- `log2`: Whether params are on log2 scale (label only)
- `title`: Plot title

**Use Cases**:
- Inflection points with NaN fraction
- Hill coefficients with dependency
- K values with effect size
- Any parameter with custom coloring

#### Complete Trans Parameter Workflow

```python
from bayesDREAM.plotting import (
    hill_xinf_samples, dependency_mask_from_n, abs_n_gt_tol_mask,
    plot_parameter_density_with_xtrue, scatter_param_mean_vs_ci
)
import numpy as np

# 1. Extract parameters
K_samps = model['GFI1B'].posterior_samples_trans['K_a'][:, 0, :].detach().cpu().numpy()
n_samps = model['GFI1B'].posterior_samples_trans['n_a'][:, 0, :].detach().cpu().numpy()

# 2. Compute derived metrics
xinf_samps = hill_xinf_samples(K_samps, n_samps, tol_n=0.2)
dep_mask = dependency_mask_from_n(n_samps, ci=95.0)
steep_mask = abs_n_gt_tol_mask(n_samps, tol=1.0)
frac_nan = np.mean(np.isnan(xinf_samps), axis=0)

# 3. Filter to dependent + steep genes
mask = dep_mask & steep_mask

# 4. Plot x_infl density with x_true
fig1 = plot_parameter_density_with_xtrue(
    xinf_samps,
    model,
    'GFI1B',
    param_name='x_infl',
    subset_mask=mask,
    log2=True,
    show_xtrue=True
)

# 5. Plot x_infl mean vs CI (colored by NaN)
fig2 = scatter_param_mean_vs_ci(
    xinf_samps,
    param_name='x_infl',
    subset_mask=mask,
    color_by=frac_nan,
    color_label='fraction NaN',
    cmap='Blues_r',
    log2=True
)

# 6. Plot n mean vs CI
fig3 = scatter_param_mean_vs_ci(
    n_samps,
    param_name='n (Hill coefficient)',
    subset_mask=dep_mask,
    log2=False
)

# 7. Plot K_a density with x_true
fig4 = plot_parameter_density_with_xtrue(
    K_samps,
    model,
    'GFI1B',
    param_name='K_a',
    subset_mask=dep_mask,
    log2=True
)
```

### DE Comparison Plots

Compare bayesDREAM results with external methods (e.g., edgeR).

#### Scatter and Heatmap: `scatter_and_heatmap_edger_vs_bayes()`

```python
from bayesDREAM.plotting import scatter_and_heatmap_edger_vs_bayes, prepare_de_for_cg

# Prepare comparison data
de_data = prepare_de_for_cg(model, de_df, cis_gene='GFI1B')

# Create scatter + heatmap
fig = scatter_and_heatmap_edger_vs_bayes(
    cis_gene='GFI1B',
    model=model,
    de_df=de_df,
    fc_thresh=0.5,
    ntc_as_control=True
)
```

#### Full Range Comparison: `plot_edger_vs_bayes_full_range()`

Compare across full dose-response range (y(x→∞) vs y(x→0)).

```python
from bayesDREAM.plotting import plot_edger_vs_bayes_full_range

fig = plot_edger_vs_bayes_full_range(
    cis_genes=['GFI1B', 'TET2', 'MYB'],
    model=model,
    de_df=de_df,
    fc_thresh=0.5
)
```

#### Observed Range Comparison: `plot_edger_vs_bayes_observed_range()`

Compare within observed x_true range (y(x_max) vs y(x_min)).

```python
from bayesDREAM.plotting import plot_edger_vs_bayes_observed_range

fig = plot_edger_vs_bayes_observed_range(
    cis_genes=['GFI1B', 'TET2'],
    model=model,
    de_df=de_df
)
```

#### Log2FC Helper Functions

```python
from bayesDREAM.plotting import compute_log2fc_metrics, compute_log2fc_obs_for_cells

# Compute full-range and observed-range log2FC
log2fc_full, log2fc_obs, x_min, x_max = compute_log2fc_metrics(
    A_samps, alpha_samps, Vmax_samps, K_samps, n_samps,
    x_true_samps
)

# Compute log2FC for specific cell subset
log2fc_obs_guide, x_min, x_max = compute_log2fc_obs_for_cells(
    theta_samps, x_true_samps, cells_mask
)
```

---

## Color Scheme Management

Create and manage consistent color schemes across all plots.

### ColorScheme Class

```python
from bayesDREAM.plotting import ColorScheme, build_guide_colors
from matplotlib import cm
import numpy as np

# Option 1: Manual palette definition
palette = {
    'GFI1B': [cm.Greens(i) for i in np.linspace(0.4, 0.9, 3)],   # GFI1B_[1-3]
    'NTC':   [cm.Greys(i)  for i in np.linspace(0.4, 0.8, 5)],   # NTC_[1-5]
    'GEMIN5': [cm.Blues(i)  for i in np.linspace(0.4, 0.8, 2)],  # GEMIN5_[1-2]
    'DDX6':  [cm.Reds(i)   for i in np.linspace(0.4, 0.8, 3)],   # DDX6_[1,3]
}
cs = ColorScheme(palette=palette)

# Option 2: Auto-generate from target list
cs = ColorScheme.from_targets(
    targets=['GFI1B', 'TET2', 'MYB', 'NTC'],
    n_guides_per_target={'GFI1B': 3, 'TET2': 2, 'MYB': 3, 'NTC': 5}
)

# Access colors
guide_color = cs.get_guide_color('GFI1B_1')  # RGBA tuple
target_color = cs.get_target_color('GFI1B')   # Representative color
guide_colors_dict = cs.guide_colors           # All guide colors
target_colors_dict = cs.target_colors         # All target colors
```

### Using ColorScheme in Plots

```python
# Use in cis fit plots
fig = scatter_by_guide(model, 'GFI1B', color_scheme=cs)
fig = scatter_ci95_by_guide(model, 'GFI1B', color_scheme=cs)
fig = violin_by_guide_log2(model, 'GFI1B', color_scheme=cs)
fig = filled_density_by_guide_log2(model, 'GFI1B', color_scheme=cs)
fig = plot_xtrue_density_by_guide(model, 'GFI1B', color_scheme=cs)

# Use in trans parameter plots
fig = plot_parameter_density_with_xtrue(
    xinf_samps, model, 'GFI1B',
    color_scheme=cs
)
```

### Color Utilities

```python
from bayesDREAM.plotting import lighten, darken

# Lighten a color (mix with white)
light_blue = lighten('blue', amount=0.3)

# Darken a color (mix with black)
dark_blue = darken('blue', amount=0.3)
```

---

## Advanced Features

### k-NN Smoothing

Control smoothing in x-y plots:

```python
# Light smoothing (50 cells)
fig = model.plot_xy_data('TET2', window=50)

# Heavy smoothing (200 cells)
fig = model.plot_xy_data('TET2', window=200)

# Proportional smoothing (10% of cells)
fig = model.plot_xy_data('TET2', window=0.1)
```

### Filtering Options

Minimum thresholds for binomial/multinomial:

```python
# Binomial: minimum denominator
fig = model.plot_xy_data(
    'chr1:999788:999865',
    modality_name='splicing_sj',
    min_counts=5  # denominator >= 5
)

# Multinomial: minimum total
fig = model.plot_xy_data(
    'chr1:12345',
    modality_name='splicing_donor',
    min_counts=10  # total >= 10
)
```

### Multi-Panel Plots

Same feature across modalities:

```python
# Gene expression
fig1 = model.plot_xy_data('GFI1B', modality_name='gene')

# SpliZ score
fig2 = model.plot_xy_data('GFI1B', modality_name='spliz')

# SpliZVD
fig3 = model.plot_xy_data('GFI1B', modality_name='splizvd')
```

### Ordering and Subsetting

```python
# Order by effect size
fig = model.plot_technical_fit(
    'alpha_y',
    technical_group_index=1,
    order_by='difference'
)

# Subset features
fig = model.plot_technical_fit(
    'alpha_y',
    subset_features=['GFI1B', 'TET2', 'MYB', 'NFE2']
)

# Alphabetical
fig = model.plot_trans_fit(
    'theta',
    order_by='alphabetical',
    function_type='additive_hill'
)
```

---

## Troubleshooting

### Error: "x_true not set"

Must run `fit_cis()` before x-y plots:

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

### Error: "Cannot plot technical_group_index=0"

Baseline group has no variation:

```python
# Wrong
fig = model.plot_technical_fit('alpha_y', technical_group_index=0)

# Correct
fig = model.plot_technical_fit('alpha_y', technical_group_index=1)

# Or plot all non-baseline
fig = model.plot_technical_fit('alpha_y')
```

### Warning: "fit_technical not run, plotting uncorrected only"

Expected if technical fit not run for modality:

```python
# Option 1: Run technical fit
model.fit_technical(modality_name='splicing_sj')

# Option 2: Explicitly request uncorrected
fig = model.plot_xy_data(
    'chr1:999788:999865',
    show_correction='uncorrected'
)
```

### Warning: "KDE failed, using histogram instead"

Expected for low-variance data (many identical values).

### Error: "Shape mismatch: alpha_y has X features, but modality has Y features"

Specify correct modality:

```python
# Wrong
fig = model.plot_technical_fit('alpha_y')  # default: primary

# Correct
fig = model.plot_technical_fit('alpha_y', modality_name='splicing_sj')
```

### Error: "'Modality' object has no attribute 'cell_names'"

Provide cell identifiers when creating modalities:

```python
model.add_custom_modality(
    name='spliz',
    counts=spliz_scores,
    feature_meta=gene_meta,
    distribution='normal',
    cell_names=model.meta['cell'].tolist()  # Required
)
```

### Performance Issues (>10k cells)

```python
# Reduce window size
fig = model.plot_xy_data('TET2', window=50)

# Proportional smoothing
fig = model.plot_xy_data('TET2', window=0.05)

# Subset cells
fig = model.plot_xy_data('TET2', subset_meta={'cell_line': 'K562'})
```

---

## See Also

- **[API_REFERENCE.md](API_REFERENCE.md)** - Complete API documentation
- **[QUICKSTART_MULTIMODAL.md](QUICKSTART_MULTIMODAL.md)** - Getting started guide
- **[DATA_ACCESS.md](DATA_ACCESS.md)** - Accessing fitted parameters
- **[CLAUDE.md](../CLAUDE.md)** - Repository overview
