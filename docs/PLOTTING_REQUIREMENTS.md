# Plotting Requirements Specification

## 1. Prior Sampling Implementation

**STATUS: ✅ COMPLETED**

All plotting functions now use actual prior samples from the Pyro model distributions.

### Implementation:
- ✅ Created `bayesDREAM/plotting/prior_sampling.py` module
- ✅ `sample_technical_priors()`: StudentT for alpha_y, Gamma for beta_o, etc.
- ✅ `sample_cis_priors()`: LogNormal for x_true, proper guide-level structure
- ✅ `sample_trans_priors()`: Hill/polynomial function priors with sparsity
- ✅ Updated `model_plots.py` to use real priors (NO MORE DUMMY WARNINGS)
- ✅ All prior distributions match Pyro model definitions

---

## 2. Raw X-Y Data Plotting

**STATUS: ✅ COMPLETED**

Plot raw data after `x_true` has been fit, showing relationship between cis gene expression and trans modality values.

### General Specifications:

**X-axis**: Always `log2(x_true)`
**Smoothing**: 100-cell k-nearest-neighbor smoothing by default (user-configurable)
**Technical Groups**: Color by technical groups set via `model.set_technical_groups(['cell_line'])`
**Technical Correction**: User chooses: uncorrected only, corrected only, or both

### Distribution-Specific Y-Axis:

| Distribution | Y-Axis | Notes |
|--------------|--------|-------|
| `negbinom` (counts) | `log2(expression)` | Log-scale normalized counts |
| `binomial` | `PSI` (percent spliced in) | Probability scale [0, 1] |
| `multinomial` | Separate plot per dimension | One subplot per category in a single row |
| `normal` | Raw value | Continuous measurement |
| `mvnormal` | Separate plot per dimension | One subplot per dimension in a single row |

### Technical Group Colors:

- **User-configurable color palette** (default: crimson for CRISPRa, dodgerblue for CRISPRi)
- **Informative legend labels**:
  - Single covariate: "CRISPRi", "CRISPRa", etc.
  - Multiple covariates: "CRISPRi:lane1", "CRISPRa:lane2", etc.
  - NOT generic "technical_group_0", "technical_group_1"

### Technical Correction Layout:

**If 1 plot (negbinom, binomial, normal)**:
- Uncorrected only: 1 plot
- Corrected only: 1 plot
- Both: 2 plots in 1 row (uncorrected | corrected)

**If multiple plots (multinomial, mvnormal with N dimensions)**:
- Uncorrected only: 1 row with N plots
- Corrected only: 1 row with N plots
- Both: 2 rows × N plots (row 1: uncorrected, row 2: corrected)

**Warning**: If `fit_technical()` not run for the modality, warn and only plot uncorrected.

### Smoothing Method (k-NN):

```python
def _knn_k(n, window):
    """Compute k for k-NN smoothing."""
    if n <= 0: return 1
    k = int(window) if isinstance(window, (int, np.integer)) else int(np.ceil(float(window) * n))
    return max(1, min(k, n))

def _smooth_knn(x, y, k):
    """k-NN smoothing along x-axis."""
    if len(x) == 0: return np.array([]), np.array([])
    order = np.argsort(x)
    x_sorted = np.asarray(x)[order].reshape(-1, 1)
    y_sorted = np.asarray(y)[order]
    k = max(1, min(k, len(x_sorted)))
    tree = cKDTree(x_sorted)
    y_hat = np.empty_like(y_sorted, dtype=float)
    for i in range(len(x_sorted)):
        _, idx = tree.query(x_sorted[i], k=k)
        y_hat[i] = np.nanmean(y_sorted[idx])
    return x_sorted.ravel(), y_hat
```

### Color Gradient for NTC Proportion:

For detailed view of NTC mixing within technical groups, use color gradient:
- Lighter = more NTC cells in k-NN window
- Darker = fewer NTC cells (more perturbed cells)
- Add colorbar: "1 - Proportion NTC (darker = fewer NTCs)"

---

## 3. Example: Splice Junction (Binomial)

**Y-axis**: PSI (percent spliced in) = `counts / denominator`

**Filters**:
- `min_denominator=3` (default, user-configurable)
- Remove non-finite values
- Remove x_true <= 0 (since we plot log2)

**Plot**:
```python
model.plot_xy_data(
    feature='chr1:12345:67890:+',  # SJ coordinate
    modality_name='splicing_sj',
    window=100,
    min_denominator=3,
    show_correction='both',  # 'uncorrected', 'corrected', 'both'
    color_palette={'CRISPRa': 'crimson', 'CRISPRi': 'dodgerblue'}
)
```

**Output**: PSI vs log2(x_true) with k-NN smoothing, colored by cell line

---

## 4. Example: Gene Counts (Negative Binomial)

**Y-axis**: `log2(expression)` where expression is normalized counts

**Normalization**:
- **Uncorrected**: `y_obs / sum_factor`
- **Corrected**: `y_obs / (sum_factor * alpha_y)`
- If `show_correction='both'`, refit sum factor after correction

**Optional Hill Function Overlay**:
If trans model has been fit, overlay predicted dose-response curve:
```python
# Extract posterior means of Hill parameters
A = posterior_samples_trans["A"].mean()
Vmax_a = posterior_samples_trans["Vmax_a"].mean()
K_a = posterior_samples_trans["K_a"].mean()
n_a = posterior_samples_trans["n_a"].mean()
# ... compute and plot Hill function
```

**Plot**:
```python
model.plot_xy_data(
    feature='TET2',
    modality_name='gene',
    window=100,
    show_correction='corrected',
    show_hill_function=True  # if trans model fitted
)
```

---

## 5. Example: Donor Usage (Multinomial)

**Y-axis**: Proportion for each acceptor category (sum to 1 within donor)

**Layout**: One subplot per acceptor in a single row

**Normalization**: Multinomial already represents proportions, no sum factor needed

**Plot**:
```python
model.plot_xy_data(
    feature='chr1:12345',  # donor site
    modality_name='splicing_donor',
    window=100,
    show_correction='uncorrected'  # technical fit may not apply to multinomial
)
```

**Output**: Row of N subplots (one per acceptor), each showing proportion vs log2(x_true)

---

## 6. Example: SpliZ Scores (Normal)

**Y-axis**: Raw SpliZ score (continuous)

**Plot**:
```python
model.plot_xy_data(
    feature='GFI1B',
    modality_name='spliz',
    window=100,
    show_correction='both'
)
```

---

## 7. Example: SpliZVD (Multivariate Normal)

**Y-axis**: Value for each dimension (z0, z1, z2)

**Layout**: 3 subplots in a row (one per dimension)

**Plot**:
```python
model.plot_xy_data(
    feature='GFI1B',
    modality_name='splizvd',
    window=100,
    show_correction='both'
)
```

**Output**:
- If uncorrected only: 1 row × 3 plots
- If corrected only: 1 row × 3 plots
- If both: 2 rows × 3 plots (6 total)

---

## 8. API Specification

### Method Signature:

```python
def plot_xy_data(
    self,
    feature: str,
    modality_name: Optional[str] = None,
    window: int = 100,
    show_correction: str = 'corrected',  # 'uncorrected', 'corrected', 'both'
    min_denominator: int = 3,  # for binomial only
    color_palette: Optional[Dict[str, str]] = None,
    show_hill_function: bool = True,  # for negbinom with trans fit
    xlabel: str = "log2(x_true)",
    figsize: Optional[Tuple[int, int]] = None,
    **kwargs
) -> plt.Figure:
    """
    Plot raw x-y data showing relationship between cis gene expression and modality values.

    Parameters
    ----------
    feature : str
        Feature name (gene, junction, donor, etc.)
    modality_name : str, optional
        Modality name (default: primary modality)
    window : int
        k-NN window size for smoothing (default: 100)
    show_correction : str
        'uncorrected': no technical correction
        'corrected': apply alpha_y technical correction
        'both': show both side-by-side
    min_denominator : int
        Minimum denominator for binomial distributions (default: 3)
    color_palette : dict, optional
        Custom colors for technical groups (e.g., {'CRISPRa': 'crimson', 'CRISPRi': 'dodgerblue'})
    show_hill_function : bool
        Overlay Hill function if trans model fitted (for negbinom only)
    xlabel : str
        X-axis label (default: "log2(x_true)")
    figsize : tuple, optional
        Figure size (auto-sized if None)
    **kwargs
        Additional plotting arguments

    Returns
    -------
    plt.Figure
        Matplotlib figure

    Raises
    ------
    ValueError
        If x_true not set (must run fit_cis first)
        If feature not found in modality
        If show_correction='corrected' but fit_technical not run

    Warnings
    --------
    If fit_technical not run for modality and show_correction='corrected',
    warns and plots uncorrected only.
    """
```

### Required Checks:

1. **x_true must exist**: Raise error if `self.x_true` is None
2. **Technical correction availability**:
   - If `show_correction='corrected'` or `'both'`
   - Check if modality has `alpha_y_prefit` available
   - If not: warn and fall back to uncorrected only
3. **Feature existence**: Check feature is in modality's feature metadata
4. **Technical groups**: Must have run `set_technical_groups()` before plotting

---

## 9. Implementation Checklist

### Phase 1: Prior Sampling ✅ COMPLETED
- ✅ Implement `sample_prior()` method for technical fit
- ✅ Implement `sample_prior()` method for cis fit
- ✅ Implement `sample_prior()` method for trans fit
- ✅ Update all plotting functions to use real priors
- ✅ Test prior sampling matches distribution definitions

### Phase 2: Raw X-Y Plotting ✅ COMPLETED
- ✅ Create `bayesDREAM/plotting/xy_plots.py`
- ✅ Implement k-NN smoothing utilities (`_knn_k`, `_smooth_knn`)
- ✅ Implement technical group label extraction (`get_technical_group_labels`)
- ✅ Implement negbinom plotting (with optional Hill overlay) - `plot_negbinom_xy`
- ✅ Implement binomial plotting (PSI) - `plot_binomial_xy`
- ✅ Implement multinomial plotting (multi-subplot) - `plot_multinomial_xy`
- ✅ Implement normal plotting - `plot_normal_xy`
- ✅ Implement mvnormal plotting (multi-subplot) - `plot_mvnormal_xy`
- ✅ Add `plot_xy_data()` method to PlottingMixin
- ⏳ Write comprehensive tests (next step)
- ✅ Add documentation examples (in docstrings)

### Key Features Implemented:
- ✅ User-configurable `window` parameter for k-NN smoothing
- ✅ User-configurable `min_counts` parameter for binomial/multinomial filtering
- ✅ Informative technical group labels (e.g., "CRISPRa" not "Group_0")
- ✅ Default color palette (crimson/dodgerblue) with user override option
- ✅ **Trans function overlay for ALL distributions (negbinom, binomial, normal, mvnormal)**
- ✅ **Supports ALL function types: additive_hill, single_hill, polynomial**
- ✅ **Automatic function type detection from posterior_samples_trans**
- ✅ Technical correction with warning if fit_technical not run
- ✅ Layout handling for single plots (both = 1×2) and multi-plots (both = 2×N)

---

## 10. Notes

- **Performance**: For large datasets (>10k cells), consider subsampling for visualization
- **Interactive**: Consider adding plotly/bokeh backend for interactive exploration
- **Batch plotting**: Add utility to plot multiple features at once
- **Export**: Support saving plots to PDF/PNG with high DPI

---

## 11. Reference Code Locations

**Splice Junction Example**: See user-provided code in conversation
**Gene Counts Example**: See user-provided code in conversation
**k-NN Smoothing**: `_smooth_knn()` and `_knn_k()` functions
**Color Gradient**: Use `matplotlib.collections.LineCollection` with colormap
