# NTC Gradient Coloring - Implementation Plan

## Status: Foundation Complete, Feature Implementation Pending

### Completed ✅
- Enhanced `_smooth_knn()` to track NTC proportions in k-NN windows (commit 7c24bbd)
- `plot_colored_line()` utility already exists for gradient coloring
- All trans function overlays completed (commit abd22f4)

### Remaining Implementation ⏳

## 1. Add Parameter to All Distribution Functions

Add `show_ntc_gradient: bool = False` parameter to:

### File: `bayesDREAM/plotting/xy_plots.py`

**Functions to update:**
1. `plot_negbinom_xy()` (line 495)
2. `plot_binomial_xy()` (line 607)
3. `plot_multinomial_xy()` (line 705)
4. `plot_normal_xy()` (line 824)
5. `plot_mvnormal_xy()` (line 941)
6. `plot_xy_data()` (line 1076) - add to signature and pass through

## 2. Implementation Pattern

For each distribution-specific function, follow this pattern:

```python
def plot_<distribution>_xy(
    model,
    feature: str,
    modality,
    x_true: np.ndarray,
    window: int,
    show_correction: str,
    color_palette: Dict[str, str],
    show_trans_function: bool,
    show_ntc_gradient: bool = False,  # ADD THIS
    xlabel: str,
    ax: Optional[plt.Axes] = None,
    **kwargs
):
    """
    Plot <distribution> ...

    Parameters
    ----------
    ...
    show_ntc_gradient : bool
        If True, color lines by NTC proportion in k-NN window (default: False)
        Lighter colors = more NTC cells, Darker colors = fewer NTC cells
        Only applies to uncorrected plots
    ...
    """

    # ... existing setup code ...

    # Detect NTC cells
    is_ntc = (model.meta['target'].str.lower() == 'ntc').values

    # Create colormap for gradient (if needed)
    if show_ntc_gradient:
        cmap = plt.cm.viridis_r  # Reversed: darker = fewer NTCs

    # Inside plotting loop for each technical group:
    def _plot_one(ax_plot, corrected):
        colorbar_added = False  # Track if colorbar added

        for group_code, group_label in zip(group_codes, group_labels):
            df_group = df[df['technical_group_code'] == group_code].copy()

            # ... compute y_plot ...

            # Get is_ntc for this group
            is_ntc_group = is_ntc[df_group.index]

            # k-NN smoothing with NTC tracking
            k = _knn_k(len(df_group), window)
            if show_ntc_gradient and not corrected:
                x_smooth, y_smooth, ntc_prop = _smooth_knn(
                    df_group['x_true'].values,
                    y_plot,
                    k,
                    is_ntc=is_ntc_group
                )

                # Use gradient coloring
                # Transform y_smooth appropriately (log2 for negbinom, etc.)
                lc = plot_colored_line(
                    x=np.log2(x_smooth),
                    y=<transformed_y_smooth>,
                    color_values=1 - ntc_prop,  # Darker = fewer NTCs
                    cmap=cmap,
                    ax=ax_plot,
                    linewidth=2
                )

                # Add colorbar (once per axis)
                if not colorbar_added:
                    fig = ax_plot.get_figure()
                    cbar = fig.colorbar(
                        plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1)),
                        ax=ax_plot
                    )
                    cbar.set_label('1 - Proportion NTC (darker = fewer NTCs)')
                    colorbar_added = True
            else:
                # Standard smoothing without NTC tracking
                x_smooth, y_smooth = _smooth_knn(
                    df_group['x_true'].values,
                    y_plot,
                    k
                )

                # Use standard coloring
                color = color_palette.get(group_label, f'C{group_code}')
                ax_plot.plot(
                    np.log2(x_smooth),
                    <transformed_y_smooth>,
                    color=color,
                    linewidth=2,
                    label=group_label
                )
```

## 3. Distribution-Specific Y-Axis Transforms

- **negbinom**: `transformed_y_smooth = np.log2(y_smooth)`
- **binomial**: `transformed_y_smooth = y_smooth` (PSI, no transform)
- **normal**: `transformed_y_smooth = y_smooth` (raw values)
- **multinomial**: Apply per-category in loop
- **mvnormal**: Apply per-dimension in loop

## 4. Special Cases

### Multinomial
- Apply gradient coloring per category subplot
- May need separate colorbar per subplot or shared colorbar

### Mvnormal
- Apply gradient coloring per dimension subplot
- Similar colorbar strategy as multinomial

### Corrected vs Uncorrected
- NTC gradient only makes sense for **uncorrected** plots
- When `show_correction='both'`, only apply gradient to uncorrected panel
- When `show_correction='corrected'`, skip gradient (warn user?)

## 5. Update plot_xy_data()

Add parameter and pass through to all distribution functions:

```python
def plot_xy_data(
    model,
    feature: str,
    modality_name: Optional[str] = None,
    window: int = 100,
    show_correction: str = 'corrected',
    min_counts: int = 3,
    color_palette: Optional[Dict[str, str]] = None,
    show_hill_function: bool = True,
    show_ntc_gradient: bool = False,  # ADD THIS
    xlabel: str = "log2(x_true)",
    figsize: Optional[Tuple[int, int]] = None,
    src_barcodes: Optional[np.ndarray] = None,
    **kwargs
) -> Union[plt.Figure, plt.Axes]:
    """
    ...
    show_ntc_gradient : bool
        Color lines by NTC proportion in k-NN window (default: False)
        Lighter = more NTC cells, Darker = fewer NTC cells
        Only applies to uncorrected plots
    ...
    """

    # ... existing code ...

    if distribution == 'negbinom':
        return plot_negbinom_xy(
            model=model,
            feature=feature,
            modality=modality,
            x_true=x_true,
            window=window,
            show_correction=show_correction,
            color_palette=color_palette,
            show_hill_function=show_hill_function,
            show_ntc_gradient=show_ntc_gradient,  # ADD THIS
            xlabel=xlabel,
            **kwargs
        )

    # ... repeat for all distributions ...
```

## 6. Documentation Updates

### Update Docstrings
- Add `show_ntc_gradient` parameter to all function docstrings
- Include examples showing usage

### Update docs/PLOTTING_REQUIREMENTS.md
- Mark NTC gradient as ✅ COMPLETED in Phase 2 checklist
- Update API specification section
- Add example usage in examples section

### Update PlottingMixin (bayesDREAM/plotting/model_plots.py)
- Update `plot_xy_data()` docstring to include `show_ntc_gradient`

## 7. Testing Checklist

After implementation:
- [ ] Test negbinom with gradient (GFI1B → TET2)
- [ ] Test binomial with gradient (SJ PSI plot)
- [ ] Test normal with gradient (SpliZ scores)
- [ ] Test multinomial with gradient (donor usage)
- [ ] Test mvnormal with gradient (SpliZVD)
- [ ] Test with `show_correction='both'` (gradient only on uncorrected)
- [ ] Test with `show_correction='corrected'` (no gradient)
- [ ] Verify colorbar appears and is labeled correctly
- [ ] Verify technical group colors are overridden when gradient enabled
- [ ] Test with datasets having no NTC cells (should handle gracefully)

## 8. Edge Cases to Handle

1. **No NTC cells**: If all `is_ntc == False`, ntc_prop will be 0 everywhere → all dark
2. **All NTC cells**: If all `is_ntc == True`, ntc_prop will be 1 everywhere → all light
3. **Corrected plots**: Skip gradient, use standard coloring
4. **Empty groups after filtering**: Handle gracefully (skip that group)
5. **Colorbar placement**: May need adjustment for multi-panel figures

## 9. Implementation Estimate

**Time**: 2-3 hours
**Lines of code**: ~150-200 lines (across all functions)
**Files modified**: 2
- `bayesDREAM/plotting/xy_plots.py`
- `docs/PLOTTING_REQUIREMENTS.md`

## 10. Alternative: Simplified Implementation

If full implementation is too complex, consider a **simplified version**:
- Implement for `negbinom` and `binomial` only (most common use cases)
- Mark multinomial/mvnormal as "not yet implemented" with warning
- This reduces scope by ~50%
