# Trans Function Overlay - Remaining Work

## Completed ✅
1. Created `predict_trans_function()` - general utility that works with all function types (additive_hill, single_hill, polynomial)
2. Updated `plot_negbinom_xy()` - uses general trans overlay instead of hardcoded Hill
3. Updated `plot_binomial_xy()` - added `show_trans_function` parameter and trans overlay
4. Updated `plot_normal_xy()` - added `show_trans_function` parameter and trans overlay
5. Updated `plot_multinomial_xy()` - added `show_trans_function` parameter to signature

## Remaining Work ⏳

### 1. Add Trans Overlay to Multinomial (plot_multinomial_xy)
Location: `bayesDREAM/plotting/xy_plots.py:805`

After the plotting loop for each category, add trans overlay:
```python
# Trans function overlay (if trans model fitted) - multinomial not yet supported
# if show_trans_function and not corrected:
#     warnings.warn("Trans function overlay for multinomial not yet implemented")
```

### 2. Add Trans Overlay to Mvnormal (plot_mvnormal_xy)
Location: `bayesDREAM/plotting/xy_plots.py:1040`

Add `show_trans_function` parameter to function signature (line 944), then after plotting loop:
```python
# Trans function overlay (if trans model fitted)
if show_trans_function and not corrected:
    x_range = np.linspace(x_true.min(), x_true.max(), 100)
    y_pred = predict_trans_function(model, feature, x_range, modality_name=modality.name)

    if y_pred is not None and y_pred.ndim == 1:
        # For mvnormal, overlay same function on first dimension only
        ax.plot(np.log2(x_range), y_pred,
               color='blue', linestyle='--', linewidth=2,
               label='Fitted Trans Function' if d == 0 else None)
```

### 3. Update plot_xy_data() Function Signatures
Location: `bayesDREAM/plotting/xy_plots.py`

Lines to update:
- Line 1191-1201: `plot_binomial_xy` call - add `show_trans_function=show_hill_function`
- Line 1204-1216: `plot_multinomial_xy` call - add `show_trans_function=show_hill_function`
- Line 1219-1229: `plot_normal_xy` call - add `show_trans_function=show_hill_function`
- Line 1232-1243: `plot_mvnormal_xy` call - add `show_trans_function=show_hill_function`

### 4. Update Docstring for plot_xy_data()
Location: `bayesDREAM/plotting/xy_plots.py:1098`

Change:
```python
show_hill_function : bool
    Overlay Hill function if trans model fitted (negbinom only, default: True)
```

To:
```python
show_hill_function : bool
    Overlay fitted trans function if trans model fitted (all distributions, default: True)
    Works with all function types: additive_hill, single_hill, polynomial
```

### 5. Optional: Add NTC Gradient Parameter

Add parameter to all distribution-specific functions and main `plot_xy_data()`:
```python
show_ntc_gradient : bool
    Color lines by NTC proportion in k-NN window (default: False)
    Lighter = more NTC cells, Darker = fewer NTC cells
```

Implementation requires:
- Add 'target' column tracking to k-NN smoothing
- Compute NTC proportion in each k-NN window
- Use `plot_colored_line()` instead of `ax.plot()` when enabled
- Add colorbar showing "1 - Proportion NTC"

### 6. Update Documentation

When logos are added to repository:
- Add logo to README.md header
- Add logo to docs/index.md (if exists)
- Update docs/PLOTTING_REQUIREMENTS.md with completion status
- Consider adding logo to package `__init__.py` docstring

## Testing Needed

After completing above changes:
1. Test trans overlay on negbinom with additive_hill
2. Test trans overlay on negbinom with polynomial
3. Test trans overlay on binomial/normal with various function types
4. Test that overlay is skipped gracefully when trans not fitted
5. Test that multinomial/mvnormal handle trans overlay appropriately

## Notes

- `show_hill_function` parameter name is kept for backwards compatibility
- All trans function types are now supported automatically via `predict_trans_function()`
- Trans overlay only shown on uncorrected plots (corrected has technical adjustments applied)
- For negbinom, baseline A is shown as horizontal line when available
