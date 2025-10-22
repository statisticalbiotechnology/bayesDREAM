# Modality-Specific Save/Load Verification

## Implementation Summary

The bayesDREAM save/load system has been successfully updated to support modality-specific saving and loading with the following capabilities:

### Key Features

1. **Selective Modality Saving**: Users can specify which modalities to save via a list
2. **Primary Modality NOT Default**: The primary modality is NOT automatically saved - it must be explicitly included in the modality list
3. **Model-Level Control**: Separate flag to control model-level backward-compatibility parameters
4. **Validation**: Automatically validates that requested modalities exist

### API Design

#### Save Methods

```python
model.save_technical_fit(
    output_dir=None,           # Where to save (default: self.output_dir)
    modalities=['gene', 'atac'], # Which modalities to save (default: all)
    save_model_level=True       # Save model-level params (default: True)
)

model.save_trans_fit(
    output_dir=None,
    modalities=['gene', 'atac'],
    save_model_level=True
)
```

#### Load Methods

```python
model.load_technical_fit(
    input_dir=None,            # Where to load from (default: self.output_dir)
    modalities=['gene', 'atac'], # Which modalities to load (default: all)
    load_model_level=True,     # Load model-level params (default: True)
    use_posterior=True          # Use posterior vs point estimates
)

model.load_trans_fit(
    input_dir=None,
    modalities=['gene', 'atac'],
    load_model_level=True
)
```

### Test Results

All tests passed (`test_modality_save_load.py`):

✅ **Test 1**: Create model with 3 modalities (gene, atac, splicing)
✅ **Test 2**: Fit technical on all 3 modalities
✅ **Test 3**: Save ONLY 'atac' and 'splicing' (exclude primary 'gene')
   - Verified: Primary NOT saved
   - Verified: Only requested modalities saved
   - Verified: Model-level params NOT saved with save_model_level=False

✅ **Test 4**: Save ONLY primary 'gene' modality explicitly
   - Verified: Only 'gene' saved when explicitly requested
   - Verified: Model-level params saved with save_model_level=True

✅ **Test 5**: Load ONLY 'atac' and 'splicing' from first save
   - Verified: Only requested modalities loaded
   - Verified: Primary 'gene' NOT loaded

✅ **Test 6**: Default save (modalities=None) saves ALL modalities
   - Verified: All modalities saved when no list provided

## Example Usage

### Example 1: Save Specific Modalities Only

```python
# Fit technical on multiple modalities
model.fit_technical(modality_name='gene', ...)
model.fit_technical(modality_name='atac', ...)
model.fit_technical(modality_name='splicing_donor', ...)

# Save only gene and ATAC (skip splicing)
model.save_technical_fit(modalities=['gene', 'atac'])
```

### Example 2: Exclude Primary Modality

```python
# Save everything EXCEPT the primary modality
model.save_technical_fit(
    modalities=['atac', 'splicing_donor'],  # Primary 'gene' NOT in list
    save_model_level=False                   # Skip model-level too
)
```

### Example 3: Load Subset of Modalities

```python
# Create new model instance
model2 = bayesDREAM(...)
model2.add_atac_modality(...)

# Load only ATAC modality
model2.load_technical_fit(modalities=['atac'], load_model_level=False)
```

### Example 4: Incremental Fitting and Saving

```python
# Fit and save modalities one at a time
model.fit_technical(modality_name='gene', ...)
model.save_technical_fit(modalities=['gene'], save_model_level=True)

model.fit_technical(modality_name='atac', ...)
model.save_technical_fit(modalities=['atac'], save_model_level=False)  # Don't overwrite model-level

model.fit_technical(modality_name='splicing_donor', ...)
model.save_technical_fit(modalities=['splicing_donor'], save_model_level=False)
```

## Files Saved

### Per-Modality Files

For `save_technical_fit()`:
- `alpha_y_prefit_{modality}.pt`: Overdispersion parameters for the modality
- `posterior_samples_technical_{modality}.pt`: Full posterior samples (if available)

For `save_trans_fit()`:
- `posterior_samples_trans_{modality}.pt`: Trans fit posterior samples

### Model-Level Files (when save_model_level=True)

For `save_technical_fit()`:
- `alpha_x_prefit.pt`: Cis gene overdispersion
- `alpha_y_prefit.pt`: Trans gene overdispersion (from primary modality)
- `posterior_samples_technical.pt`: Full posterior samples (from primary modality)

For `save_trans_fit()`:
- `posterior_samples_trans.pt`: Model-level posterior samples (from primary modality)

For `save_cis_fit()` (always model-level):
- `x_true.pt`: True cis gene expression
- `posterior_samples_cis.pt`: Cis fit posterior samples

## Behavior Verification

| Scenario | Expected Behavior | Verified |
|----------|------------------|----------|
| Save with `modalities=['atac']` | Only ATAC saved, primary NOT saved | ✅ |
| Save with `modalities=['gene', 'atac']` | Both gene and ATAC saved | ✅ |
| Save with `modalities=None` | ALL modalities saved | ✅ |
| Save with `save_model_level=False` | No model-level files created | ✅ |
| Save with `save_model_level=True` | Model-level files created | ✅ |
| Load with `modalities=['atac']` | Only ATAC loaded | ✅ |
| Load with `load_model_level=False` | No model-level params loaded | ✅ |
| Invalid modality name | Raises ValueError with available list | ✅ |

## Documentation

The following documentation has been updated:

1. **`docs/SAVE_LOAD_GUIDE.md`**:
   - Updated quick reference tables
   - Added detailed parameter documentation
   - Added "Example 4: Modality-Specific Save/Load"
   - Created "Modality-Specific Parameters" section with use cases

2. **`examples/README.md`**:
   - Added "Advanced: Modality-Specific Save/Load" section
   - Included examples of using `modalities` parameter

3. **Code Documentation**:
   - Updated docstrings for `save_technical_fit()`, `load_technical_fit()`
   - Updated docstrings for `save_trans_fit()`, `load_trans_fit()`

## Bug Fixes Included

While implementing this feature, two critical bugs were discovered and fixed:

1. **IndexError in fit_trans (line 3052)**: Added bounds checking for `alpha_y_prefit` sampling
2. **IndexError in fit_technical (line 1600)**: Fixed cis gene extraction to check modality features

## Conclusion

The modality-specific save/load functionality is **fully implemented and tested**. Users now have complete control over which modalities to persist, enabling:

- **Storage optimization**: Save only what's needed
- **Faster loading**: Load subset of modalities for targeted analysis
- **Incremental fitting**: Fit and save modalities independently
- **Compute resource flexibility**: Fit different modalities on different machines
- **Explicit control**: Primary modality must be explicitly included (not automatic)

The implementation correctly addresses the user's requirement:
> "which modalities to save is an argument where the user provides a list of modality names (primary should not always be saved on default, instead that would also be part of the list)"
