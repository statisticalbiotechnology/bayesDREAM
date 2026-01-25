# Recent Changes Archive (2025-12)

Archived from OUTSTANDING_TASKS.md on 2025-01-23.

---

## Binomial/Multinomial Trans Fitting Bug Fixes (2025-12-12)

**Status**: Fixed in commits 204efc7, 1db8b2a, 59f08d9, b7b53da

**Changes**:
- Fixed critical alpha_y_full construction bug in binomial trans fitting (was using ones instead of zeros)
- Enabled multinomial technical correction (matching binomial behavior)
- Fixed losses_trans storage on model/modality objects
- Added diagnostic output for non-finite log_prob issues
- Improved Hill function plotting for binomial/multinomial distributions

See `docs/archive/2025-12/commits/` for detailed analysis of these fixes.

---

## Cis Modality Design (2025-11-10)

**Change**: bayesDREAM now uses a separate 'cis' modality for cis gene/feature modeling.

**Design**:
- During `bayesDREAM()` initialization, the 'cis' modality is extracted from the primary modality
- The primary modality contains only trans features (cis feature excluded)
- `fit_cis()` always uses the 'cis' modality, regardless of primary modality type

**Benefits**:
- Consistent interface: `fit_cis()` works the same for gene, ATAC, or any modality
- Clear separation: cis vs trans features are explicitly separated
- Extensibility: Easy to support new modality types

**Parameters**:
```python
# For gene modality (default)
model = bayesDREAM(
    meta=meta,
    counts=gene_counts,
    cis_gene='GFI1B',
    guide_covariates=['cell_line']
)
# Creates: 'cis' modality (just GFI1B) + 'gene' modality (trans genes)

# For ATAC as primary modality (implemented via generic negbinom)
model = bayesDREAM(
    meta=meta,
    counts=atac_counts,
    modality_name='atac',
    cis_feature='chr9:123-456',
    feature_meta=region_meta,
    guide_covariates=['cell_line']
)
# Creates: 'cis' modality (chr9:123-456) + 'atac' modality (other regions)

# For any custom negbinom modality as primary
model = bayesDREAM(
    meta=meta,
    counts=custom_counts,
    modality_name='my_custom_modality',
    cis_feature='feature_123',
    feature_meta=feature_meta,
    guide_covariates=['cell_line']
)
# Creates: 'cis' modality (feature_123) + 'my_custom_modality' (other features)
```

---

## API Refactoring (2025-11-07)

**Change**: Cleaned up initialization parameters for clarity and extensibility.

**Removed Parameters**:
- `modalities` - Always start with empty dict, build from counts
- `primary_modality` - Replaced with `modality_name`
- `gene_meta` - Replaced with `feature_meta`

**New/Renamed Parameters**:
- `modality_name` (default='gene') - Name/type of primary modality
- `feature_meta` - General feature-level metadata for any modality
- `guide_covariates` - Now explicitly visible in signature (was implicit)

**Benefits**:
- Clearer intent: `modality_name='atac'` vs `primary_modality='atac'`
- More general: `feature_meta` works for genes, ATAC, transcripts, etc.
- Explicit parameters: `guide_covariates` no longer hidden
- Validation: Primary modality MUST be negbinom (enforced at initialization)
