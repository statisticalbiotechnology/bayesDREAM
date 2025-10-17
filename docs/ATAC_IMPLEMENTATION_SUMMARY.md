# ATAC Modality Implementation Summary

**Date**: 2025-10-17
**Status**: ✅ Core implementation complete, ready for testing with real data

---

## What Was Implemented

### 1. ATAC Modality Creation (`add_atac_modality`)

**Location**: `bayesDREAM/model.py` lines 3736-3878

**Features**:
- Validates region metadata with required columns: `region_id`, `region_type`, `chrom`, `start`, `end`, `gene`
- Supports three region types: `promoter`, `gene_body`, `distal`
- Validates gene annotations for promoter/gene_body regions
- Applies zero-variance filtering (same as gene expression)
- Creates `negbinom` modality (fragment counts)
- Stores `cis_region` in `cis_feature_map` for later lookup

**Example**:
```python
model.add_atac_modality(
    atac_counts=atac_df,         # (regions, cells) counts
    region_meta=region_meta_df,  # Required columns specified above
    name='atac',
    cis_region='chr9:132283881-132284881'  # Optional: promoter for cis fitting
)
```

### 2. Enhanced `fit_cis()` Method

**Location**: `bayesDREAM/model.py` lines 1734-1942

**Important Design Choice**:
- **`fit_cis()` ALWAYS uses the PRIMARY modality** (specified at initialization)
- This ensures cis_gene/cis_feature is always in the modality being modeled
- For ATAC-only workflows, set `primary_modality='atac'` at initialization

**New Parameters**:
- `cis_feature`: Explicit feature ID to use as cis proxy from the primary modality (overrides cis_gene lookup)
- `manual_guide_effects`: DataFrame with columns `['guide', 'log2FC']` for priors
- `prior_strength`: Weight for manual guide effect priors (default: 1.0)
- `modality_name`: DEPRECATED - ignored, always uses primary modality (shows warning)

**Functionality**:
- Always uses the primary modality (enforced)
- Uses `cis_feature_map[primary_modality]` from `add_atac_modality()` if available
- Handles both DataFrame and numpy array modality counts
- Validates manual guide effects and converts to tensors
- Infrastructure ready for Bayesian integration (see pseudocode at lines 1917-1941)

**Example**:
```python
# Using ATAC promoter as cis proxy (ATAC is primary modality)
model.fit_cis(
    cis_feature='chr9:132283881-132284881',  # Must exist in primary modality
    manual_guide_effects=guide_df,            # Optional priors
    prior_strength=1.0
)
```

### 3. Gene Expression Optional

**Location**: `bayesDREAM/model.py` lines 3280-3407

**Changes**:
- `bayesDREAM.__init__()` accepts `counts=None`
- Creates placeholder counts matching meta['cell'] columns when needed
- Placeholder uses `cis_gene` as index to pass validation
- Allows primary modality to be added later via `add_atac_modality()`

**Example**:
```python
# ATAC-only workflow
model = bayesDREAM(
    meta=meta,
    counts=None,              # No gene expression
    cis_gene='GFI1B',
    primary_modality='atac'   # Will be added next
)

model.add_atac_modality(atac_counts, region_meta, cis_region='chr9:1000-2000')
```

### 4. Manual Guide Effects Infrastructure

**Location**: `bayesDREAM/model.py` lines 1876-1942

**Features**:
- Validates guide effects DataFrame (columns: `guide`, `log2FC`)
- Maps guide names to guide_code integers
- Creates tensors: `manual_guide_log2fc_tensor` (G,) and `manual_guide_mask_tensor` (G,)
- Mask indicates which guides have manual priors (1.0) vs not (0.0)
- Includes pseudocode for Bayesian integration in `_model_x`

**Design Decisions Deferred** (marked in pseudocode):
1. Prior standard deviation function (currently: `1.0 / prior_strength`)
2. Should priors override or combine with hierarchical priors?
3. Should NTC guide always have log2FC=0 enforced?
4. How to handle cell-line-specific effects with manual priors?

**Example**:
```python
guide_effects = pd.DataFrame({
    'guide': ['guide1', 'guide2', 'ntc'],
    'log2FC': [-2.5, -1.8, 0.0]
})

model.fit_cis(
    manual_guide_effects=guide_effects,
    prior_strength=1.0  # Equal weight with data
)
```

---

## Test Suite

**Location**: `tests/test_atac_modality.py`

**Tests Implemented**:
1. ✅ ATAC modality with gene expression (combined workflow)
2. ✅ Using ATAC promoter as cis proxy (parameter validation)
3. ✅ ATAC-only initialization without gene expression
4. ✅ Manual guide effects infrastructure validation

**All tests pass successfully.**

---

## Files Modified

### Core Implementation
1. **bayesDREAM/model.py**
   - Added `add_atac_modality()` method (~143 lines)
   - Enhanced `fit_cis()` signature and logic (~210 lines with infrastructure)
   - Modified `__init__()` for optional gene expression (~30 lines)

### Documentation
2. **docs/ATAC_DESIGN.md**
   - Comprehensive design document
   - Data structure specifications
   - API examples and workflows
   - Implementation status tracking

3. **ATAC_IMPLEMENTATION_SUMMARY.md** (this file)
   - Summary of completed work
   - Usage examples
   - Next steps

### Testing
4. **tests/test_atac_modality.py**
   - Complete test suite (~270 lines)
   - Covers all major use cases

---

## Example Workflows

### Workflow 1: Gene + ATAC (Combined)

```python
from bayesDREAM import bayesDREAM
import pandas as pd

# Initialize with gene expression
model = bayesDREAM(
    meta=meta,
    counts=gene_counts,
    cis_gene='GFI1B',
    primary_modality='gene'
)

# Add ATAC modality
model.add_atac_modality(
    atac_counts=atac_counts,
    region_meta=region_meta,
    name='atac'
)

# Fit using gene expression (standard workflow)
model.set_technical_groups(['cell_line'])
model.fit_technical(sum_factor_col='sum_factor')
model.fit_cis(sum_factor_col='sum_factor')
model.fit_trans(sum_factor_col='sum_factor_adj', function_type='additive_hill')

# Access ATAC data for downstream analysis
atac_mod = model.get_modality('atac')
promoters = atac_mod.feature_meta[atac_mod.feature_meta['region_type'] == 'promoter']
```

### Workflow 2: ATAC-Only

```python
# Initialize without gene expression
model = bayesDREAM(
    meta=meta,
    counts=None,              # No gene expression
    cis_gene='GFI1B',
    primary_modality='atac'
)

# Add ATAC modality
model.add_atac_modality(
    atac_counts=atac_counts,
    region_meta=region_meta,
    cis_region='chr9:132283881-132284881'  # GFI1B promoter
)

# Fit using ATAC promoter as cis proxy
# Note: fit_cis automatically uses primary modality (atac)
model.set_technical_groups(['cell_line'])
model.fit_technical(modality_name='atac', sum_factor_col='sum_factor')

model.fit_cis(
    cis_feature='chr9:132283881-132284881',  # From primary modality (atac)
    sum_factor_col='sum_factor'
)

model.fit_trans(
    modality_name='atac',
    sum_factor_col='sum_factor_adj',
    function_type='additive_hill'
)
```

### Workflow 3: With Manual Guide Effects

```python
# Prepare manual guide effects (e.g., from pilot experiments)
guide_effects = pd.DataFrame({
    'guide': ['guide1', 'guide2', 'guide3', 'ntc'],
    'log2FC': [-2.5, -1.8, -1.2, 0.0]
})

# Fit cis with manual priors
# Note: fit_cis uses primary modality automatically
model.fit_cis(
    cis_feature='chr9:132283881-132284881',  # From primary modality
    manual_guide_effects=guide_effects,
    prior_strength=1.0,  # Equal weight with data
    sum_factor_col='sum_factor'
)
```

---

## Key Technical Details

### Region Metadata Requirements

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `region_id` | str | Yes | Unique identifier (e.g., "chr1:1000-2000") |
| `region_type` | str | Yes | One of: 'promoter', 'gene_body', 'distal' |
| `chrom` | str | Yes | Chromosome |
| `start` | int | Yes | Start coordinate (0-based) |
| `end` | int | Yes | End coordinate |
| `gene` | str | Yes | Associated gene (empty string for distal) |
| `gene_name` | str | No | Gene symbol (optional) |
| `gene_id` | str | No | Ensembl ID (optional) |

### Distribution and Filtering

- **Distribution**: `negbinom` (same as gene expression)
- **Filtering**: Regions with zero variance across all cells are removed
- **Normalization**: Uses sum factors (same as gene expression)

### Cis Feature Lookup

**Important**: `fit_cis()` ALWAYS uses the primary modality.

Priority order for determining which feature to use:
1. Explicit `cis_feature` parameter in `fit_cis()`
2. `cis_feature_map[primary_modality]` (set by `add_atac_modality(..., cis_region='...')`)
3. `self.cis_gene` (must exist in primary modality's feature_meta)

---

## Next Steps

### High Priority
1. **Bayesian integration of manual guide effects**
   - Implement in `_model_x` (lines ~2040-2200 in model.py)
   - Make design decisions on prior combination
   - Test with simulated data to evaluate prior_strength values

2. **Documentation updates**
   - Add ATAC section to `docs/QUICKSTART_MULTIMODAL.md`
   - Update `docs/API_REFERENCE.md` with new parameters
   - Add manual guide effects section

### Medium Priority
3. **Real data testing**
   - Test with actual ATAC-seq data
   - Validate region type detection
   - Assess prior_strength recommendations

4. **Per-modality technical fitting**
   - Enable `fit_technical(modality_name='atac')` fully
   - ATAC-specific overdispersion parameters

### Low Priority
5. **Convenience functions**
   - `create_atac_modality()` helper
   - Automatic promoter detection from genomic coordinates
   - Support for peak files (BED format)

---

## Open Questions

These require experimentation with real data:

1. **Prior strength defaults**: What should be the default for `prior_strength`?
   - 0.0 = ignore manual effects entirely
   - 1.0 = equal weight with data
   - Higher = trust manual effects more

2. **Sum factor calculation for ATAC**: Should we use:
   - Total fragments per cell (analogous to total counts)
   - Median-of-ratios (like DESeq2)
   - User-provided values only?

3. **Multiple promoters per gene**: How to handle genes with multiple promoter regions?
   - Average accessibility?
   - Use most accessible?
   - Let user specify which one?

4. **Region overlap handling**: What if regions overlap?
   - Current design: treat as independent
   - Alternative: merge overlapping regions?

---

## Notes for Future Development

### Technical Debt
- The placeholder approach for optional gene expression works but could be cleaner
- Consider refactoring `_model_x` to explicitly support manual priors (not just pseudocode)

### Extensions
- Support for differential accessibility analysis
- Integration with enhancer-gene linking methods
- Multiple cis features per guide (e.g., promoter + enhancer)

### Compatibility
- All changes are backward compatible
- Existing workflows unchanged
- New features are opt-in via parameters
