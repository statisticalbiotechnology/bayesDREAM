# ATAC Modality Design

## Implementation Status

**Current Status**: ✅ Core implementation complete (2025-10-17)

**Completed Features**:
- ✅ `add_atac_modality()` method with region type validation
- ✅ Support for promoter, gene_body, and distal region types
- ✅ `cis_feature` parameter in `fit_cis()` for flexible cis proxies
- ✅ Manual guide effects infrastructure (ready for Bayesian integration)
- ✅ Gene expression optional - ATAC-only workflows supported
- ✅ Test suite covering all major use cases

**Remaining Work**:
- ⏳ Bayesian integration of manual guide effects in `_model_x` (design decisions needed)
- ⏳ Documentation updates (QUICKSTART_MULTIMODAL.md, API_REFERENCE.md)

## Overview

Support for ATAC-seq data in bayesDREAM, allowing users to:
1. Use ATAC accessibility as a modality alongside or instead of gene expression
2. Use promoter accessibility as a proxy for cis gene expression
3. Provide manual guide effect estimates (log2FC) as priors

## Key Design Principles

1. **Gene expression is optional**: Users can run bayesDREAM with ATAC data only
2. **Flexible region types**: Support promoter, gene_body, and distal (enhancer) regions
3. **General guide effects**: Manual log2FC priors work for any modality, not just ATAC
4. **Modality consistency**: ATAC follows same patterns as other modalities (negative binomial distribution)

## Data Structures

### ATAC Counts
- **Format**: DataFrame or numpy array
- **Shape**: `(n_regions, n_cells)`
- **Values**: Fragment counts in genomic regions
- **Distribution**: `negbinom` (same as gene expression)

### Region Metadata (required)
DataFrame with the following columns:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `region_id` | str | Unique region identifier | `chr1:1000-2000` |
| `region_type` | str | Type: 'promoter', 'gene_body', 'distal' | `promoter` |
| `chrom` | str | Chromosome | `chr1` |
| `start` | int | Start coordinate (0-based) | `1000` |
| `end` | int | End coordinate | `2000` |
| `gene` | str | Associated gene (NA for distal) | `GFI1B` |
| `gene_name` | str | Gene symbol (optional) | `GFI1B` |
| `gene_id` | str | Ensembl ID (optional) | `ENSG00000168243` |

**Notes**:
- `region_type` must be one of: `'promoter'`, `'gene_body'`, `'distal'`
- `gene` is required for `promoter` and `gene_body` regions
- `gene` should be `NA`, `NaN`, or empty string for `distal` regions
- Can include additional annotation columns (e.g., `strand`, `tss_distance`)

### Manual Guide Effects (optional)
DataFrame with columns:

| Column | Type | Description |
|--------|------|-------------|
| `guide` | str | Guide RNA identifier |
| `log2FC` | float | Expected log2 fold-change vs NTC |

**Notes**:
- Used as priors in `fit_cis()`
- Strength of prior can be controlled via parameters
- Works for any modality, not just ATAC

## API Design

### 1. Create ATAC Modality

```python
model.add_atac_modality(
    atac_counts,           # DataFrame or array: (regions, cells)
    region_meta,           # DataFrame with required columns
    name='atac',           # Modality name (default: 'atac')
    cis_region=None        # Which region to use as cis proxy
)
```

**Parameters**:
- `atac_counts`: Fragment counts per region per cell
- `region_meta`: Metadata DataFrame (required columns listed above)
- `name`: Name for this ATAC modality (default: 'atac')
- `cis_region`: Region ID to use as cis gene proxy (e.g., 'promoter_GFI1B')

**Returns**: None (modality added to `self.modalities`)

### 2. Initialize with ATAC Only

```python
# Option A: Add ATAC after initialization
model = bayesDREAM(
    meta=meta,
    counts=None,              # No gene expression
    cis_gene='GFI1B',
    primary_modality='atac'
)
model.add_atac_modality(atac_counts, region_meta, cis_region='promoter_GFI1B')

# Option B: Pre-construct ATAC modality
atac_mod = create_atac_modality(atac_counts, region_meta)
model = bayesDREAM(
    meta=meta,
    modalities={'atac': atac_mod},
    cis_gene='GFI1B',
    primary_modality='atac'
)
```

### 3. Fit Cis with Manual Guide Effects

```python
# Prepare manual guide effects (optional)
guide_effects = pd.DataFrame({
    'guide': ['guide1', 'guide2', 'ntc'],
    'log2FC': [-2.5, -1.8, 0.0]
})

# IMPORTANT: fit_cis always uses the PRIMARY modality
# The cis_feature must exist in the primary modality
model.fit_cis(
    cis_feature='promoter_GFI1B',        # Use this region as cis proxy (must be in primary modality)
    manual_guide_effects=guide_effects,   # Optional: use as priors
    prior_strength=1.0                    # Optional: prior weight
)
```

**Important Design Choice**:
- **fit_cis ALWAYS uses the primary modality** (specified at initialization)
- This ensures the cis_gene/cis_feature is always in the modality being modeled
- For ATAC-only workflows, set `primary_modality='atac'` at initialization

**New Parameters**:
- `cis_feature`: Feature ID to use as cis proxy from the primary modality (replaces implicit cis_gene lookup)
- `manual_guide_effects`: DataFrame with guide → log2FC mappings
- `prior_strength`: Weight for manual guide effect priors (0=ignore, higher=stronger)
- `modality_name`: DEPRECATED - ignored, always uses primary modality

### 4. Combined Gene + ATAC Analysis

```python
# Initialize with gene expression
model = bayesDREAM(
    meta=meta,
    counts=gene_counts,
    cis_gene='GFI1B',
    primary_modality='gene'
)

# Add ATAC as additional modality
model.add_atac_modality(
    atac_counts=atac_counts,
    region_meta=region_meta
)

# Fit using gene expression (standard workflow)
model.fit_technical(covariates=['cell_line'], sum_factor_col='sum_factor')
model.fit_cis(sum_factor_col='sum_factor')
model.fit_trans(sum_factor_col='sum_factor_adj', function_type='additive_hill')

# Access ATAC data for downstream analysis
atac_mod = model.get_modality('atac')
promoter_regions = atac_mod.feature_meta[atac_mod.feature_meta['region_type'] == 'promoter']
```

## Implementation Steps

### Phase 1: ATAC Modality Creation
1. ✅ Design data structures
2. ✅ Implement `add_atac_modality()` method
3. ✅ Add region metadata validation
4. ⏳ Create `create_atac_modality()` helper function (optional convenience)

### Phase 2: Generalize Cis Fitting
1. ✅ Add `cis_feature` parameter to `fit_cis()`
2. ✅ Support feature lookup by ID (not just gene name)
3. ✅ Add `manual_guide_effects` parameter
4. ✅ Implement infrastructure for prior incorporation in `_model_x` (pseudocode)

### Phase 3: Enable Gene Expression Optional
1. ✅ Update `__init__` to handle `counts=None`
2. ✅ Update validation to allow non-gene primary modalities
3. ✅ Ensure base class methods work without gene counts (placeholder approach)

### Phase 4: Testing & Documentation
1. ✅ Create test with ATAC-only workflow
2. ✅ Create test with Gene + ATAC workflow
3. ⏳ Add to QUICKSTART_MULTIMODAL.md
4. ⏳ Update API_REFERENCE.md

## Technical Considerations

### 1. Cis Feature Lookup
Current code assumes `cis_gene` is a gene name that exists in `counts.index`. We need to:
- Add `cis_feature` parameter to specify exact feature ID
- Look up feature in the appropriate modality's feature metadata
- Use that row as the cis expression for downstream modeling

### 2. Manual Guide Effects as Priors
Implementation options:
- **Option A**: Use as informative priors in Pyro model (Gaussian centered on log2FC)
- **Option B**: Use as initialization for guide-level means
- **Option C**: Combine with data using weighted average

**Recommendation**: Option A with configurable prior strength

### 3. Negative Binomial for ATAC
ATAC fragment counts follow similar distribution to RNA-seq:
- Overdispersed count data
- Need size factor normalization
- Same `fit_technical()` and `fit_trans()` infrastructure applies

### 4. Region Type Filtering
Users may want to:
- Fit models on promoter regions only
- Fit models on distal regions only
- Include all region types

**Solution**: Use modality subsetting:
```python
atac_promoters = atac_mod.get_feature_subset(
    atac_mod.feature_meta[atac_mod.feature_meta['region_type'] == 'promoter'].index
)
```

## Example Workflows

### Workflow 1: ATAC-Only Analysis
```python
from bayesDREAM import bayesDREAM
import pandas as pd

# Load data
meta = pd.read_csv('meta.csv')
atac_counts = pd.read_csv('atac_counts.csv', index_col=0)
region_meta = pd.read_csv('region_meta.csv')
guide_effects = pd.read_csv('guide_log2fc.csv')

# Initialize without gene expression
model = bayesDREAM(
    meta=meta,
    counts=None,
    cis_gene='GFI1B',
    primary_modality='atac'
)

# Add ATAC modality
model.add_atac_modality(
    atac_counts=atac_counts,
    region_meta=region_meta,
    cis_region='chr9:132283881-132284881'  # GFI1B promoter
)

# Fit technical
model.set_technical_groups(['cell_line'])
model.fit_technical(
    modality_name='atac',
    sum_factor_col='sum_factor'
)

# Fit cis with manual guide effects
# Note: fit_cis uses primary modality (atac)
model.fit_cis(
    cis_feature='chr9:132283881-132284881',  # Must exist in primary modality (atac)
    manual_guide_effects=guide_effects,
    prior_strength=1.0
)

# Fit trans
model.fit_trans(
    modality_name='atac',
    sum_factor_col='sum_factor_adj',
    function_type='additive_hill'
)
```

### Workflow 2: Gene Expression + ATAC
```python
# Initialize with genes
model = bayesDREAM(
    meta=meta,
    counts=gene_counts,
    cis_gene='GFI1B',
    primary_modality='gene'
)

# Add ATAC
model.add_atac_modality(atac_counts, region_meta)

# Fit on gene expression (standard)
model.set_technical_groups(['cell_line'])
model.fit_technical(sum_factor_col='sum_factor')
model.fit_cis(sum_factor_col='sum_factor')
model.fit_trans(sum_factor_col='sum_factor_adj', function_type='additive_hill')

# Access ATAC for downstream analysis
atac_mod = model.get_modality('atac')
gfi1b_promoter = atac_mod.feature_meta[
    (atac_mod.feature_meta['region_type'] == 'promoter') &
    (atac_mod.feature_meta['gene'] == 'GFI1B')
]
```

## Open Questions

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
