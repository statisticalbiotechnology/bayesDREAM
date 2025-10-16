# Per-Modality Fitting Storage Plan

## Problem Statement

Currently, `fit_technical()` and `fit_trans()` store results at the model level:
- `self.alpha_y_prefit`
- `self.posterior_samples_technical`
- `self.posterior_samples_trans`

This assumes a single modality. For multi-modal support, each modality needs its own fitting results.

## Proposed Architecture

### 1. Modality-Level Storage

Each `Modality` object should store its own fitting results:

```python
# Already added to Modality class:
modality.alpha_y_prefit = None          # Technical fit: overdispersion
modality.sigma_y_prefit = None          # Technical fit: variance (normal)
modality.cov_y_prefit = None            # Technical fit: covariance (mvnormal)
modality.posterior_samples_technical = None
modality.posterior_samples_trans = None
```

### 2. Model-Level Storage (Backward Compatibility)

For the primary modality (gene expression), maintain model-level storage:

```python
# bayesDREAM class:
self.alpha_y_prefit = None  # Points to primary modality results
self.posterior_samples_technical = None
self.posterior_samples_trans = None
```

These should be **references** or **aliases** to the primary modality's results.

### 3. Fitting Methods Modification

#### fit_technical()

**Current behavior:**
- Fits technical model using `self.counts`
- Stores results in `self.alpha_y_prefit`, `self.posterior_samples_technical`

**New behavior:**
- **MUST** know which modality to fit (different counts, cells, denominators!)
- Accept `modality_name` parameter (defaults to primary modality for backward compat)
- Get counts, denominator, distribution from the specified modality
- Handle cell subsetting: modality may have different cells than model.meta
- Store results in the specified modality
- For primary modality, also store at model level (backward compat)

```python
def fit_technical(
    self,
    sum_factor_col=None,
    distribution=None,  # Auto-detect from modality
    denominator=None,   # Auto-detect from modality
    modality_name=None, # Defaults to primary modality
    n_steps=5000,
    lr=0.01,
    device=None
):
    """
    Fit technical model.

    NOTE: technical_group_code MUST be set before calling this method.
    Call set_technical_groups(covariates) first!

    Parameters
    ----------
    modality_name : str, optional
        Name of modality to fit. If None, uses primary modality (for backward compat).
    distribution : str, optional
        Distribution type. If None, auto-detected from modality.
    denominator : np.ndarray, optional
        Denominator for binomial. If None, auto-detected from modality.
    """
    # Check technical_group_code exists
    if "technical_group_code" not in self.meta.columns:
        raise ValueError(
            "technical_group_code not set. Call set_technical_groups(covariates) before fit_technical()."
        )
    # Determine which modality to use
    if hasattr(self, 'modalities'):  # MultiModalBayesDREAM
        if modality_name is None:
            modality_name = self.primary_modality
        modality = self.get_modality(modality_name)

        # Get data from modality
        counts_to_fit = modality.counts
        distribution = distribution or modality.distribution
        denominator_to_use = denominator if denominator is not None else modality.denominator

        # Get cell names from modality (may differ from model.meta!)
        if modality.cell_names is not None:
            modality_cells = modality.cell_names
        else:
            # Modality doesn't have cell names, assume same as model
            modality_cells = self.meta['cell'].values

        # Subset meta to cells in this modality
        meta_subset = self.meta[self.meta['cell'].isin(modality_cells)].copy()

        # Further subset to NTC cells
        meta_ntc = meta_subset[meta_subset['target'] == 'ntc'].copy()

        # Subset sum_factor to match NTC cells in this modality
        if sum_factor_col is not None and sum_factor_col in meta_ntc.columns:
            sum_factors_ntc = meta_ntc[sum_factor_col].values
        else:
            sum_factors_ntc = None

    else:  # Regular bayesDREAM
        modality = None
        counts_to_fit = self.counts
        distribution = distribution or 'negbinom'
        denominator_to_use = denominator
        meta_ntc = self.meta[self.meta['target'] == 'ntc'].copy()
        sum_factors_ntc = meta_ntc[sum_factor_col].values if sum_factor_col else None

    # ... existing fitting code using counts_to_fit, meta_ntc, sum_factors_ntc ...

    # Store results
    if modality is not None:
        # Store in modality
        modality.alpha_y_prefit = posterior_samples["alpha_y"]
        modality.posterior_samples_technical = posterior_samples

        # Mark exon skipping aggregation as locked
        if modality.is_exon_skipping():
            modality.mark_technical_fit_complete()

        # If primary modality, also store at model level (backward compat)
        if modality_name == self.primary_modality:
            self.alpha_y_prefit = modality.alpha_y_prefit
            self.posterior_samples_technical = modality.posterior_samples_technical
    else:
        # Regular bayesDREAM - store at model level
        self.alpha_y_prefit = posterior_samples["alpha_y"]
        self.posterior_samples_technical = posterior_samples
```

#### fit_trans()

**Current behavior:**
- Uses `self.alpha_y_prefit` from technical fit
- Stores results in `self.posterior_samples_trans`

**New behavior:**
- Should accept optional `modality_name` parameter
- Access technical fit results from the modality (not `self.alpha_y_prefit`)
- Store trans results in the modality

```python
def fit_trans(
    self,
    sum_factor_col=None,
    function_type='additive_hill',
    distribution='negbinom',
    denominator=None,
    modality_name=None,  # NEW
    n_steps=10000,
    lr=0.01,
    device=None
):
    """
    Fit trans model.

    Parameters
    ----------
    modality_name : str, optional
        Name of modality to fit. If None, uses primary modality.
    """
    # Determine which modality to use
    if hasattr(self, 'modalities'):
        if modality_name is None:
            modality_name = self.primary_modality
        modality = self.get_modality(modality_name)
        counts_to_fit = modality.counts

        # Get technical fit results from modality (NOT self.alpha_y_prefit!)
        alpha_y_prefit = modality.alpha_y_prefit
        if alpha_y_prefit is None:
            raise ValueError(f"Modality '{modality_name}' has not been fit with fit_technical()")
    else:
        modality = None
        counts_to_fit = self.counts
        alpha_y_prefit = self.alpha_y_prefit

    # ... existing fitting code using alpha_y_prefit ...

    # Store results
    if modality is not None:
        modality.posterior_samples_trans = posterior_samples_y

        # If primary modality, also store at model level (backward compat)
        if modality_name == self.primary_modality:
            self.posterior_samples_trans = modality.posterior_samples_trans
    else:
        self.posterior_samples_trans = posterior_samples_y
```

### 4. MultiModalBayesDREAM Enhancements

**No new methods needed!** The existing `fit_technical()` and `fit_trans()` methods work for all modalities.

Users just specify `modality_name`:

```python
# Set technical groups once (required before fit_technical)
model.set_technical_groups(['cell_line'])

# Fit primary modality (gene expression)
model.fit_technical()  # modality_name defaults to 'gene'
model.fit_trans(function_type='additive_hill')

# Fit splicing modality (technical_groups already set)
model.fit_technical(modality_name='splicing_donor')
model.fit_trans(function_type='additive_hill', modality_name='splicing_donor')
```

The methods auto-detect:
- Distribution type from modality
- Denominator from modality (if binomial)
- Cell subset from modality
- Correct sum factors for the modality's cells

### 5. Accessing Results

**Old way (still works for primary modality):**
```python
# Works for backward compatibility
alpha_y = model.alpha_y_prefit
posteriors = model.posterior_samples_trans
```

**New way (modality-specific):**
```python
# Set technical groups first
model.set_technical_groups(['cell_line'])

# Fit specific modality
model.fit_technical(modality_name='splicing_donor')
model.fit_trans(function_type='additive_hill', modality_name='splicing_donor')

# Access modality-specific results
donor_mod = model.get_modality('splicing_donor')
alpha_y = donor_mod.alpha_y_prefit
posteriors = donor_mod.posterior_samples_trans
```

## Implementation Steps

1. ✅ Add storage attributes to Modality class
2. ✅ Modify `fit_technical()` to:
   - Accept `modality_name` parameter
   - Get counts/distribution/denominator from modality
   - Handle cell subsetting (modality cells may differ from model.meta)
   - Subset sum_factor correctly to modality's NTC cells
   - Store results in modality (and model-level for primary)
3. ✅ Modify `fit_trans()` to:
   - Accept `modality_name` parameter
   - Get counts/distribution/denominator from modality
   - Get technical fit results from modality (not self.alpha_y_prefit)
   - Handle cell subsetting
   - Store results in modality (and model-level for primary)
4. ✅ Update tests to verify per-modality storage (test_per_modality_fitting.py)
5. ✅ Update documentation

## Additional Implementation: Technical Group Management

Added `set_technical_groups(covariates)` method to explicitly set technical_group_code before fitting.
This simplifies the workflow:
- Call `set_technical_groups(['cell_line'])` once at the beginning
- `fit_technical()` REQUIRES technical_group_code to be set (raises error if not)
- `fit_cis()` and `fit_trans()` OPTIONALLY use technical_group_code if present
- NO `covariates` parameter in any fit method (clean separation of concerns)

## Backward Compatibility

**New multi-modal API:**

```python
from bayesDREAM import MultiModalBayesDREAM

model = MultiModalBayesDREAM(meta=meta, counts=counts, cis_gene='GFI1B')
model.add_splicing_modality(sj_counts, sj_meta, ...)

# Set technical groups ONCE (required before fit_technical)
model.set_technical_groups(['cell_line'])

# Fit primary modality (no covariates arg!)
model.fit_technical()  # Fits primary modality
model.fit_trans()      # Fits primary modality
# Results stored in both model-level AND modality-level (backward compat)

# Fit other modalities (technical_groups already set)
model.fit_technical(modality_name='splicing_donor')
model.fit_trans(modality_name='splicing_donor')
# Results stored in splicing_donor modality only
```

## Key Design Decisions

1. **Primary modality gets dual storage**: Results stored in both modality and model level for backward compatibility
2. **Non-primary modalities**: Results stored only in modality
3. **Distribution auto-detection**: When fitting a modality, use its distribution type if not explicitly specified
4. **Technical fit requirement**: Trans fitting checks that technical fit has been done for that modality
5. **Exon skipping lock**: When technical fit is complete, lock the aggregation method

## Testing Strategy

1. Test backward compatibility with existing single-modality code
2. Test per-modality fitting with multiple modalities
3. Test that technical fit results are correctly accessed in trans fitting
4. Test that primary vs non-primary modalities store correctly
5. Test error handling (trans without technical, invalid modality name, etc.)
