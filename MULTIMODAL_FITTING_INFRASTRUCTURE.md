# Multi-Modal Fitting Infrastructure

**Date**: 2025-10-13
**Status**: Infrastructure Complete, Ready for Distribution-Specific Implementation

## Summary

I've successfully structured the bayesDREAM codebase to support multi-modal fitting across different molecular modalities (genes, transcripts, splicing, custom measurements). The infrastructure is now in place to "plug in" different distribution-specific implementations without breaking the existing gene expression (negative binomial) workflow.

## What Was Done

### 1. Created `bayesDREAM/distributions.py` (~700 lines)

This new module provides distribution-specific observation samplers for Pyro models:

**Supported Distributions:**
- `negbinom`: Negative binomial (gene counts, transcript counts)
- `multinomial`: Categorical/proportional (isoform usage, donor/acceptor usage)
- `binomial`: Binary outcomes (exon skipping, raw SJ counts)
- `normal`: Continuous measurements (SpliZ scores)
- `mvnormal`: Multivariate normal (SpliZVD z0, z1, z2)

**Key Components:**
```python
# Get appropriate sampler for a distribution and model type
sampler = get_observation_sampler('multinomial', 'trans')

# Check if distribution requires denominator
requires_denominator('binomial')  # True

# Check if distribution uses 3D data structure
is_3d_distribution('multinomial')  # True

# Access full registry
DISTRIBUTION_REGISTRY  # Dictionary mapping distributions to samplers
```

**Design Philosophy:**
- **Function types** (Hill, polynomial) are **shared** across all modalities
- **Observation likelihoods** are **distribution-specific**
- Each distribution has two samplers:
  - `sample_<dist>_technical()`: For NTC-only technical fitting
  - `sample_<dist>_trans()`: For trans effects modeling

### 2. Extended `bayesDREAM/multimodal.py`

Added two new methods to `MultiModalBayesDREAM`:

#### `fit_modality_technical(modality_name, covariates, ...)`

Fits technical model for a specific modality.

**Current Behavior:**
- For primary modality with `negbinom` distribution: delegates to `fit_technical()`
- For other distributions: raises `NotImplementedError` with helpful message

**Future Implementation:**
Will use distribution-specific samplers from `distributions.py` to fit technical variation for any modality.

#### `fit_modality_trans(modality_name, sum_factor_col, function_type, ...)`

Fits trans model for a specific modality.

**Current Behavior:**
- For primary modality with `negbinom` distribution: delegates to `fit_trans()`
- For other distributions: raises `NotImplementedError` with helpful message

**Future Implementation:**
Will use distribution-specific samplers from `distributions.py`. The function type (how f(x) is parameterized) stays the same, but the observation model (how f(x) feeds into the likelihood) changes per distribution.

### 3. Updated Package Exports

Modified `bayesDREAM/__init__.py` to export distribution utilities:
```python
from bayesDREAM import (
    MultiModalBayesDREAM,
    get_observation_sampler,
    requires_denominator,
    is_3d_distribution,
    DISTRIBUTION_REGISTRY
)
```

### 4. Testing & Validation

Created `test_multimodal_fitting.py` to verify:
- ✅ All imports work correctly
- ✅ Distribution registry is functional
- ✅ Helper functions work as expected
- ✅ `MultiModalBayesDREAM` can be created
- ✅ New methods exist and are callable
- ✅ Gene modality excludes cis gene (for trans modeling)
- ✅ Base class retains cis gene (for cis modeling)
- ✅ Full backward compatibility with existing workflow

**Test Command:**
```bash
/opt/anaconda3/envs/pyroenv/bin/python test_multimodal_fitting.py
```

**Note**: Remember to use `/opt/anaconda3/envs/pyroenv/bin/python` for testing (pyroenv has torch/pyro).

### 5. Documentation Updates

Updated `MULTIMODAL_IMPLEMENTATION.md` with:
- New `distributions.py` file description
- Updated method list for `MultiModalBayesDREAM`
- Restructured limitations into: Completed ✅, In Progress 🚧, Future 🔮
- Clear roadmap for next steps

## Architecture Overview

### Current Implementation (Negative Binomial Only)

```
fit_technical() / fit_trans()
        ↓
  _model_technical / _model_y (Pyro models)
        ↓
  Hardcoded NegativeBinomial observation
        ↓
    y_obs ~ NegativeBinomial(phi, logits)
```

### Target Architecture (All Distributions)

```
fit_modality_technical(modality) / fit_modality_trans(modality)
        ↓
  Get modality distribution type
        ↓
  _model_technical / _model_y (Pyro models with distribution parameter)
        ↓
  Get sampler: get_observation_sampler(distribution, model_type)
        ↓
  Call distribution-specific sampler:
    - sample_negbinom_trans(y_obs, mu_y, phi_y, ...)
    - sample_multinomial_trans(y_obs, mu_y, ...)
    - sample_binomial_trans(y_obs, denominator, mu_y, ...)
    - sample_normal_trans(y_obs, mu_y, sigma_y, ...)
    - sample_mvnormal_trans(y_obs, mu_y, cov_y, ...)
```

## What Remains to Be Done

### Immediate Next Step: Integrate Samplers into Pyro Models

The key missing piece is modifying the Pyro models to accept a `distribution` parameter and use the appropriate sampler:

1. **Modify `_model_technical`** (in `model.py` line 712):
   - Add `distribution` parameter (default: 'negbinom')
   - Replace hardcoded `pyro.sample("y_obs_ntc", dist.NegativeBinomial(...))` (line 785-791)
   - Call appropriate sampler: `get_observation_sampler(distribution, 'technical')(...)`

2. **Modify `_model_y`** (in `model.py` line 1599):
   - Add `distribution` parameter (default: 'negbinom')
   - Replace hardcoded `pyro.sample("y_obs", dist.NegativeBinomial(...))` (line 1815-1822)
   - Call appropriate sampler: `get_observation_sampler(distribution, 'trans')(...)`

3. **Update `fit_modality_technical` and `fit_modality_trans`**:
   - Remove `NotImplementedError` for non-negbinom distributions
   - Pass modality distribution type to Pyro models
   - Handle distribution-specific parameters (denominator for binomial, covariance for mvnormal, etc.)

### Example Integration (Pseudocode)

```python
# In _model_y
def _model_y(
    self,
    N, T,
    y_obs_tensor,
    distribution='negbinom',  # NEW PARAMETER
    denominator_tensor=None,  # For binomial
    ...
):
    # ... existing code for computing mu_y from function types ...

    # Replace hardcoded NegativeBinomial sampling
    # OLD CODE (line 1815-1822):
    # pyro.sample(
    #     "y_obs",
    #     dist.NegativeBinomial(total_count=phi_y_used, logits=logits),
    #     obs=y_obs_tensor
    # )

    # NEW CODE:
    sampler = get_observation_sampler(distribution, 'trans')

    if distribution == 'negbinom':
        sampler(y_obs_tensor, mu_y, phi_y_used, alpha_y_full,
                groups_tensor, sum_factor_tensor, N, T, C)
    elif distribution == 'multinomial':
        sampler(y_obs_tensor, mu_y, alpha_y_full, groups_tensor, N, T, K, C)
    elif distribution == 'binomial':
        sampler(y_obs_tensor, denominator_tensor, mu_y, alpha_y_full,
                groups_tensor, N, T, C)
    # ... etc for other distributions
```

## Key Design Decisions

### 1. Function Types vs. Observation Models

**Decision**: Separate function parameterization (f(x)) from observation likelihoods.

**Rationale**:
- Hill functions and polynomials are biological dose-response models
- These should be **shared** across all modalities (genes, transcripts, splicing all respond to cis perturbation)
- Only the **observation model** (how f(x) maps to observed data) changes by distribution

### 2. Distribution Registry Pattern

**Decision**: Use a registry dictionary to map distributions to samplers.

**Rationale**:
- Easy to add new distributions without modifying core logic
- Centralized validation and metadata (requires_denominator, is_3d)
- Clear separation of concerns
- Testable in isolation

### 3. Backward Compatibility First

**Decision**: Maintain full backward compatibility with existing negbinom workflow.

**Rationale**:
- Users can continue using `bayesDREAM` or `MultiModalBayesDREAM` with gene expression
- Pipeline scripts (`run_technical.py`, `run_cis.py`, `run_trans.py`) work unchanged
- New functionality is opt-in (use `fit_modality_technical()` / `fit_modality_trans()`)

### 4. Infrastructure Before Implementation

**Decision**: Build complete infrastructure (API, samplers, tests, docs) before modifying Pyro models.

**Rationale**:
- Validates the design without risking breaking changes
- Provides clear roadmap for implementation
- Enables incremental development (can implement one distribution at a time)
- Easy to test each component independently

## Files Created/Modified

### New Files
- `bayesDREAM/distributions.py` (~700 lines)
- `test_multimodal_fitting.py` (~200 lines)

### Modified Files
- `bayesDREAM/multimodal.py`: Added `fit_modality_technical()` and `fit_modality_trans()` (~640 lines total)
- `bayesDREAM/__init__.py`: Added distribution exports
- `MULTIMODAL_IMPLEMENTATION.md`: Updated with new infrastructure and status

### Files Ready for Modification (Next Step)
- `bayesDREAM/model.py`: Will need to modify `_model_technical` (line 712) and `_model_y` (line 1599)

## Testing Notes

**Environment**: Use pyroenv conda environment
**Python Path**: `/opt/anaconda3/envs/pyroenv/bin/python`

**Run Tests:**
```bash
cd "/Users/lrosen/Library/Mobile Documents/com~apple~CloudDocs/Documents/Postdoc/bayesDREAM code/bayesDREAM_forClaude"
/opt/anaconda3/envs/pyroenv/bin/python test_multimodal_fitting.py
```

**Expected Output:**
```
Testing imports...
✓ All imports successful

Testing distribution registry...
✓ All distributions registered
✓ Helper functions work correctly
✓ get_observation_sampler works correctly

Creating toy data...
  - Metadata: 50 cells
  - Gene counts: 21 genes × 50 cells

Test 1: Create MultiModalBayesDREAM...
✓ MultiModalBayesDREAM created successfully

[... 7 tests total, all passing ...]

All tests passed! ✓
```

## Usage Example

```python
from bayesDREAM import MultiModalBayesDREAM

# Create model with gene expression (negbinom)
model = MultiModalBayesDREAM(
    meta=meta,
    counts=gene_counts,
    cis_gene='GFI1B',
    primary_modality='gene'
)

# Add other modalities
model.add_splicing_modality(sj_counts, sj_meta, ['donor', 'acceptor'])
model.add_transcript_modality(tx_counts, tx_meta, ['counts', 'usage'])

# Fit gene modality (works now - uses existing implementation)
model.fit_technical(covariates=['cell_line'])
model.fit_cis()
model.fit_trans(function_type='additive_hill')

# Fit other modalities (infrastructure ready, raises NotImplementedError with clear message)
model.fit_modality_trans('splicing_donor', function_type='additive_hill')
# NotImplementedError: Trans fitting for 'multinomial' distribution not yet implemented.
# Distribution-specific observation models are implemented in bayesDREAM/distributions.py.
# The missing piece is integrating these into the _model_y Pyro model.
```

## Summary

✅ **Infrastructure Complete**: The codebase is now structured to support multi-modal fitting.

✅ **Backward Compatible**: Existing workflows continue to work unchanged.

✅ **Distribution Samplers Ready**: All 5 distributions have implemented observation models.

✅ **API Defined**: `fit_modality_technical()` and `fit_modality_trans()` provide clear entry points.

✅ **Tested**: All infrastructure components validated.

✅ **Documented**: Clear roadmap and implementation guide.

🚧 **Next Step**: Integrate distribution-specific samplers into `_model_technical` and `_model_y` Pyro models.

The heavy lifting of architectural design is complete. Now you can incrementally implement distribution-specific fitting by modifying the Pyro models to use the samplers from `distributions.py`.
