# Debugging Poor Binomial Trans Fits

**Date**: November 28, 2025
**Status**: Critical bugs identified

## Summary

Almost all binomial trans fits show negative slopes despite data showing clear trends. Investigation revealed **critical bugs in the binomial additive_hill formulation**.

## Root Cause: Invalid Probability Generation

### The Problem

For binomial distributions with `function_type='additive_hill'`, the model computes:

```python
# In trans.py, lines 519-522
Hilla = Hill_based_positive(x_true, Vmax=1.0, A=0, K=K_a, n=n_a)  # Returns [0, 1]
Hillb = Hill_based_positive(x_true, Vmax=1.0, A=0, K=K_b, n=n_b)  # Returns [0, 1]
combined_hill = alpha * Hilla + beta * Hillb  # Can be [0, 2] !!
y_dose_response = A + Vmax_sum * combined_hill  # Can EXCEED 1 !!
```

### Why This Is Broken

1. **`alpha` and `beta` are RelaxedBernoulli variables** with typical values ~0.5 each
2. **If both are active and both Hills saturate**:
   - `combined_hill = 1.0 * 1.0 + 1.0 * 1.0 = 2.0`
   - `y_dose_response = A + (upper_limit - A) * 2.0 = 2*upper_limit - A`
   - **This can exceed 1.0!** (invalid probability)

3. **The sampler clamps invalid values** (distributions.py line 334):
   ```python
   mu_y_clamped = torch.clamp(mu_y, min=1e-6, max=1-1e-6)
   ```

4. **Clamping introduces bias**:
   - Breaks gradients (clamp has zero gradient in saturated region)
   - Model can't learn correct parameters
   - Causes poor fits with wrong slopes

### Evidence

- Comment in trans.py (line 511) says: "If alpha=beta=1 and both Hills=1, y = A + Vmax_sum = upper_limit"
- **This comment is WRONG!** It should be `y = A + Vmax_sum * (1 + 1) = A + 2*Vmax_sum`
- Line 530 explicitly says: `# NO clamp on combined_hill or y_dose_response`
- But the sampler DOES clamp, creating the mismatch

## Fixes Implemented

### 1. Added `min_denominator` Parameter

**File**: `bayesDREAM/fitting/trans.py`
**Lines**: 792, 810-813, 972-991

Allows filtering observations where denominator < threshold:

```python
model.fit_trans(
    sum_factor_col='sum_factor_adj',
    function_type='additive_hill',
    modality_name='splicing_sj',
    min_denominator=3  # Exclude observations with <3 reads
)
```

**How it works**:
- Sets `denominator=0` for filtered observations
- Updated binomial sampler (distributions.py lines 336-376) to mask observations with denominator=0
- Masked observations contribute 0 to log-likelihood

### 2. Fixed `predict_trans_function` to Match Model

**File**: `bayesDREAM/plotting/xy_plots.py`
**Lines**: 916-930

Previously computed Hill functions incorrectly. Now matches actual model:

```python
# Compute Hills with Vmax=1 (normalized)
Hill_a = Hill_based_positive(x_range, Vmax=1.0, A=0, K=K_a, n=n_a)
Hill_b = Hill_based_positive(x_range, Vmax=1.0, A=0, K=K_b, n=n_b)

# Combined can exceed 1!
combined_hill = alpha * Hill_a + beta * Hill_b

# Final prediction (can exceed upper_limit)
y_pred = A + Vmax_sum * combined_hill
```

### 3. Updated Binomial Sampler to Handle Masking

**File**: `bayesDREAM/fitting/distributions.py`
**Lines**: 293-376

Now properly handles observations with denominator=0:
- Creates `valid_mask = denominator_tensor > 0`
- Computes log-probability manually
- Zeros out masked observations before summing

## Debugging Code

### 1. Subset by Cell Line (CRISPRi or CRISPRa Only)

To test without technical correction:

```python
import pandas as pd
import numpy as np

# Load cis fit
model.load_cis_fit()

# Option A: Subset to CRISPRi only
crispri_mask = model.meta['cell_line'].str.contains('CRISPRi', case=False, na=False)
model_crispri = model.subset_cells(crispri_mask)

# Option B: Subset to CRISPRa only
crispra_mask = model.meta['cell_line'].str.contains('CRISPRa', case=False, na=False)
model_crispra = model.subset_cells(crispra_mask)

# Fit trans without technical correction (only 1 cell_line group)
model_crispri.fit_trans(
    sum_factor_col='sum_factor_adj',
    function_type='additive_hill',
    modality_name='splicing_sj'
)
```

**Note**: `subset_cells()` may not exist yet. Alternative approach:

```python
# Manual subsetting
crispri_cells = model.meta[model.meta['cell_line'].str.contains('CRISPRi', case=False, na=False)]['cell'].values

# Create new metadata and counts subsets
meta_subset = model.meta[model.meta['cell'].isin(crispri_cells)].copy()

# For each modality, subset counts
for mod_name in model.list_modalities():
    mod = model.get_modality(mod_name)
    if mod.cell_names is not None:
        # Find indices of cells to keep
        cell_indices = [i for i, cell in enumerate(mod.cell_names) if cell in crispri_cells]

        # Subset counts
        if mod.cells_axis == 1:
            counts_subset = mod.counts[:, cell_indices]
        else:
            counts_subset = mod.counts[cell_indices, :]

        mod.counts = counts_subset
        mod.cell_names = [mod.cell_names[i] for i in cell_indices]

# Update model metadata
model.meta = meta_subset

# Recreate guide-level metadata (required for fitting)
model._prepare_guide_metadata(model.get_modality(model.primary_modality))

# Now fit trans
model.fit_trans(
    sum_factor_col='sum_factor_adj',
    function_type='additive_hill',
    modality_name='splicing_sj'
)
```

### 2. Filter by Denominator Threshold

Use the new `min_denominator` parameter:

```python
model.fit_trans(
    sum_factor_col='sum_factor_adj',
    function_type='additive_hill',
    modality_name='splicing_sj',
    min_denominator=3  # Only use observations with ≥3 reads
)
```

This will print:
```
[INFO] Filtering observations with denominator < 3
[INFO] Masked 12345/50000 observations (24.7%)
```

## Recommended Next Steps

### Immediate Testing

1. **Test with min_denominator=3**: See if filtering low-coverage observations improves fits
2. **Test without technical correction**: Subset to CRISPRi or CRISPRa only to eliminate technical effects
3. **Check alpha and beta values**: Print posterior means to see if they sum to >1

```python
# After fitting
alpha_mean = model.modalities['splicing_sj'].posterior_samples_trans['alpha'].mean(dim=0)
beta_mean = model.modalities['splicing_sj'].posterior_samples_trans['beta'].mean(dim=0)
alpha_plus_beta = alpha_mean + beta_mean

print(f"Alpha + Beta range: [{alpha_plus_beta.min():.3f}, {alpha_plus_beta.max():.3f}]")
print(f"Alpha + Beta > 1: {(alpha_plus_beta > 1).sum()} / {len(alpha_plus_beta)} features")
```

### Longer-Term Fixes

The fundamental issue is that `combined_hill = alpha * Hill_a + beta * Hill_b` can exceed 1. Options:

#### Option 1: Constrain alpha + beta ≤ 1

Add a constraint during sampling:
```python
# Instead of independent alpha and beta
alpha = pyro.sample("alpha", RelaxedBernoulli(...))
beta_max = 1.0 - alpha
beta = pyro.sample("beta", RelaxedBernoulli(...)) * beta_max
```

#### Option 2: Use Normalized Weights

Make alpha and beta sum to 1 via softmax:
```python
logit_alpha = pyro.sample("logit_alpha", dist.Normal(0, 1))
logit_beta = pyro.sample("logit_beta", dist.Normal(0, 1))
weights = torch.softmax(torch.stack([logit_alpha, logit_beta]), dim=0)
alpha, beta = weights[0], weights[1]
```

#### Option 3: Clamp combined_hill

Add explicit clamping before computing y_dose_response:
```python
combined_hill = torch.clamp(alpha * Hilla + beta * Hillb, min=0.0, max=1.0)
y_dose_response = A + Vmax_sum * combined_hill
```

#### Option 4: Use Multiplicative Combination

Instead of addition:
```python
combined_hill = alpha * Hilla * (1 - beta * (1 - Hillb))
```

This ensures the result stays in [0, 1].

## Technical Correction Status

The inverse technical correction for priors (added in commit fa959a4) **appears correct**:

```python
# For binomial (trans.py lines 1079-1091)
logit_baseline = logit(p_observed) - alpha_y_add
p_baseline = sigmoid(logit_baseline)
```

This matches the forward correction in the sampler (distributions.py lines 349-353):
```python
logit_final = logit_mu + alpha_y_used
```

So the technical correction is not the primary issue - the core problem is the invalid probability generation from the additive Hill formulation.

## Files Modified

1. **bayesDREAM/fitting/trans.py**
   - Added `min_denominator` parameter (line 792)
   - Added documentation (lines 810-813)
   - Added filtering logic (lines 972-991)

2. **bayesDREAM/fitting/distributions.py**
   - Updated `sample_binomial_trans` to handle masking (lines 293-376)
   - Computes log-probability manually
   - Zeros out masked observations

3. **bayesDREAM/plotting/xy_plots.py**
   - Fixed `predict_trans_function` for binomial (lines 916-930)
   - Now matches actual model computation
   - Added warning comments about exceeding bounds

## Commits

Will be committed after user confirms debugging approach.
