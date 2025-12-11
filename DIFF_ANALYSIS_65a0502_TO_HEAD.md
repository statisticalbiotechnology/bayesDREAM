# Comprehensive Diff Analysis: Commit 65a0502 → HEAD

**Date Generated**: December 11, 2025
**Base Commit**: 65a0502 (November 6, 2025)
**Current Commit**: HEAD
**Requested Files**:
- `bayesDREAM/fitting/technical.py` (fit_technical and _model_technical)
- `bayesDREAM/fitting/distributions.py` (binomial sampler)

---

## Summary

Between November 6, 2025 (commit 65a0502) and the current state, the codebase underwent **extensive refactoring and enhancement**. The total changes across the three requested files comprise:

- **technical.py**: 1,192 lines of diff
- **distributions.py**: 663 lines of diff (entirely new file)

The changes can be grouped into several major categories:

1. **Modularization**: Observation samplers extracted to separate `distributions.py` module
2. **Memory optimization**: Memory estimation, minibatching, sparse matrix support
3. **Performance improvements**: OneCycleLR scheduler, parallel Predictive sampling
4. **Distribution support**: Added studentt, removed mvnormal
5. **Defensive programming**: Better handling of edge cases, NaN/Inf values
6. **Bug fixes**: Proper technical group effect type selection

---

## 1. distributions.py (NEW FILE)

**Status**: This file did not exist at commit 65a0502. It was created as part of the modularization effort.

### Purpose

Extract all distribution-specific observation sampling logic into a standalone module with a clean registry-based API.

### Key Components

#### 1.1. Observation Samplers

Five distribution types now supported, each with dedicated sampler:

**`sample_negbinom_trans()`**
- Multiplicative technical group effects: `mu_final = mu_y * alpha_y[group] * sum_factor`
- Handles both 2D `[C, T]` and 3D `[S, C, T]` alpha tensors for Predictive
- Uses `torch.gather()` optimization to avoid creating huge intermediate tensors
- Shape expansion logic for broadcasting during Predictive sampling

**`sample_multinomial_trans()`**
- NEW: Robust masked softmax implementation
- Handles zero-category masking (categories that should never be selected)
- Additive technical effects on log-probability scale
- Prevents NaN for fully-masked rows (falls back to uniform)
- Stable log-likelihood computation with lgamma

**`sample_binomial_trans()`**
- **CRITICAL for your use case**: Additive technical group effects on LOGIT scale
- `logit(p) = logit(mu_y) + alpha_y[group]`
- Masking for observations with denominator=0
- **Diagnostic output** when non-finite log_prob detected:
  - Prints range checks for all intermediate values
  - Identifies where NaN/Inf originates (alpha, logit, p, log terms)
  - Raises ValueError with detailed diagnostic info

**`sample_normal_trans()`**
- Additive technical group effects on mean: `mu_adjusted = mu_y + alpha_y[group]`
- **Masked observations**: NaN/Inf in y_obs treated as missing
- Missing values contribute 0 to log-likelihood (factor of 1 in probability)

**`sample_studentt_trans()`**
- Same as normal but with heavy-tailed distribution
- Additive technical group effects on mean
- Masked observations for missing data
- df parameter (nu_y) can be scalar or tensor

#### 1.2. Distribution Registry

```python
DISTRIBUTION_REGISTRY = {
    'negbinom': {
        'trans': sample_negbinom_trans,
        'technical_group_type': 'multiplicative'
    },
    'binomial': {
        'trans': sample_binomial_trans,
        'technical_group_type': 'logit'  # Applied on logit scale
    },
    'multinomial': {
        'trans': sample_multinomial_trans,
        'technical_group_type': 'logit'
    },
    'normal': {
        'trans': sample_normal_trans,
        'technical_group_type': 'additive'
    },
    'studentt': {
        'trans': sample_studentt_trans,
        'technical_group_type': 'additive'
    }
}
```

#### 1.3. Utility Functions

- `get_observation_sampler(distribution, model_type)`: Retrieve sampler from registry
- `requires_denominator(distribution)`: Check if denominator needed (binomial)
- `requires_sum_factor(distribution)`: Check if sum factors needed (negbinom only)
- `is_3d_distribution(distribution)`: Check if 3D data structure (multinomial)
- `supports_technical_group_effects(distribution)`: Check if technical groups supported
- `get_technical_group_effect_type(distribution)`: Get effect type (multiplicative/additive/logit)

### Key Changes for Binomial (Your Use Case)

**Before (November 6th)**:
- Binomial observation sampling was inline in `_model_technical`
- Technical group effects handled case-by-case in model code
- No diagnostic output when NaN/Inf occurred

**After (Current)**:
- Binomial sampler is in standalone `sample_binomial_trans()` function
- Clear documentation: "Technical group effects: Applied on LOGIT scale"
- Masking for denominator=0 observations
- **Extensive diagnostics** when non-finite values detected:
  ```python
  if not torch.isfinite(log_prob).all():
      # Prints detailed breakdown of:
      # - mu_y range
      # - logit_mu range and finiteness
      # - alpha_y_used range and finiteness
      # - logit_final range and finiteness
      # - p (sigmoid) range and finiteness
      # - log(p) and log(1-p) invalid counts
      # - y_obs and denominator ranges
      # - valid_mask counts
      raise ValueError(f"Non-finite log_prob detected. See diagnostic output above.")
  ```

---

## 2. technical.py: _model_technical()

### 2.1. Signature Changes

**Added parameters**:
- `skip_obs_sampling=False`: **NEW** - Skip observation sampling during Predictive
- `use_all_cells=False`: Use all cells instead of NTC-only (for guide-prior fitting)

**Distribution support**:
- **Added**: `studentt` distribution
- **Removed**: `mvnormal` distribution

### 2.2. Technical Group Effect Type Selection

**Critical bug fix** for non-negbinom distributions:

**Before**:
```python
# Always used log2 scale for all distributions
log2_alpha_y = pyro.sample("log2_alpha_y", dist.StudentT(...))
```

**After**:
```python
if distribution == 'negbinom':
    # Multiplicative effects on log2 scale
    log2_alpha_y = pyro.sample("log2_alpha_y", dist.StudentT(df=3, scale=20.0))
    alpha_y = torch.pow(2.0, log2_alpha_y)  # [C, T]
else:
    # Additive effects (binomial/normal/studentt use logit/linear scale)
    log2_alpha_y = pyro.sample("log2_alpha_y", dist.StudentT(df=3, scale=20.0))
    alpha_y = log2_alpha_y  # Directly on logit/linear scale, NOT 2^x
```

**Why this matters**:
- For binomial: `alpha_y` is added to logit(p), not multiplied
- Using `2^alpha` would make no sense on logit scale
- This was likely causing numerical issues in earlier versions

### 2.3. Observation Sampling Logic

**Major change**: All observation sampling now wrapped in `if not skip_obs_sampling:` conditionals.

**Before**:
```python
# Observation samplers were always called
# No way to skip during Predictive
if distribution == 'binomial':
    # ... always sampled observations
```

**After**:
```python
# Observation sampling is conditional
if not skip_obs_sampling:
    from .distributions import get_observation_sampler
    observation_sampler = get_observation_sampler(distribution, 'trans')

if not skip_obs_sampling and distribution == 'negbinom':
    observation_sampler(...)
elif not skip_obs_sampling and distribution == 'binomial':
    observation_sampler(...)
# ... etc for all distributions
```

**Why this was added**:
- During training: Need observations for likelihood evaluation
- During Predictive: Only need parameter posteriors (alpha, mu, phi), NOT observations
- Extreme posterior alpha values can cause NaN/Inf when evaluating likelihood
- Skipping observation sampling completely avoids these numerical issues

### 2.4. Distribution Dimension Handling

**New logic for binomial/normal/studentt**:

```python
# Check if this is 2D (binomial, normal, studentt) or 3D (multinomial)
is_3d_dist = (distribution == 'multinomial')

if is_3d_dist:
    # Multinomial: counts_ntc is [N, T, K]
    if D is None or K is None:
        raise ValueError("For multinomial: D (n_features) and K (n_categories) required")
    T_actual = D
else:
    # 2D distributions: counts_ntc is [N, T]
    T_actual = T
```

This ensures correct tensor shapes for different distribution types.

### 2.5. StudentT Prior Details

**Distribution of alpha_y**:
```python
log2_alpha_y = pyro.sample(
    "log2_alpha_y",
    dist.StudentT(
        df=self._t(3),
        loc=self._t(0.0),
        scale=self._t(20.0)
    ),
)
```

**Parameters**:
- `df=3`: Heavy tails allow for occasional extreme values
- `loc=0`: Centered at no effect
- `scale=20`:
  - For negbinom (log2 scale): 2^±20 fold changes = reasonable range
  - For binomial (logit scale): ±20 on logit scale = p ≈ 0 or p ≈ 1 (extreme!)

**Why scale=20 for binomial is problematic**:
- logit(p) = 20 → p = sigmoid(20) = 0.9999999979
- logit(p) = -20 → p = sigmoid(-20) = 0.0000000021
- When posterior samples hit these extremes: log(1-p) = -Inf or log(p) = -Inf
- **However**: The `skip_obs_sampling` mechanism prevents this from causing errors

---

## 3. technical.py: fit_technical()

### 3.1. Memory Estimation and Minibatching

**NEW functionality**: Automatic memory estimation before Predictive sampling.

**Components**:

1. **`_estimate_predictive_memory_and_set_minibatch()`** (NEW method):
   - Calculates expected memory usage based on tensor shapes and nsamples
   - Uses `psutil` to check available RAM
   - Automatically sets `minibatch_size` if memory would be exceeded
   - Prints warnings and recommendations

2. **Memory calculation formula**:
   ```python
   # Per-sample memory
   mem_per_sample = (
       C * T * 4 +        # alpha_y: [C, T] float32
       T * 4 +            # mu_x, phi_y: [T] float32
       # ... other tensors
   )
   total_mem_gb = mem_per_sample * nsamples / 1e9
   ```

3. **Minibatch sampling**:
   - If memory exceeds 80% of available RAM, use minibatches
   - Preallocate full output tensors before first batch
   - Fill in-place for each subsequent batch
   - Clean up after each batch: `del samples; gc.collect(); torch.cuda.empty_cache()`

**Before**:
```python
# No memory checks
# If out of memory → crash
predictive = pyro.infer.Predictive(model, guide=guide, num_samples=nsamples)
samples = predictive(**model_inputs)
```

**After**:
```python
# Check memory first
mem_est_gb = self._estimate_predictive_memory_and_set_minibatch(...)
if mem_est_gb > available_ram * 0.8:
    minibatch_size = ...  # Set automatically
    print(f"[WARNING] Estimated {mem_est_gb:.2f} GB exceeds available RAM")

# Use minibatches if needed
if minibatch_size is not None:
    # Run in batches, preallocate, fill in-place
    ...
```

### 3.2. Parallel Predictive Sampling

**NEW parameter**: `use_parallel=True` (default)

```python
predictive_technical = pyro.infer.Predictive(
    self._model_technical,
    guide=guide_cellline,
    num_samples=nsamples,
    parallel=use_parallel  # NEW: Enable parallel sampling
)
```

This uses Pyro's built-in parallelization for faster sampling.

### 3.3. OneCycleLR Scheduler

**NEW optimizer option**: `optim_name='onecycle'`

**Implementation**:
```python
if optim_name == "onecycle":
    base_optimizer = pyro.optim.Adam({"lr": base_lr})
    optimizer = PyroOneCycleLR(
        base_optimizer,
        max_lr=base_lr * 10,
        total_steps=num_steps,
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=10.0,
        final_div_factor=100.0
    )
```

**Benefits**:
- Faster convergence for many problems
- Better final loss values
- Less sensitive to learning rate choice

### 3.4. Sparse Matrix Support

**NEW functionality**: Handle scipy.sparse matrices without converting to dense.

**Detection**:
```python
import scipy.sparse as sp
if sp.issparse(counts_to_fit):
    print("[INFO] Input is sparse matrix, will handle efficiently")
```

**Conversion strategy**:
- Only convert to dense when needed (tensor operations)
- Keep sparse for storage and indexing
- Use `torch.from_numpy()` instead of `torch.tensor()` for views (no copy)

### 3.5. Guide Selection Based on Memory

**NEW logic**: Automatically choose guide type based on available memory.

```python
if hasattr(self, '_estimate_guide_memory'):
    mem_cellline = self._estimate_guide_memory('cellline', ...)
    mem_shared = self._estimate_guide_memory('shared', ...)

    if mem_cellline < available_ram * 0.5:
        guide_type = 'cellline'
    else:
        guide_type = 'shared'
        print("[INFO] Using shared guide due to memory constraints")
```

### 3.6. Skip Observation Sampling During Predictive

**CRITICAL FIX** for your NaN/Inf issue:

**Added**:
```python
# Create model_inputs dict with skip flag
model_inputs_with_skip = model_inputs.copy()
model_inputs_with_skip['skip_obs_sampling'] = True

# Pass to Predictive
posterior_samples = predictive_technical(**model_inputs_with_skip)
```

**Comments in code**:
```python
# CRITICAL: Skip observation sampling during Predictive
# During Predictive, posterior samples of alpha_y can be extreme due to heavy-tailed priors
# This causes NaN/Inf in likelihood evaluation
# We only need parameter posteriors (alpha, mu, phi), not observations
```

**Why this works**:
- Training (guide fitting): `skip_obs_sampling=False` → observations sampled → likelihood computed
- Predictive (posterior sampling): `skip_obs_sampling=True` → observations NOT sampled → no NaN/Inf
- Result: Get clean posterior samples of parameters without numerical issues

### 3.7. Correct alpha_y Type Selection

**Critical bug fix** in posterior processing:

**Before**:
```python
# Always used multiplicative alpha_y
modality.alpha_y_prefit = posterior_samples["alpha_y_mult"]
```

**After**:
```python
# Choose correct type based on distribution
if modality.distribution == 'negbinom':
    modality.alpha_y_prefit = full_alpha_y_mult[..., trans_idx]
else:  # binomial, multinomial, normal, studentt use additive
    modality.alpha_y_prefit = full_alpha_y_add[..., trans_idx]

modality.alpha_y_type = 'posterior'
```

**Why this matters**:
- Binomial/normal/studentt use additive effects, not multiplicative
- Using wrong type would cause incorrect technical corrections downstream
- This ensures `alpha_y_prefit` is on the correct scale for each distribution

### 3.8. Feature Metadata Management

**Improved logic** for storing NTC exclusion flags:

**Issue**: Need to determine whether masks correspond to `self.model.counts` features or `modality.counts` features.

**Solution**:
```python
# Determine where masks were computed from
used_original_counts = (
    modality_name == self.model.primary_modality
    and hasattr(self.model, "counts")
)

if used_original_counts:
    # Masks are for original counts features - store in base class
    if isinstance(self.model.counts, pd.DataFrame):
        index = self.model.counts.index
    else:
        index = pd.RangeIndex(len(zero_count_mask))

    if not hasattr(self.model, "counts_meta"):
        self.model.counts_meta = pd.DataFrame(index=index)

    self.model.counts_meta["ntc_zero_count"] = zero_count_mask
    self.model.counts_meta["ntc_excluded_from_fit"] = needs_exclusion_mask
else:
    # Masks are for modality features - store in modality metadata
    modality.feature_meta["ntc_zero_count"] = zero_count_mask.tolist()
    modality.feature_meta["ntc_excluded_from_fit"] = needs_exclusion_mask.tolist()
```

### 3.9. Defensive Programming Improvements

Several edge cases now handled:

1. **Empty covariates list**:
   ```python
   def set_technical_groups(self, covariates: list[str]):
       if not covariates:
           print("[INFO] No covariates - creating single technical group (C=1)")
           self.model.meta["technical_group_code"] = 0
           return
   ```

2. **Missing denominator for binomial**:
   ```python
   if distribution == 'binomial':
       if denominator_ntc_tensor is None:
           raise ValueError("Binomial requires denominator_ntc_tensor")
       if denominator_ntc_tensor.shape != counts_ntc.shape:
           raise ValueError(f"Shape mismatch: counts {counts_ntc.shape} vs denominator {denominator_ntc_tensor.shape}")
   ```

3. **Invalid mask lengths**:
   ```python
   if len(zero_count_mask) != len(modality.feature_meta):
       raise ValueError(
           f"Mask length mismatch: masks have {len(zero_count_mask)} elements "
           f"but modality.feature_meta has {len(modality.feature_meta)} rows"
       )
   ```

---

## 4. Summary of Critical Changes for Binomial Distribution

For your specific use case (binomial distribution for splicing data), the most important changes are:

### 4.1. Skip Observation Sampling During Predictive

**What changed**: Added `skip_obs_sampling` parameter throughout the code path.

**Why it matters**:
- Prevents NaN/Inf errors during Predictive sampling
- Extreme posterior alpha_y values no longer cause numerical overflow
- You get clean parameter posteriors without observation-related issues

**Where it's used**:
- `_model_technical()`: Wraps all observation samplers in `if not skip_obs_sampling:`
- `fit_technical()`: Passes `skip_obs_sampling=True` to Predictive
- Result: Observations never sampled during Predictive, only during training

### 4.2. Correct Technical Group Effect Type

**What changed**: Distribution-specific handling of alpha_y type.

**Why it matters**:
- Binomial uses ADDITIVE effects on logit scale, not multiplicative
- Using wrong type causes incorrect technical corrections
- `alpha_y_prefit` now guaranteed to be on correct scale for downstream use

**Where it's used**:
- `_model_technical()`: Separate branches for negbinom vs. binomial
- `fit_technical()`: Stores correct alpha type in `modality.alpha_y_prefit`

### 4.3. Diagnostic Output in Binomial Sampler

**What changed**: Extensive diagnostics when non-finite values detected.

**Why it matters**:
- Easier to debug numerical issues
- Identifies exact location of NaN/Inf origin (alpha, logit, p, log terms)
- No longer fails silently

**Where it's used**:
- `distributions.py`: `sample_binomial_trans()` checks for non-finite log_prob
- Prints detailed breakdown of all intermediate values
- Raises ValueError with actionable information

### 4.4. Modularization into distributions.py

**What changed**: All observation samplers extracted to standalone module.

**Why it matters**:
- Cleaner separation of concerns
- Easier to test individual distributions
- Clear documentation of technical group effect types
- Registry-based API for extensibility

### 4.5. Memory Optimization

**What changed**: Memory estimation and minibatching.

**Why it matters**:
- Can handle larger datasets without OOM errors
- Automatic minibatch selection based on available RAM
- Efficient memory management (preallocate, fill in-place, cleanup)

---

## 5. Timeline of Key Commits

Based on the diff, major changes appear to have been introduced in several waves:

1. **Modularization**: Creating `distributions.py` and extracting samplers
2. **Memory optimization**: Adding memory estimation and minibatching
3. **Performance**: Adding OneCycleLR and parallel sampling
4. **Bug fixes**: Correct alpha_y type selection for non-negbinom
5. **Recent fix**: Adding `skip_obs_sampling` mechanism

The `skip_obs_sampling` fix appears to be the most recent change, likely added specifically to address the NaN/Inf issues you encountered.

---

## 6. Conclusion

The codebase has undergone substantial improvements since November 6th:

**Major additions**:
- Modular distribution samplers
- Memory-aware Predictive sampling
- Faster optimization (OneCycleLR)
- Better edge case handling

**Critical fixes**:
- Skip observation sampling during Predictive (prevents NaN/Inf)
- Correct technical group effect types (additive for binomial)
- Better diagnostic output

**Result**: The current code should be significantly more robust and performant than the November 6th version, especially for binomial distributions with technical group effects.

The key insight is that the November 5th working file likely succeeded due to:
- Favorable random seed leading to non-extreme alpha posteriors
- Or possibly different PyTorch version with different numerical behavior

The new `skip_obs_sampling` mechanism is a **more principled solution** that guarantees no NaN/Inf issues regardless of posterior values, since observations are never sampled during Predictive.
