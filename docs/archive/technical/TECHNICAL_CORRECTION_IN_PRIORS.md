# Technical Correction in Trans Prior Computation

**Date**: November 27, 2025
**Status**: Fixed in commit fa959a4

## The Problem

Prior parameters (A and Vmax) for the trans model were being computed from **observed data that included technical batch effects**, leading to **biased priors**. This created a mismatch:

- **Likelihood model**: Applies technical corrections (via α_y parameters)
- **Prior computation**: Did NOT account for technical effects

### Example

Consider two cell lines with different baseline expression:
- **Cell line A**: `α_y_add = +0.5` (logit shift for binomial)
- **Cell line B**: `α_y_add = -0.5` (logit shift for binomial)

**Before the fix**:
- Observed proportions reflect these technical biases
- A (5th percentile) and Vmax (95th percentile) computed from **biased** data
- Priors don't represent true baseline biology
- Model tries to fit baseline + technical effects, but priors already include effects

**After the fix**:
- Apply inverse correction: `logit(p_baseline) = logit(p_observed) - α_y_add`
- A and Vmax computed from **corrected** (baseline) data
- Priors represent true biological variation
- Model correctly separates baseline biology from technical effects

## The Solution

Apply **inverse technical correction** to data BEFORE computing guide means for priors:

### Distribution-Specific Corrections

| Distribution | Technical Effect (Forward) | Inverse Correction |
|--------------|----------------------------|-------------------|
| **negbinom** | `mu_corrected = mu × α_y_mult` | `mu_baseline = mu_obs / α_y_mult` |
| **normal** | `mu_corrected = mu + α_y_add` | `mu_baseline = mu_obs - α_y_add` |
| **studentt** | `mu_corrected = mu + α_y_add` | `mu_baseline = mu_obs - α_y_add` |
| **binomial** | `logit(p_corrected) = logit(p) + α_y_add` | `logit(p_baseline) = logit(p_obs) - α_y_add`<br>`p_baseline = sigmoid(logit_baseline)` |
| **multinomial** | `log(probs_corrected) = log(probs) + α_y_add` | `log(probs_baseline) = log(probs_obs) - α_y_add`<br>`probs_baseline = softmax(log_probs_baseline)` |

### Implementation Details

**Location**: `bayesDREAM/fitting/trans.py`, lines 1054-1129

**Logic flow**:
1. Normalize data (e.g., divide by sum factors, denominators)
2. **NEW**: Apply inverse technical correction if `alpha_y_prefit` is available
3. Compute guide means from corrected data
4. Calculate A (5th percentile) and Vmax (95th percentile) from guide means
5. Use corrected priors to initialize Hill function parameters

**Edge case handling**:
- NaN values from invalid corrections are excluded from prior computation
- Warning issued if corrections produce >0.1% non-finite values
- Uses `nanmean` and `nanvar` to ignore invalid entries

## Impact

### When Technical Effects Are Large

If technical effects cause 2-fold differences between batches:

**Before fix**:
- A and Vmax span 4-fold range (includes 2× technical + 2× biological)
- Priors too wide, poor constraint on Hill function
- Slower convergence, less reliable posteriors

**After fix**:
- A and Vmax span 2-fold range (biological only)
- Tighter priors, better constraint on Hill function
- Faster convergence, more reliable posteriors

### When Technical Effects Are Small

- Minimal impact if technical effects <10%
- No harm: inverse correction is nearly identity transform
- Ensures consistency across all datasets

## Validation

### Expected behavior after fix

1. **Console output**: Should see informative messages during `fit_trans()`:
   ```
   [INFO] Correcting for technical effects before computing priors (distribution: binomial)
   [INFO] binomial: Applied inverse logit correction (subtract alpha_y_add on logit scale)
   ```

2. **Prior values**: A and Vmax should be more similar across cell lines
   - Before: May differ by 2-5× between cell lines
   - After: Should differ by <1.5× (biological variation only)

3. **Convergence**: Check loss curves (`model.losses_trans`)
   - Should converge faster with corrected priors
   - Final ELBO may be higher (better fit)

4. **Posterior credible intervals**: Check width of posteriors for Hill parameters
   - Should be narrower with better priors
   - Less uncertainty in effect size estimates

### Testing recommendations

1. **Compare old vs. new fits** on same dataset:
   ```python
   # Plot prior distributions for A and Vmax
   import matplotlib.pyplot as plt

   # From fit_trans output
   print(f"A range: {Amean_tensor.min():.3f} - {Amean_tensor.max():.3f}")
   print(f"Vmax range: {Vmax_mean_tensor.min():.3f} - {Vmax_mean_tensor.max():.3f}")
   ```

2. **Check for warnings** about non-finite values:
   - If >1% of values become non-finite, technical effects may be too extreme
   - Consider refitting technical model with more regularization

3. **Posterior checks**:
   ```python
   # Compare prior vs posterior for A
   model.plot_trans_fit('A', function_type='additive_hill')
   ```

## Related Documentation

- **Hill function priors**: `docs/HILL_FUNCTION_PRIORS.md`
- **Technical fitting**: `docs/INITIALIZATION.md`
- **Distribution-specific details**: `bayesDREAM/fitting/distributions.py`

## Questions?

This fix ensures that priors represent baseline biology, not confounded by technical batch effects. If you have questions about:
- When this matters most
- How to diagnose biased priors in old fits
- Testing the fix on your data

Please refer to the commit message (fa959a4) or the implementation in `trans.py` lines 1054-1129.
