# Multinomial Technical Correction Implementation

## Summary

Multinomial trans fitting now supports technical correction, matching the working binomial implementation. The technical correction is applied on the **logit scale** (additive), consistent with how fit_technical handles multinomial data.

---

## Key Changes (Commit 59f08d9)

### 1. **Enabled alpha_y_full Construction for Multinomial**

**Old Code** (line 659):
```python
if alpha_y is not None and groups_tensor is not None and distribution != 'multinomial':
    # ... only builds alpha_y_full for non-multinomial
```

**New Code**:
```python
if alpha_y is not None and groups_tensor is not None:
    # ... builds alpha_y_full for ALL distributions, including multinomial
```

**Impact:** Multinomial can now use technical correction like binomial.

---

### 2. **Added 3D and 4D Shape Handling**

Multinomial requires special handling because alpha_y has an extra category dimension:

**Training:** `alpha_y` is **[C, T, K]** (C groups, T features, K categories)
**Predictive:** `alpha_y` is **[S, C, T, K]** (S samples, C groups, T features, K categories)

**New Code** handles these shapes:
```python
if alpha_y.dim() == 4:  # Predictive multinomial: (S, C-1, T, K) or (S, C, T, K)
    if alpha_y.shape[1] == C:
        alpha_y_full = alpha_y  # Already includes reference
    else:
        # Add reference group with zeros
        baseline_shape = (alpha_y.shape[0], 1, alpha_y.shape[2], alpha_y.shape[3])
        baseline = torch.zeros(baseline_shape, device=self.model.device)
        alpha_y_full = torch.cat([baseline, alpha_y], dim=1)

elif alpha_y.dim() == 3:
    if distribution == 'multinomial' and K is not None and alpha_y.shape[-1] == K:
        # Training multinomial: (C-1, T, K) or (C, T, K)
        if alpha_y.shape[0] == C:
            alpha_y_full = alpha_y  # Already includes reference
        else:
            # Add reference group with zeros
            baseline_shape = (1, alpha_y.shape[1], alpha_y.shape[2])
            baseline = torch.zeros(baseline_shape, device=self.model.device)
            alpha_y_full = torch.cat([baseline, alpha_y], dim=0)
```

---

### 3. **Uses Zeros Baseline (Additive Correction)**

Like binomial, multinomial uses **additive correction on logit scale**:
```python
# Technical correction: logit(p_corrected) = logit(p_baseline) + alpha_y
# Therefore reference group (group 0) has alpha_y = 0 (not 1)
baseline = torch.zeros(baseline_shape, device=self.model.device)
```

This matches fit_technical implementation (technical.py:260):
```python
alpha_logits_y = pyro.sample("alpha_logits_y", dist.StudentT(df=3, loc=0.0, scale=20.0))
# Reference group (group 0) implicitly has alpha = 0
```

---

## How Multinomial Technical Correction Works

### In fit_technical (technical.py):
1. Sample `alpha_logits_y` for C-1 non-reference groups: **[C-1, T, K]**
2. Add reference group (zeros): **[C, T, K]**
3. Apply on logit scale per category:
   ```python
   logits = log(mu_y) + alpha_y_full[group, :, :]  # [T, K]
   probs = softmax(logits, dim=-1)  # Normalize across K categories
   ```

### In fit_trans (trans.py):
1. Load `alpha_y_prefit`: **[C, T, K]** from fit_technical
2. Build `alpha_y_full`: Check if reference included, add zeros if needed
3. Pass to multinomial sampler (distributions.py):
   ```python
   # Forward correction (in sampler)
   logits = log(mu_y) + alpha_y_full[groups_tensor, :, :]
   probs = masked_softmax(logits, dim=-1)  # Normalize per category
   ```

---

## Implementation Details

### Priors (Already Implemented)

Multinomial already uses **Dirichlet priors** for A and upper_limit (lines 177-207):
```python
if distribution == 'multinomial' and Amean_tensor.ndim > 1:
    # Amean_tensor: [T, K], Vmax_mean_tensor: [T, K]
    if use_data_driven_priors:
        # Data-driven Dirichlet: concentration ∝ mean probabilities
        A_mean_normalized = Amean_tensor / Amean_tensor.sum(dim=-1, keepdim=True)
        A = pyro.sample("A", dist.Dirichlet(A_mean_normalized * K))  # [T, K]
        
        upper_mean_normalized = Vmax_mean_tensor / Vmax_mean_tensor.sum(dim=-1, keepdim=True)
        upper_limit = pyro.sample("upper_limit", dist.Dirichlet(upper_mean_normalized * K))
    else:
        # Uniform priors: equal concentration for all categories
        concentration_A = self._t(1.0).expand([T, K])
        A = pyro.sample("A", dist.Dirichlet(concentration_A))
        
        concentration_upper = self._t(1.0).expand([T, K])
        upper_limit = pyro.sample("upper_limit", dist.Dirichlet(concentration_upper))
```

This matches binomial's Beta priors (which are special cases of Dirichlet for K=2).

### Hill Functions (K-1 Independent)

Multinomial fits **K-1 independent Hill functions** (lines 434-530):
```python
# A and Vmax_sum are [T, K] from Dirichlet
# Extract K-1 for fitting Hills (Kth doesn't get a Hill function)
A_kminus1 = A[..., :K_minus_1]  # [T, K-1]
Vmax_sum_kminus1 = Vmax_sum[..., :K_minus_1]  # [T, K-1]

# For each category k in K-1:
#   y_k = A_k + Vmax_sum_k * (alpha * Hill_a_k + beta * Hill_b_k)

# Kth category is residual: y_K = 1 - sum(y_1, ..., y_{K-1})
```

### Polynomial (K Independent in Logit Space)

Multinomial fits **K independent polynomials** (lines 572-602):
```python
# For multinomial: K independent polynomials in logit space
for d in range(1, polynomial_degree + 1):
    with pyro.plate(f"poly_category_plate_deg{d}", K, dim=-2):
        coeff = pyro.sample(f"poly_coeff_{d}", dist.Normal(0., sigma_coeff))  # [K, T]

# Compute logits for each category
logits_K = logit_A + alpha * poly_val  # [N, T, K]

# Apply softmax to get probabilities that sum to 1
y_dose_response = torch.softmax(logits_K, dim=-1)  # [N, T, K]
```

---

## Testing Checklist

- [x] alpha_y_full construction handles 3D and 4D shapes
- [x] Uses zeros baseline (additive correction)
- [x] Comments updated to reflect multinomial support
- [x] Priors already use Dirichlet (matching binomial's Beta)
- [x] Hill functions: K-1 independent (Kth residual)
- [x] Polynomial: K independent in logit space with softmax
- [ ] Test with real multinomial data (donor/acceptor usage)
- [ ] Verify technical correction is applied correctly
- [ ] Compare to binomial behavior (should be analogous)

---

## Next Steps

1. **Test with multinomial data** (e.g., donor/acceptor splice site usage)
2. **Verify fits improve** with technical correction enabled
3. **Compare to binomial** - behavior should be analogous
4. **Check diagnostics** - ensure priors vs posteriors look reasonable

---

## Related Files

- `bayesDREAM/fitting/trans.py` (lines 659-715, 779-790): alpha_y_full construction and usage
- `bayesDREAM/fitting/technical.py` (lines 215-290): Multinomial technical effects in fit_technical
- `bayesDREAM/fitting/distributions.py` (lines 206-290): Multinomial sampler with technical correction
- `BINOMIAL_CHANGES_SINCE_BCD1C4F.md`: Comparison document for binomial changes
