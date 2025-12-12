# Binomial fit_trans Changes Since Commit bcd1c4f

## Summary
There were **3 major changes** to binomial trans fitting since commit bcd1c4f that could explain the shift from "downward sloping fits" to "flat fits":

---

## 1. **CRITICAL BUG: alpha_y_full Construction** (Fixed today)

### Old Code (commit bcd1c4f):
```python
if alpha_y is not None and groups_tensor is not None and distribution != 'multinomial':
    if alpha_y.dim() == 2:  # Training: (C-1, T)
        ones_shape = (1, T)
        alpha_y_full = torch.cat([torch.ones(ones_shape, device=self.model.device), alpha_y], dim=0)
```

**Problem:** Always used `torch.ones()` for reference group baseline.

### Current Code (after today's fix):
```python
if alpha_y is not None and groups_tensor is not None and distribution != 'multinomial':
    # CRITICAL: Check if alpha_y already includes reference group
    if alpha_y.dim() == 2:  # Training: Could be (C-1, T) or (C, T)
        if alpha_y.shape[0] == C:
            alpha_y_full = alpha_y  # Already includes reference
        else:
            ones_shape = (1, T)
            if distribution == 'negbinom':
                baseline = torch.ones(ones_shape, device=self.model.device)  # Multiplicative
            else:
                baseline = torch.zeros(ones_shape, device=self.model.device)  # Additive
            alpha_y_full = torch.cat([baseline, alpha_y], dim=0)
```

**Fix:** 
- Checks if reference group already included
- Uses `torch.zeros()` for binomial (additive correction on logit scale)
- Uses `torch.ones()` for negbinom (multiplicative correction)

**Impact:** This was causing BOTH technical groups to get wrong corrections:
- Group 0 got α=1.0 (multiplicative) instead of α=0.0 → pushed fits DOWN
- Group 1 got α=0.0 instead of correct value → couldn't match data

---

## 2. **Hill Function Parameterization Change**

### Old Code (commit bcd1c4f):
```python
elif function_type == 'additive_hill':
    Hilla = Hill_based_positive(x_true.unsqueeze(-1), Vmax=Vmax_a, A=0, K=K_a, n=n_a, epsilon=epsilon_tensor)
    Hillb = Hill_based_positive(x_true.unsqueeze(-1), Vmax=Vmax_b, A=0, K=K_b, n=n_b, epsilon=epsilon_tensor)
    y_dose_response = A + (alpha * Hilla) + (beta * Hillb)

# For binomial, clamp output to [0, 1]
if distribution == 'binomial':
    y_dose_response = torch.clamp(y_dose_response, min=epsilon_tensor, max=1.0 - epsilon_tensor)
```

**Old formulation:** `y = A + alpha*Hill_a(Vmax_a) + beta*Hill_b(Vmax_b)`
- Learned separate `Vmax_a` and `Vmax_b` parameters
- Clamped output to [0, 1]

### Current Code:
```python
elif function_type == 'additive_hill':
    Hilla = Hill_based_positive(x_true.unsqueeze(-1), Vmax=self._t(1.0), A=0, K=K_a, n=n_a, epsilon=epsilon_tensor)
    Hillb = Hill_based_positive(x_true.unsqueeze(-1), Vmax=self._t(1.0), A=0, K=K_b, n=n_b, epsilon=epsilon_tensor)
    combined_hill = alpha * Hilla + beta * Hillb
    y_dose_response = A + Vmax_sum * combined_hill

# NO clamp on combined_hill or y_dose_response
```

**New formulation:** `y = A + Vmax_sum * (alpha*Hill_a(1.0) + beta*Hill_b(1.0))`
- Fixed Vmax=1.0 in Hill functions
- Learned single `Vmax_sum` parameter that scales the combined hill
- NO clamping (could exceed [0, 1]!)

**Impact:**
- Different parameter identifiability
- Old: `Vmax_a` and `Vmax_b` could vary independently
- New: Single `Vmax_sum` constrains total amplitude
- Removing clamp means model could predict p > 1 or p < 0 (invalid probabilities!)

---

## 3. **Priors Changed for binomial**

### Old Code (commit bcd1c4f):
```python
if distribution == 'binomial':
    # Beta prior for Vmax_a (in [0,1])
    Vmax_mean_clamped = Vmax_mean_tensor.clamp(min=0.01, max=0.99)
    concentration_vmax = self._t(10.0)
    alpha_vmax = Vmax_mean_clamped * concentration_vmax
    beta_vmax = (1 - Vmax_mean_clamped) * concentration_vmax
    Vmax_a = pyro.sample("Vmax_a", dist.Beta(alpha_vmax, beta_vmax))
```

### Current Code:
```python
if distribution in ['binomial']:
    # Use A and Vmax_sum (upper_limit) parameterization
    # No separate Vmax_a/Vmax_b
    # Vmax_sum represents total amplitude (A to upper_limit)
    
    upper_limit = pyro.sample("upper_limit", dist.Beta(alpha_upper, beta_upper).expand([T]))
    Vmax_sum = upper_limit - A
    
    # Then compute: y = A + Vmax_sum * hill
```

**Impact:**
- Changed from learning `Vmax_a`, `Vmax_b` directly
- Now learns `upper_limit`, derives `Vmax_sum = upper_limit - A`
- Different prior structure

---

## Which Change Caused the Problem?

**Most likely:** The **alpha_y_full bug (#1)** is the primary culprit for flat fits being too low.

**Potentially:** The **removal of clamping (#2)** could allow invalid probabilities and weird optimizer behavior.

**Less likely:** The **Vmax parameterization change (#2, #3)** changes identifiability but shouldn't fundamentally break fitting.

---

## Recommendations

1. ✅ **alpha_y_full bug is fixed** (today's fix)

2. ⚠️ **Consider adding back clamping for binomial**:
   ```python
   if distribution == 'binomial':
       y_dose_response = torch.clamp(y_dose_response, min=epsilon_tensor, max=1.0 - epsilon_tensor)
   ```

3. ⚠️ **Test if Vmax parameterization should be reverted** to old formulation for binomial

4. 📊 **Re-run fits with fixed alpha_y_full** to see if problem is resolved
