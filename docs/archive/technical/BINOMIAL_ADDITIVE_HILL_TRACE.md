# Trace: fit_trans with Binomial Distribution and Additive Hill Function

This document traces the complete execution flow when calling:
```python
model.fit_trans(
    sum_factor_col='sum_factor',
    function_type='additive_hill',
    modality_name='splicing_sj'  # binomial distribution
)
```

## Part 1: fit_trans() Entry Point (lines 617-1257)

### Step 1: Setup and Validation (lines 679-738)

```python
# Get modality
modality_name = 'splicing_sj'  # User specified
modality = self.model.get_modality('splicing_sj')

# Auto-detect distribution
distribution = modality.distribution  # 'binomial'

# Auto-detect denominator
denominator = modality.denominator  # 2D array [features × cells] with total counts

# Set default niters
niters = 100_000  # Default for non-multinomial, non-polynomial

# Check technical fit
if modality.alpha_y_prefit is None and 'technical_group_code' in self.model.meta.columns:
    raise ValueError(...)  # Would fail if technical fit not done

# Get counts
counts_to_fit = modality.counts  # Shape: [features, cells]

# Get technical results from MODALITY (not model!)
alpha_y_prefit = modality.alpha_y_prefit  # From fit_technical()
alpha_y_type = 'posterior'  # Always posterior from fit_technical
```

**Key Point**: For binomial, we have:
- `counts`: Numerator (e.g., junction read counts)
- `denominator`: Total counts (e.g., gene expression)
- Distribution will model: `y_obs ~ Binomial(denominator, probability)`

### Step 2: Check Requirements (lines 732-738)

```python
# Binomial requires denominator
if requires_denominator('binomial') and denominator is None:
    raise ValueError(...)

# Binomial does NOT require sum_factor (unlike negbinom)
# sum_factor_col can be None
```

### Step 3: Cell Subsetting (lines 752-769)

```python
# Subset meta to modality cells
modality_cells = modality.cell_names  # From modality
modality_cell_set = set(modality_cells)
meta_subset = self.model.meta[self.model.meta['cell'].isin(modality_cell_set)].copy()

# Check technical groups
if "technical_group_code" in meta_subset.columns:
    C = meta_subset['technical_group_code'].nunique()  # e.g., C=3 (3 cell lines)
    groups_tensor = torch.tensor(meta_subset['technical_group_code'].values, dtype=torch.long)
    # groups_tensor shape: [N] - one group code per cell
else:
    C = None
    groups_tensor = None
```

### Step 4: Prepare Observation Data (lines 767-806)

```python
# Get cell indices
cell_indices = [i for i, c in enumerate(modality_cells) if c in modality_cell_set]
N = len(cell_indices)  # Number of cells

# Subset counts (2D for binomial)
if modality.cells_axis == 1:
    y_obs = counts_to_fit[:, cell_indices].T  # [T, N] -> [N, T]
else:
    y_obs = counts_to_fit[cell_indices, :]  # Already [N, T]
T = y_obs.shape[1]  # Number of features (splice junctions)

# y_obs shape: [N, T] - observed junction counts (numerator)

# Handle sum factors (usually None for binomial, but can be provided)
if sum_factor_col is not None:
    sum_factor_tensor = torch.tensor(meta_subset[sum_factor_col].values, dtype=torch.float32)
else:
    sum_factor_tensor = torch.ones(N, dtype=torch.float32)

# Handle denominator (CRITICAL for binomial!)
if modality.cells_axis == 1:
    denominator_subset = denominator[:, cell_indices].T  # [T, N] -> [N, T]
else:
    denominator_subset = denominator[cell_indices, :]  # [N, T]
denominator_tensor = torch.tensor(denominator_subset, dtype=torch.float32)

# denominator_tensor shape: [N, T] - total counts for each junction
```

### Step 5: Prepare x_true and Hyperparameters (lines 818-876)

```python
# Get x_true (cis gene expression)
if self.model.x_true_type == 'point':
    x_true_mean = self.model.x_true  # [N]
elif self.model.x_true_type == 'posterior':
    x_true_mean = self.model.x_true.mean(dim=0)  # [S, N] -> [N]

# Convert hyperparameters to tensors
beta_o_alpha_tensor = torch.tensor(9.0)  # Overdispersion prior shape
beta_o_beta_tensor = torch.tensor(3.0)   # Overdispersion prior rate
K_alpha_tensor = torch.tensor(2.0)       # K (EC50) prior variance
Vmax_alpha_tensor = torch.tensor(2.0)    # Vmax prior variance
n_mu_tensor = torch.tensor(0.0)          # n (Hill coefficient) prior mean
epsilon_tensor = torch.tensor(1e-6)      # Numerical stability
p_n_tensor = torch.tensor(1e-6)          # Sparsity prior

# Convert observations
y_obs_tensor = torch.tensor(y_obs, dtype=torch.float32)  # [N, T]
```

### Step 6: Compute Safe Hill Coefficient Bounds (lines 833-852)

```python
# Compute nmin/nmax to avoid overflow in x**n
x_for_bounds = self.model.x_true  # [N] or [S, N]
x_min = torch.clamp(x_for_bounds.min(), min=1e-12)
x_max = x_for_bounds.max()

log_fmax = torch.log(torch.tensor(torch.finfo(torch.float32).max))  # ~88

# Compute boundaries
nmin_cand = (-log_fmax / torch.abs(torch.log(x_min))) if x_min < 1 else -inf
nmax_cand = (log_fmax / torch.abs(torch.log(x_max))) if x_max > 1 else inf

# Clamp to reasonable box
BOX_LOW = -20.0
BOX_HIGH = 20.0
nmin = max(nmin_cand, BOX_LOW)  # e.g., -20.0
nmax = min(nmax_cand, BOX_HIGH)  # e.g., 20.0
```

### Step 7: Compute Data-Driven Priors (lines 855-876)

```python
# Compute K_max from x_true range
guides_tensor = torch.tensor(self.model.meta['guide_code'].values)
K_max_tensor = torch.max(
    torch.stack([torch.mean(x_true_mean[guides_tensor == g])
                 for g in torch.unique(guides_tensor)])
)  # e.g., max mean x_true across guides

# Compute Vmax and A from observed data
# For binomial: work with probabilities (y_obs / denominator)
y_obs_factored = y_obs_tensor / denominator_tensor  # [N, T] probabilities

# Vmax: max probability across guides
Vmax_mean_tensor = torch.max(
    torch.stack([torch.mean(y_obs_factored[guides_tensor == g, :], dim=0)
                 for g in torch.unique(guides_tensor)]),
    dim=0
)[0]  # [T]

# A (baseline): min probability across guides
Amean_tensor = torch.min(
    torch.stack([torch.mean(y_obs_factored[guides_tensor == g, :], dim=0)
                 for g in torch.unique(guides_tensor)]),
    dim=0
)[0]  # [T]

# Clamp to valid range [epsilon, 1.0]
Amean_tensor = Amean_tensor.clamp(min=epsilon_tensor, max=1.0-epsilon_tensor)
Vmax_mean_tensor = Vmax_mean_tensor.clamp(min=epsilon_tensor, max=1.0-epsilon_tensor)
```

**Example values**:
- Suppose junction 1 has observed PSI range 0.1 to 0.8 across guides
- `Amean_tensor[0] = 0.1` (baseline probability)
- `Vmax_mean_tensor[0] = 0.8` (max change amplitude)

### Step 8: Initialize Guide and Optimizer (lines 892-969)

```python
# For additive_hill, use AutoNormalMessenger guide
guide_y = pyro.infer.autoguide.AutoNormalMessenger(self._model_y)
optimizer = pyro.optim.Adam({"lr": 1e-3})
svi = pyro.infer.SVI(self._model_y, guide_y, optimizer, loss=pyro.infer.Trace_ELBO())
```

### Step 9: Training Loop (lines 975-1064)

```python
for step in range(100_000):
    # Compute temperature (for relaxed Bernoulli sparsity)
    fraction_done = step / 100_000
    current_temp = 1.0 + (0.1 - 1.0) * fraction_done  # 1.0 -> 0.1

    # Sample from posterior (if x_true or alpha_y are posterior)
    if alpha_y_type == "posterior":
        samp = torch.randint(high=alpha_y_prefit.shape[0], size=(1,)).item()
        x_true_sample = self.model.x_true[samp]  # [N]
        alpha_y_sample = alpha_y_prefit[samp]     # [C-1, T]
    else:
        x_true_sample = self.model.x_true
        alpha_y_sample = alpha_y_prefit

    # SVI step
    loss = svi.step(
        N=N,
        T=T,
        y_obs_tensor=y_obs_tensor,
        sum_factor_tensor=sum_factor_tensor,
        beta_o_alpha_tensor=beta_o_alpha_tensor,
        beta_o_beta_tensor=beta_o_beta_tensor,
        K_max_tensor=K_max_tensor,
        Vmax_mean_tensor=Vmax_mean_tensor,
        Amean_tensor=Amean_tensor,
        x_true_sample=x_true_sample,
        nmin=nmin,
        nmax=nmax,
        alpha_y_sample=alpha_y_sample,
        C=C,
        groups_tensor=groups_tensor,
        temperature=torch.tensor(current_temp),
        function_type='additive_hill',
        distribution='binomial',
        denominator_tensor=denominator_tensor,
        ...
    )
```

### Step 10: Sample Posterior (lines 1066-1187)

```python
# Generate posterior samples
predictive_y = pyro.infer.Predictive(
    self._model_y,
    guide=guide_y,
    num_samples=1000
)

posterior_samples_y = predictive_y(
    N=N, T=T,
    y_obs_tensor=y_obs_tensor,
    denominator_tensor=denominator_tensor,
    x_true_sample=x_true_mean,
    alpha_y_sample=alpha_y_prefit.mean(dim=0) if alpha_y_type == "posterior" else alpha_y_prefit,
    temperature=torch.tensor(0.1),  # Final temp
    use_straight_through=True,
    function_type='additive_hill',
    distribution='binomial',
    ...
)
```

### Step 11: Store Results (lines 1234-1254)

```python
# Store in modality
modality.posterior_samples_trans = posterior_samples_y

# If primary modality, also store at model level
if modality_name == self.model.primary_modality:
    self.model.posterior_samples_trans = posterior_samples_y
```

---

## Part 2: _model_y() Probabilistic Model (lines 47-612)

Now let's trace what happens inside `_model_y()` during one SVI step with binomial + additive_hill:

### Step 1: Setup (lines 82-125)

```python
# N = number of cells
# T = number of features (junctions)
# distribution = 'binomial'
# function_type = 'additive_hill'
# C = number of technical groups (e.g., 3)

# x_true is observed (sampled from posterior)
x_true = x_true_sample  # [N]

# alpha_y is observed (sampled from technical fit posterior)
alpha_y = alpha_y_sample  # [C-1, T]
```

### Step 2: Sample Overdispersion Parameters (lines 108-113)

```python
# Sample global overdispersion parameter
beta_o ~ Gamma(alpha=9, beta=3)  # scalar

# Sample per-feature overdispersion
with trans_plate:  # T features
    o_y ~ Exponential(beta_o)  # [T]
    phi_y = 1 / (o_y**2)        # [T] precision

phi_y_used = phi_y.unsqueeze(-2)  # [1, T]
```

**Note**: For binomial, `phi_y` is used in Beta concentration parameters (see below).

### Step 3: Sample Baseline Parameter A (lines 143-175)

```python
with trans_plate:  # T features
    # Compute weight for baseline adjustment
    weight = o_y / (o_y + (beta_o_beta / beta_o_alpha).clamp_min(epsilon))  # [T]

    # For binomial: use Vmax_mean_tensor and Amean_tensor directly (no category reduction)
    Amean_for_A = Amean_tensor  # [T]
    Vmax_for_A = Vmax_mean_tensor  # [T]

    # Adjust baseline using weight
    Amean_adjusted = ((1 - weight) * Amean_for_A) + (weight * Vmax_for_A) + epsilon
    # Amean_adjusted: [T]

    # For binomial: A ~ Beta (constrained to [0,1])
    Amean_clamped = Amean_adjusted.clamp(min=0.01, max=0.99)  # [T]
    concentration = 10.0
    alpha_beta = Amean_clamped * concentration      # [T]
    beta_beta = (1 - Amean_clamped) * concentration # [T]

    A ~ Beta(alpha_beta, beta_beta)  # [T] in [0, 1]
```

**Example**:
- If `Amean_clamped[0] = 0.2` (20% baseline probability)
- `alpha_beta[0] = 0.2 * 10 = 2.0`
- `beta_beta[0] = 0.8 * 10 = 8.0`
- `A[0]` will be sampled from `Beta(2, 8)` with mean ≈ 0.2

### Step 4: Sample Sparsity Indicator (lines 177-181)

```python
with trans_plate:
    # Relaxed Bernoulli for sparsity (continuous during training)
    alpha ~ RelaxedBernoulli(temperature=current_temp, probs=1e-6)  # [T]
    # alpha ≈ 0 means junction is not dose-responsive
    # alpha ≈ 1 means junction is dose-responsive
```

### Step 5: Sample Hill Function Parameters - First Function (lines 183-263)

```python
# Global hyperparameters for Hill coefficient variance
sigma_n_a ~ Exponential(1/5)  # scalar
sigma_n_b ~ Exponential(1/5)  # scalar (for additive_hill)

with trans_plate:  # T features
    # Compute K and Vmax prior standard deviations
    K_sigma = (K_max_tensor / (2 * sqrt(K_alpha_tensor))) + epsilon  # scalar

    # For binomial: Vmax is probability, so use Beta prior
    Vmax_mean_clamped = Vmax_mean_tensor.clamp(min=0.01, max=0.99)  # [T]
    concentration_vmax = 10.0
    alpha_vmax = Vmax_mean_clamped * concentration_vmax      # [T]
    beta_vmax = (1 - Vmax_mean_clamped) * concentration_vmax # [T]

    # Sample first Hill function parameters
    Vmax_a ~ Beta(alpha_vmax, beta_vmax)  # [T] in [0, 1]
    K_a ~ Gamma(((K_max/2)^2) / (K_sigma^2), (K_max/2) / (K_sigma^2))  # [T]

    # Sample Hill coefficient (raw, before sparsity gating)
    n_a_raw ~ Normal(n_mu_tensor, sigma_n_a)  # [T]

    # Apply sparsity gating and clamp to safe bounds
    n_a = (alpha * n_a_raw).clamp(min=nmin, max=nmax)  # [T]
    # When alpha ≈ 0, n_a ≈ 0 (no Hill function effect)
```

**Example values** for junction 0:
- `Vmax_mean_clamped[0] = 0.8` (max observed PSI)
- `alpha_vmax[0] = 8.0`, `beta_vmax[0] = 2.0`
- `Vmax_a[0]` sampled from `Beta(8, 2)` with mean ≈ 0.8
- `K_a[0]` sampled from `Gamma(...)` with mean ≈ `K_max/2`
- `n_a_raw[0]` sampled from `Normal(0, sigma_n_a)`
- `n_a[0] = alpha[0] * n_a_raw[0]` (gated by sparsity)

### Step 6: Sample Hill Function Parameters - Second Function (lines 266-311)

```python
# For additive_hill, sample second set of parameters
with trans_plate:
    beta ~ RelaxedBernoulli(temperature=current_temp, probs=1e-6)  # [T]
    # Second sparsity indicator

    # Second Hill function parameters (same priors as first)
    Vmax_b ~ Beta(alpha_vmax, beta_vmax)  # [T] in [0, 1]
    K_b ~ Gamma(...)  # [T]

    n_b_raw ~ Normal(n_mu_tensor, sigma_n_b)  # [T]
    n_b = (beta * n_b_raw).clamp(min=nmin, max=nmax)  # [T]
```

### Step 7: Compute Dose-Response Function (lines 378-394)

```python
# For binomial with additive_hill (non-multinomial branch)
with trans_plate:
    # Compute first Hill function (positive activation)
    Hilla = Hill_based_positive(
        x_true.unsqueeze(-1),  # [N, 1]
        Vmax=Vmax_a,           # [T]
        A=0,                   # No offset in Hill function itself
        K=K_a,                 # [T]
        n=n_a,                 # [T]
        epsilon=epsilon_tensor
    )  # Returns [N, T]

    # Hilla[i, j] = Vmax_a[j] * (x_true[i]^n_a[j]) / (K_a[j]^n_a[j] + x_true[i]^n_a[j])

    # Compute second Hill function (can be inhibitory if n_b < 0)
    Hillb = Hill_based_positive(
        x_true.unsqueeze(-1),  # [N, 1]
        Vmax=Vmax_b,           # [T]
        A=0,
        K=K_b,                 # [T]
        n=n_b,                 # [T]
        epsilon=epsilon_tensor
    )  # Returns [N, T]

    # Combine both Hill functions additively
    y_dose_response = A + (alpha * Hilla) + (beta * Hillb)  # [N, T]
    # A: [T] baseline probability
    # alpha * Hilla: [T] * [N, T] = [N, T] first dose-response
    # beta * Hillb: [T] * [N, T] = [N, T] second dose-response

    # Clamp to valid probability range [epsilon, 1-epsilon]
    y_dose_response = torch.clamp(y_dose_response, min=epsilon, max=1.0 - epsilon)
```

**Example computation** for cell i, junction 0:
- `A[0] = 0.2` (baseline PSI)
- `alpha[0] = 0.95` (junction is dose-responsive)
- `Hilla[i, 0] = 0.6 * x_true[i]^2 / (K_a[0]^2 + x_true[i]^2)`
  - If `x_true[i] = 10`, `K_a[0] = 5`, `n_a[0] = 2`, `Vmax_a[0] = 0.6`:
  - `Hilla[i, 0] = 0.6 * 100 / (25 + 100) = 0.48`
- `beta[0] = 0.1` (second function mostly inactive)
- `Hillb[i, 0] ≈ 0.1` (small contribution)
- `y_dose_response[i, 0] = 0.2 + 0.95*0.48 + 0.1*0.1 = 0.2 + 0.456 + 0.01 = 0.666`

So the model predicts PSI ≈ 66.6% for this cell-junction combination.

### Step 8: Prepare for Observation Sampling (lines 499-510)

```python
# For binomial: mu_y is the probability (no further transformation)
mu_y = y_dose_response  # [N, T] in [0, 1]

# Prepare alpha_y_full (add reference group)
if alpha_y is not None and C is not None:
    # alpha_y has shape [C-1, T] (excluding reference group)
    # Add reference group (all ones)
    ones_shape = (1, T)
    alpha_y_full = torch.cat([torch.ones(ones_shape), alpha_y], dim=0)  # [C, T]
else:
    alpha_y_full = None
```

### Step 9: Call Binomial Observation Sampler (lines 573-583)

```python
# Get binomial sampler
from .distributions import get_observation_sampler
observation_sampler = get_observation_sampler('binomial', 'trans')

# Call sampler
observation_sampler(
    y_obs_tensor=y_obs_tensor,        # [N, T] observed counts (numerator)
    denominator_tensor=denominator_tensor,  # [N, T] total counts
    mu_y=mu_y,                        # [N, T] probabilities from dose-response
    alpha_y_full=alpha_y_full,        # [C, T] or None
    groups_tensor=groups_tensor,      # [N] group assignments
    N=N,
    T=T,
    C=C
)
```

---

## Part 3: Binomial Observation Sampler (distributions.py)

The actual binomial sampler from `bayesDREAM/fitting/distributions.py` (lines 217-274):

```python
def sample_binomial_trans(
    y_obs_tensor,
    denominator_tensor,
    mu_y,
    alpha_y_full,
    groups_tensor,
    N, T, C=None
):
    """
    Sample binomial observations with optional technical group correction.

    Technical group effects: Applied on LOGIT scale to maintain p ∈ [0, 1].

    Parameters
    ----------
    y_obs_tensor : [N, T] observed counts (numerator)
    denominator_tensor : [N, T] total counts
    mu_y : [N, T] baseline probabilities from dose-response
    alpha_y_full : [C, T] or None - technical group effects (additive on logit scale)
    groups_tensor : [N] group assignments

    Notes
    -----
    logit(p) = logit(mu_y) + alpha_y[group]
    p = sigmoid(logit(p))
    """
    # Convert mu_y to logit scale
    mu_y_clamped = torch.clamp(mu_y, min=1e-6, max=1-1e-6)  # [N, T]
    logit_mu = torch.log(mu_y_clamped) - torch.log(1 - mu_y_clamped)  # [N, T]

    # Apply technical group effects on logit scale (additive, not multiplicative!)
    if alpha_y_full is not None and groups_tensor is not None:
        alpha_y_used = alpha_y_full[groups_tensor, :]  # [N, T]
        logit_final = logit_mu + alpha_y_used  # [N, T] - additive on logit scale
    else:
        logit_final = logit_mu  # [N, T]

    # Sample from Binomial using logits (more numerically stable than probs)
    with pyro.plate("obs_plate", N, dim=-2):
        pyro.sample(
            "y_obs",
            dist.Binomial(total_count=denominator_tensor, logits=logit_final),
            obs=y_obs_tensor
        )
```

**Example with technical correction**:
- Cell i is in group 0 (e.g., cell line A)
- Junction 0 has dose-response probability `mu_y[i, 0] = 0.666`
- Technical correction: `alpha_y_full[0, 0] = 0.2` (additive logit shift for cell line A)
- `logit_mu[i, 0] = log(0.666) - log(0.334) = 0.688`
- `logit_final[i, 0] = 0.688 + 0.2 = 0.888`  (additive on logit scale)
- `p_final[i, 0] = sigmoid(0.888) = 0.708`
- Sample: `y_obs[i, 0] ~ Binomial(total_count=denominator[i, 0], logits=0.888)`
  - If `denominator[i, 0] = 100`, then `y_obs[i, 0]` is sampled count between 0-100 with mean ≈ 70.8

**Key insight**: Technical correction is **additive on logit scale**, not multiplicative on probability scale. This preserves the [0,1] constraint and allows both positive and negative corrections.

---

## Summary of Complete Flow

### Data Flow:
1. **Input**: Splice junction counts (numerator) and gene expression (denominator)
2. **Cis gene**: `x_true` provides perturbation strength per cell
3. **Hill functions**: Two additive Hill functions map `x_true` → probability
4. **Technical groups**: Logit-scale correction for batch effects
5. **Binomial likelihood**: Sample counts given total and probability

### Parameter Flow:
```
Hyperpriors:
  β_o ~ Gamma(9, 3)
  σ_n_a ~ Exponential(1/5)
  σ_n_b ~ Exponential(1/5)

Per-feature parameters (T junctions):
  o_y ~ Exponential(β_o)                    # Overdispersion
  A ~ Beta(adjusted mean)                   # Baseline probability [0,1]
  α ~ RelaxedBernoulli(1e-6)               # Sparsity indicator
  β ~ RelaxedBernoulli(1e-6)               # Second sparsity

  # First Hill function
  Vmax_a ~ Beta(prior mean)                # Amplitude [0,1]
  K_a ~ Gamma(prior mean)                  # Half-max point
  n_a ~ α * Normal(0, σ_n_a)               # Hill coefficient (gated)

  # Second Hill function
  Vmax_b ~ Beta(prior mean)
  K_b ~ Gamma(prior mean)
  n_b ~ β * Normal(0, σ_n_b)

Dose-response (per cell i, junction j):
  Hill_a[i,j] = Vmax_a[j] * x[i]^n_a[j] / (K_a[j]^n_a[j] + x[i]^n_a[j])
  Hill_b[i,j] = Vmax_b[j] * x[i]^n_b[j] / (K_b[j]^n_b[j] + x[i]^n_b[j])

  p[i,j] = A[j] + α[j]*Hill_a[i,j] + β[j]*Hill_b[i,j]  # In [0,1]

Technical correction (if groups exist):
  logit(p_corrected[i,j]) = logit(p[i,j]) + log(alpha_y[group[i], j])

Likelihood:
  y_obs[i,j] ~ Binomial(denominator[i,j], p_corrected[i,j])
```

### Key Differences from Negbinom:
1. **Function space**: Hill functions operate in **probability space [0,1]**, not count space
2. **Priors**: `A`, `Vmax_a`, `Vmax_b` use **Beta** priors (not Exponential/Gamma)
3. **Technical correction**: Applied on **logit scale** (not multiplicative)
4. **Likelihood**: **Binomial** (not NegativeBinomial)
5. **No sum factors**: Denominator serves normalization role

This trace shows how binomial + additive_hill models splice junction usage (PSI) as a function of cis gene perturbation strength, accounting for baseline usage, two independent dose-response curves, and technical batch effects.
