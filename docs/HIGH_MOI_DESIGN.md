# High MOI (Multiple Guides per Cell) Design

## Overview

Enhancement to support high MOI experiments where cells can have multiple guides, with **additive guide effects** in cis modeling.

## Current Architecture (Single Guide per Cell)

### Initialization
- `meta` has columns: `cell`, `guide`, `target`, `sum_factor`
- Each cell has exactly one guide
- `guide_code` created from unique guides

### Cis Modeling
- N = number of cells
- G = number of unique guides
- `guides_tensor`: [N] - maps each cell to its guide index
- `x_eff_g`: [G] - effect per guide (sampled in model)
- `x_mean = x_eff_g[guides_tensor]`: [N] - lookup effect for each cell

## New Architecture (High MOI)

### Initialization

#### New Parameters
```python
bayesDREAM(
    meta=meta,  # Now WITHOUT 'guide' and 'target' columns
    counts=counts,
    guide_assignment=guide_matrix,  # Binary matrix (cells × guides)
    guide_meta=guide_meta,  # DataFrame with 'guide' and 'target' columns
    ...
)
```

#### Data Structures

**guide_assignment**: Binary matrix `[N, G]`
- Shape: (n_cells, n_guides)
- guide_assignment[i, j] = 1 if cell i has guide j, else 0
- Each row can have multiple 1s (multiple guides per cell)
- NTC cells have all zeros OR have ntc guide indices set to 1

**guide_meta**: DataFrame
- Required columns: `guide`, `target`
- Index: guide identifiers (must match column order of guide_assignment)
- Example:
  ```python
  guide_meta = pd.DataFrame({
      'guide': ['guide_1', 'guide_2', 'ntc_1', 'ntc_2'],
      'target': ['GFI1B', 'GFI1B', 'ntc', 'ntc']
  }, index=['guide_1', 'guide_2', 'ntc_1', 'ntc_2'])
  ```

#### Backward Compatibility
- If `guide_assignment` is None, use existing behavior:
  - Require `guide` and `target` columns in `meta`
  - Create one-hot encoding internally
- If `guide_assignment` is provided:
  - `guide` and `target` columns NOT required in `meta`
  - Must provide `guide_meta`

### Cis Modeling Changes

#### Guide Effects (Additive)
```python
# Sample per-guide effects (same as before)
with pyro.plate("guides_plate", G):
    x_eff_g = ...  # Shape: [G]
    sigma_eff = ...  # Shape: [G]

# Aggregate to cells (NEW for high MOI)
if self.model.is_high_moi:
    # Sum guide effects for each cell
    # guide_assignment_tensor: [N, G], x_eff_g: [G]
    x_mean = torch.matmul(guide_assignment_tensor.float(), x_eff_g)  # [N]

    # For sigma: take weighted average or RMS
    # Option 1: Average sigma (simpler)
    guides_per_cell = guide_assignment_tensor.sum(dim=1)  # [N]
    sigma_mean = torch.matmul(guide_assignment_tensor.float(), sigma_eff) / guides_per_cell.clamp(min=1)

    # Option 2: Root mean square (more conservative)
    # sigma_sq = torch.matmul(guide_assignment_tensor.float(), sigma_eff ** 2) / guides_per_cell.clamp(min=1)
    # sigma_mean = torch.sqrt(sigma_sq)
else:
    # Single guide per cell (existing behavior)
    x_mean = x_eff_g[..., guides_tensor]  # [N]
    sigma_mean = sigma_eff[..., guides_tensor]  # [N]
```

## Implementation Steps

### Step 1: Update `_BayesDREAMCore.__init__` (core.py)

1. **Add optional parameters**:
   ```python
   def __init__(
       self,
       meta: pd.DataFrame,
       counts: pd.DataFrame,
       guide_assignment: np.ndarray = None,  # NEW: [N, G] binary matrix
       guide_meta: pd.DataFrame = None,      # NEW: guide metadata
       ...
   )
   ```

2. **Detect mode and validate**:
   ```python
   # Detect high MOI mode
   if guide_assignment is not None or guide_meta is not None:
       if guide_assignment is None or guide_meta is None:
           raise ValueError("Both guide_assignment and guide_meta must be provided for high MOI mode")
       self.is_high_moi = True
   else:
       self.is_high_moi = False
   ```

3. **Validation for high MOI**:
   ```python
   if self.is_high_moi:
       # Validate guide_assignment
       if guide_assignment.ndim != 2:
           raise ValueError("guide_assignment must be 2D matrix (cells × guides)")

       N_cells, G_guides = guide_assignment.shape

       # Validate against meta and counts
       if N_cells != len(meta):
           raise ValueError(f"guide_assignment has {N_cells} cells but meta has {len(meta)}")

       # Validate guide_meta
       if len(guide_meta) != G_guides:
           raise ValueError(f"guide_meta has {len(guide_meta)} guides but guide_assignment has {G_guides}")

       required_cols = {'guide', 'target'}
       missing = required_cols - set(guide_meta.columns)
       if missing:
           raise ValueError(f"guide_meta missing columns: {missing}")

       # Store
       self.guide_assignment = guide_assignment
       self.guide_meta = guide_meta.copy()
       self.guide_assignment_tensor = torch.tensor(guide_assignment, dtype=torch.float32, device=self.device)

       # Create guide_code mapping
       self.guide_meta['guide_code'] = range(G_guides)

       # Determine which cells are NTC (all guides are NTC)
       ntc_guide_mask = guide_meta['target'] == 'ntc'
       ntc_guide_indices = np.where(ntc_guide_mask)[0]

       # Cell is NTC if it ONLY has NTC guides (and at least one guide)
       has_any_guide = guide_assignment.sum(axis=1) > 0
       has_only_ntc = guide_assignment[:, ntc_guide_indices].sum(axis=1) == guide_assignment.sum(axis=1)
       is_ntc_cell = has_any_guide & has_only_ntc

       # Add 'target' to meta for compatibility
       self.meta['target'] = ['ntc' if is_ntc else self.cis_gene for is_ntc in is_ntc_cell]

       # Add guide_code column (not meaningful in high MOI, but kept for compatibility)
       # Use -1 to indicate high MOI mode
       self.meta['guide_code'] = -1

       print(f"[INFO] High MOI mode: {N_cells} cells, {G_guides} guides")
       print(f"[INFO] Average guides per cell: {guide_assignment.sum(axis=1).mean():.2f}")
       print(f"[INFO] NTC cells: {is_ntc_cell.sum()}, Non-NTC cells: {(~is_ntc_cell).sum()}")
   ```

4. **For single-guide mode** (backward compatibility):
   ```python
   else:
       # Existing validation (requires 'guide' and 'target' in meta)
       required_cols = {"target", "cell", sum_factor_col, "guide"}
       ...
       # Create guide_code as before
       self.meta['guide_code'] = pd.Categorical(self.meta['guide_used']).codes
   ```

### Step 2: Update `CisFitter._model_x` (fitting/cis.py)

1. **Change guide effect aggregation** (around line 153):
   ```python
   ##########################
   ## Cell-level variables ##
   ##########################
   if alpha_x is not None:
       ones_ = torch.ones(alpha_x.shape[:-1] + (1,), device=self.model.device)
       alpha_x_full = torch.cat([ones_, alpha_x], dim=-1)
       alpha_x_used = alpha_x_full[..., groups_tensor]
   else:
       alpha_x_used = torch.ones_like(sum_factor_tensor)

   # Aggregate guide effects to cells
   if self.model.is_high_moi:
       # High MOI: sum guide effects (additive)
       # guide_assignment_tensor: [N, G], x_eff_g: [G]
       x_mean_per_guide_sum = torch.matmul(
           self.model.guide_assignment_tensor,
           x_eff_g
       )  # [N]

       # For sigma: average across guides in each cell
       guides_per_cell = self.model.guide_assignment_tensor.sum(dim=1).clamp(min=1)  # [N]
       sigma_mean = torch.matmul(
           self.model.guide_assignment_tensor,
           sigma_eff
       ) / guides_per_cell  # [N]

       x_mean = x_mean_per_guide_sum
   else:
       # Single guide per cell: use indexing
       x_mean = x_eff_g[..., guides_tensor]  # [N]
       sigma_mean = sigma_eff[..., guides_tensor]  # [N]

   ######################
   ## Cell-level plate ##
   ######################
   with pyro.plate("data_plate", N):
       log_x_true = pyro.sample(
           "log_x_true",
           dist.Normal(torch.log2(x_mean), sigma_mean)  # Use sigma_mean instead of indexing
       )
       x_true = pyro.deterministic("x_true", self._t(2.0) ** log_x_true)
       mu_obs = alpha_x_used * x_true * sum_factor_tensor
       pyro.sample(
           "x_obs",
           dist.NegativeBinomial(
               total_count=phi_x_used,
               logits=torch.log(mu_obs) - torch.log(phi_x_used),
               validate_args=False
           ),
           obs=x_obs_tensor
       )
   ```

2. **Update fit_cis guide count** (around line 277):
   ```python
   if self.model.is_high_moi:
       G = self.model.guide_assignment.shape[1]  # Number of guides
       guides_tensor = None  # Not used in high MOI
   else:
       G = self.model.meta['guide_code'].nunique()
       guides_tensor = torch.tensor(
           self.model.meta['guide_code'].values,
           dtype=torch.long,
           device=self.model.device
       )
   ```

3. **Update guide means computation** (around line 397):
   ```python
   if self.model.is_high_moi:
       # For high MOI, compute per-guide means differently
       # For each guide, find cells that have it
       guide_means = []
       for g in range(G):
           cells_with_guide = self.model.guide_assignment[:, g] == 1
           if cells_with_guide.sum() > 0:
               guide_mean = torch.log2(torch.mean(x_obs_factored[cells_with_guide]))
               guide_means.append(guide_mean)
       guide_means = torch.tensor(guide_means, dtype=torch.float32, device=self.model.device)
   else:
       # Existing single-guide logic
       unique_guides = torch.unique(guides_tensor)
       guide_means = torch.tensor([
           torch.log2(torch.mean(x_obs_factored[guides_tensor == g]))
           for g in unique_guides
           if torch.sum(x_obs_factored[guides_tensor == g]) > 0
       ], dtype=torch.float32, device=self.model.device)
   ```

### Step 3: Handle Other Components

#### Technical Fitting
- Should work without changes (operates on NTC cells, doesn't depend on guide structure)

#### Trans Fitting
- Uses `x_true` from cis fit (per-cell values)
- Should work without changes since x_true is still per-cell

#### Permutation
- May need updates for high MOI mode
- Can add in future PR if needed

## Testing Strategy

### Test Cases

1. **Basic high MOI functionality**:
   - Create synthetic data with 2 guides per cell
   - Verify additive effects
   - Check that x_true reflects sum of guide effects

2. **Mixed MOI (some cells single, some multiple)**:
   - Test with variable guides per cell
   - Verify correct aggregation

3. **NTC handling**:
   - Ensure NTC cells work correctly
   - Verify target assignment

4. **Backward compatibility**:
   - Existing tests should pass unchanged
   - Single-guide mode should work as before

### Example Test Data
```python
# 10 cells, 4 guides (2 targeting GFI1B, 2 NTC)
guide_assignment = np.array([
    [1, 1, 0, 0],  # Cell 0: guides 0,1 (both target GFI1B)
    [1, 0, 0, 0],  # Cell 1: guide 0 (targets GFI1B)
    [0, 1, 0, 0],  # Cell 2: guide 1 (targets GFI1B)
    [1, 1, 0, 0],  # Cell 3: guides 0,1
    [0, 0, 1, 1],  # Cell 4: guides 2,3 (both NTC)
    [0, 0, 1, 0],  # Cell 5: guide 2 (NTC)
    [0, 0, 0, 1],  # Cell 6: guide 3 (NTC)
    ...
])

guide_meta = pd.DataFrame({
    'guide': ['guide_A', 'guide_B', 'ntc_1', 'ntc_2'],
    'target': ['GFI1B', 'GFI1B', 'ntc', 'ntc']
}, index=['guide_A', 'guide_B', 'ntc_1', 'ntc_2'])
```

## Open Questions

1. **Sigma aggregation**: Average vs RMS vs other methods?
   - **Recommendation**: Start with average (simpler), can add options later

2. **Covariate handling**: Should guide_used still be created?
   - **Recommendation**: Not needed in high MOI mode

3. **Manual guide priors**: How to specify for high MOI?
   - **Recommendation**: Use guide names from guide_meta (same as before)

4. **Sparse vs dense matrix**: Should we use sparse representation?
   - **Recommendation**: Start with dense, optimize later if needed

## Documentation Updates Needed

- Update API_REFERENCE.md with new parameters
- Add HIGH_MOI_GUIDE.md with usage examples
- Update QUICKSTART.md with high MOI example
- Add to OUTSTANDING_TASKS.md when complete
