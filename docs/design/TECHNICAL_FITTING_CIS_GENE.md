# Technical Fitting and Cis Gene Handling

## Overview

When fitting technical effects (`fit_technical`) on the **primary modality**, bayesDREAM needs to handle the cis gene specially:

1. **Extract `alpha_x_prefit`**: The overdispersion parameter for the cis gene
2. **Store `alpha_y_prefit`**: The overdispersion parameters for all trans genes (excluding cis gene)

This ensures that:
- The cis gene's technical effects are estimated but not included in trans gene modeling
- `fit_cis()` can use the pre-fitted `alpha_x` to account for technical variation
- `fit_trans()` models trans genes without the cis gene

## Implementation

### When `fit_technical()` is called on the PRIMARY modality

**Location**: `bayesDREAM/model.py` lines 1578-1626

```python
if modality_name == self.primary_modality:
    # Store at model level for backward compatibility
    self.loss_technical = losses
    self.posterior_samples_technical = posterior_samples

    if self.cis_gene is not None:
        # Get full alpha_y: Shape (S, C, T_total)
        full_alpha_y = posterior_samples["alpha_y"]

        # Check if cis_gene is in the original counts
        if hasattr(self, 'counts') and self.cis_gene in self.counts.index:
            # Find index of cis_gene
            cis_idx = list(self.counts.index).index(self.cis_gene)

            # Extract alpha for cis gene: (S, C)
            self.alpha_x_prefit = full_alpha_y[:, :, cis_idx]
            self.alpha_x_type = 'posterior'

            # Store alphas for trans genes only
            trans_indices = [i for i in range(full_alpha_y.shape[2]) if i != cis_idx]
            self.alpha_y_prefit = full_alpha_y[:, :, trans_indices]
            self.alpha_y_type = 'posterior'

            print(f"[INFO] Extracted alpha_x_prefit for cis gene '{self.cis_gene}'")
            print(f"[INFO] alpha_y_prefit excludes cis gene ({len(trans_indices)} trans genes)")
        else:
            # Cis gene not in this modality (e.g., ATAC with placeholder cis_gene)
            self.alpha_y_prefit = full_alpha_y
            self.alpha_y_type = 'posterior'
            print(f"[INFO] Cis gene not in primary modality - alpha_x will be fitted in fit_cis")
```

### Key Behaviors

1. **Gene expression primary modality** (standard workflow):
   - `fit_technical()` on 'gene' modality
   - `alpha_x_prefit` extracted from fitted overdispersion for `cis_gene`
   - `alpha_y_prefit` contains all other genes (trans genes)

2. **ATAC primary modality** (ATAC-only workflow):
   - `fit_technical()` on 'atac' modality
   - If `cis_gene` is just a placeholder (not in ATAC features), `alpha_x_prefit` is NOT set
   - `fit_cis()` will fit alpha_x fresh (no pre-fitted technical effects for cis)

## Warning in `fit_cis()` (But Still Allowed)

**Location**: `bayesDREAM/model.py` lines 1905-1912

When `fit_cis()` is called with technical covariates but `alpha_x_prefit` doesn't exist:

```python
if technical_covariates:
    # ... setup technical groups ...

    # Check if alpha_x_prefit should exist but doesn't
    if self.alpha_x_prefit is None:
        warnings.warn(
            f"Technical covariates provided but alpha_x_prefit not set. "
            f"You should run fit_technical() on the primary modality ('{self.primary_modality}') first "
            f"to estimate technical effects for the cis gene. "
            f"Proceeding without technical correction for cis gene (alpha_x will be fitted fresh)."
        )
```

**Important**: This is a **warning, not an error**. The code will proceed.

### What Happens in `_model_x`

**Location**: `bayesDREAM/model.py` lines 1669-1677

```python
if alpha_x_sample is not None:
    alpha_x = alpha_x_sample  # Use pre-fitted from fit_technical
elif groups_tensor is not None:
    # Technical covariates exist but no pre-fitted alpha_x
    # FIT alpha_x FRESH in this model
    with pyro.plate("c_plate", C - 1):
        alpha_alpha = pyro.sample("alpha_alpha", dist.Exponential(...))
        alpha_mu = pyro.sample("alpha_mu", dist.Gamma(1, 1))
        alpha_x = pyro.sample("alpha_x", dist.Gamma(...))
else:
    alpha_x = None  # No technical covariates
```

### Behavior Summary

| `fit_technical` run? | Technical covariates | `alpha_x_prefit` | What happens | Warning? |
|---------------------|---------------------|-----------------|--------------|----------|
| ✅ Yes (primary) | Yes | Set | Uses pre-fitted alpha_x | No |
| ❌ No | Yes | None | **Fits alpha_x fresh** | ⚠️ Yes |
| ❌ No | No | None | No alpha_x (no correction) | ⚠️ Yes (different warning) |
| ✅ Yes (primary) | No | Set | Uses pre-fitted alpha_x | No |

**Key Point**: Even without `fit_technical`, if technical covariates are provided, `_model_x` will **still fit alpha_x** - it just won't use pre-fitted values from NTC cells. This is allowed but not recommended.

## Example Workflows

### Workflow 1: Standard Gene Expression (recommended)

```python
model = bayesDREAM(
    meta=meta,
    counts=gene_counts,
    cis_gene='GFI1B',
    primary_modality='gene'
)

# Fit technical on primary modality (gene)
model.set_technical_groups(['cell_line'])
model.fit_technical(modality_name='gene', sum_factor_col='sum_factor')
# Result: alpha_x_prefit set for GFI1B, alpha_y_prefit set for trans genes

# Fit cis (uses alpha_x_prefit for technical correction)
model.fit_cis(sum_factor_col='sum_factor')

# Fit trans (uses alpha_y_prefit for trans genes)
model.fit_trans(sum_factor_col='sum_factor_adj', function_type='additive_hill')
```

### Workflow 2: ATAC-Only (cis gene not in ATAC features)

```python
model = bayesDREAM(
    meta=meta,
    counts=None,
    cis_gene='GFI1B',  # Just for naming, not in ATAC
    primary_modality='atac'
)

model.add_atac_modality(
    atac_counts=atac_counts,
    region_meta=region_meta,
    cis_region='chr9:1000-2000'  # Promoter region for GFI1B
)

# Fit technical on ATAC
model.set_technical_groups(['cell_line'])
model.fit_technical(modality_name='atac', sum_factor_col='sum_factor')
# Result: alpha_y_prefit set for all ATAC regions
# alpha_x_prefit NOT set (GFI1B not in ATAC features)

# Fit cis (alpha_x fitted fresh, no pre-fitted technical correction)
# Warning will be shown if technical_covariates exist
model.fit_cis(
    cis_feature='chr9:1000-2000',  # Use ATAC promoter as proxy
    sum_factor_col='sum_factor'
)
```

### Workflow 3: ATAC-Only (cis gene IS an ATAC feature)

```python
model = bayesDREAM(
    meta=meta,
    counts=None,
    cis_gene='chr9:1000-2000',  # Use ATAC region ID as cis_gene
    primary_modality='atac'
)

model.add_atac_modality(
    atac_counts=atac_counts,
    region_meta=region_meta,
    cis_region='chr9:1000-2000'
)

# Fit technical on ATAC
model.set_technical_groups(['cell_line'])
model.fit_technical(modality_name='atac', sum_factor_col='sum_factor')
# Result: alpha_x_prefit set for 'chr9:1000-2000'
# alpha_y_prefit set for all other ATAC regions

# Fit cis (uses alpha_x_prefit for technical correction)
model.fit_cis(sum_factor_col='sum_factor')
```

## Summary

| Scenario | `fit_technical` on primary | `cis_gene` in primary | `alpha_x_prefit` set? | `alpha_y_prefit` content |
|----------|---------------------------|----------------------|----------------------|--------------------------|
| Gene expression | Yes | Yes | ✅ Yes | Trans genes (exclude cis) |
| ATAC (cis not in features) | Yes | No | ❌ No | All ATAC regions |
| ATAC (cis in features) | Yes | Yes | ✅ Yes | Other ATAC regions (exclude cis) |
| No fit_technical | No | N/A | ❌ No | Not set |

## Technical Details

### Shape Conventions

- `alpha_x_prefit`: Shape `(S, C)` where S = samples, C = technical groups
- `alpha_y_prefit`: Shape `(S, C, T)` where T = number of trans genes/features
- `alpha_x_type`: Either `'posterior'` (from MCMC) or `'point'` (single value)
- `alpha_y_type`: Either `'posterior'` or `'point'`

### When to Use Point Estimates vs Posterior

- **Posterior** (default): Full uncertainty from MCMC sampling
- **Point**: For computational efficiency or when uncertainty is not needed

Use `set_alpha_x()` or `set_alpha_y()` to manually override with point estimates if needed.
