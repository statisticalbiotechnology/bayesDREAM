# Cell Subsetting for Debug

## The Issue

When trying to test without technical correction by subsetting to CRISPRi or CRISPRa only,
`model.subset_cells()` doesn't exist.

## Workaround: Manual Subsetting

Since `subset_cells()` doesn't exist yet, you can manually subset for testing:

```python
# Option 1: Set alpha_y to None (simplest)
model.modalities['splicing_sj'].alpha_y_prefit_add = None
model.modalities['splicing_sj'].alpha_y_prefit = None

# Fit without technical correction
model.fit_trans(
    modality_name='splicing_sj',
    function_type='additive_hill',
    min_denominator=3
)
```

```python
# Option 2: Subset meta to single cell_line
# This only works if you refit from scratch

# Filter metadata to CRISPRa only
crispra_mask = model.meta['cell_line'].str.contains('CRISPRa', case=False, na=False)
meta_subset = model.meta[crispra_mask].copy()

# Reinitialize model with subset
from bayesDREAM import bayesDREAM
model_crispra = bayesDREAM(
    meta=meta_subset,
    counts=gene_counts[meta_subset['cell']],  # Subset counts to matching cells
    gene_meta=gene_meta,
    cis_gene='GFI1B',
    output_dir='./output',
    label='crispra_only'
)

# Add splicing modality
model_crispra.add_splicing_modality(
    sj_counts=sj_counts,
    sj_meta=sj_meta,
    splicing_types=['sj']
)

# Fit without technical groups (only 1 cell_line)
model_crispra.fit_trans(
    modality_name='splicing_sj',
    function_type='additive_hill'
)
```

## The Real Problem: Dimension Mismatch

The min_denominator error is NOT because of subsetting - it's because:

1. **alpha_y_prefit** was fitted on **genes** (1801 features)
2. But you're trying to apply it to **splicing junctions** (4281 features)

### Fix: Run fit_technical on the Same Modality

```python
# IMPORTANT: Fit technical on splicing_sj, not genes!
model.set_technical_groups(['cell_line'])
model.fit_technical(
    modality_name='splicing_sj',  # Same modality as trans fitting
    sum_factor_col=None  # Binomial doesn't use sum factors
)

# Now alpha_y_prefit will have shape [C, 4281] matching splicing junctions
model.fit_trans(
    modality_name='splicing_sj',
    function_type='additive_hill',
    min_denominator=3
)
```

## Diagnostic Script Fix

The diagnostic script is failing because it's looking for `technical_group_code` at the wrong
time. The error suggests it's checking `model.meta` instead of using the already-set groups.

Try running the diagnostic AFTER calling `set_technical_groups()`:

```python
# Make sure this was called first
model.set_technical_groups(['cell_line'])

# Now diagnostic should work
from diagnose_alpha_y import diagnose_alpha_y
results = diagnose_alpha_y(model, modality_name='splicing_sj')
```

If it still fails, the issue is in the diagnostic script itself - it may need to be updated
to handle the case where technical_group_code is already set.
