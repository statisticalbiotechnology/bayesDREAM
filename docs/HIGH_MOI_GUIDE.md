# bayesDREAM High MOI Guide

This guide explains how to use bayesDREAM with **high MOI (multiplicity of infection)** data, where individual cells can contain multiple CRISPR guides.

## Overview

### What is High MOI?

In traditional CRISPR screens, each cell receives a single guide RNA. However, in **high MOI screens**, cells can be infected with multiple viral particles, resulting in:

- **Multiple guides per cell**: A single cell may contain 2, 3, or more distinct guide RNAs
- **Additive effects**: When a cell has multiple guides targeting the same gene, their effects combine additively
- **Mixed targeting**: Cells can have guides targeting different genes

### When to Use High MOI Mode

Use high MOI mode when:
- Your experimental design intentionally delivers multiple guides per cell
- Single-cell data shows cells with multiple guide barcodes
- You want to model combinatorial perturbation effects
- You have guide assignment information from single-cell barcode sequencing

Use **single-guide mode** (default) when:
- Each cell has at most one guide
- Your screen was designed for single-guide delivery (low MOI)
- You have traditional `guide` and `target` columns in your metadata

## Data Preparation

### Required Inputs

High MOI mode requires two additional inputs beyond standard bayesDREAM:

1. **`guide_assignment`**: Binary matrix indicating which guides are present in each cell
2. **`guide_meta`**: DataFrame with guide annotations

### Guide Assignment Matrix

A **binary matrix** with shape `(n_cells, n_guides)`:
- Rows = cells (matching order in `meta`)
- Columns = guides
- Entry `[i, j]` = 1 if cell `i` contains guide `j`, otherwise 0

**Example:**
```python
import numpy as np

# 5 cells, 4 guides
guide_assignment = np.array([
    [1, 1, 0, 0],  # Cell 0: has guides 0 and 1
    [0, 1, 0, 0],  # Cell 1: has guide 1 only
    [0, 0, 1, 0],  # Cell 2: has guide 2 only
    [0, 0, 0, 1],  # Cell 3: has guide 3 (NTC)
    [1, 0, 0, 0],  # Cell 4: has guide 0 only
])
```

**Important**:
- Matrix must be **dense** (not sparse) NumPy array
- Rows must match cell order in `meta` DataFrame
- Column order must match guide order in `guide_meta`

### Guide Metadata

A **DataFrame** with guide-level annotations:

**Required columns:**
- `guide`: Guide identifier (e.g., "sgGFI1B_1", "sgNTC_1")
- `target`: Target gene name (e.g., "GFI1B", "ntc")

**Example:**
```python
import pandas as pd

guide_meta = pd.DataFrame({
    'guide': ['sgGFI1B_1', 'sgGFI1B_2', 'sgMYB_1', 'sgNTC_1'],
    'target': ['GFI1B', 'GFI1B', 'MYB', 'ntc']
})
```

**Important**:
- Use `'ntc'` (lowercase) to mark non-targeting controls
- Number of rows must equal number of columns in `guide_assignment`
- Index can be guide names (will be preserved)

### Cell Metadata Differences

**Single-guide mode** `meta` has:
```python
meta = pd.DataFrame({
    'cell': ['cell_1', 'cell_2', 'cell_3'],
    'guide': ['sgGFI1B_1', 'sgGFI1B_2', 'sgNTC_1'],  # Required
    'target': ['GFI1B', 'GFI1B', 'ntc'],             # Required
    'cell_line': ['K562', 'K562', 'K562'],
    'sum_factor': [1.0, 1.1, 0.9]
})
```

**High MOI mode** `meta` has:
```python
meta = pd.DataFrame({
    'cell': ['cell_1', 'cell_2', 'cell_3'],
    # NO 'guide' or 'target' columns
    'cell_line': ['K562', 'K562', 'K562'],
    'sum_factor': [1.0, 1.1, 0.9]
})
```

The `target` column will be **automatically created** based on guide assignments.

## Complete Workflow

### Step 1: Load Data

```python
import pandas as pd
import numpy as np
from bayesDREAM import bayesDREAM

# Load cell metadata (no 'guide' or 'target' columns)
meta = pd.read_csv('meta.csv')

# Load gene counts
gene_counts = pd.read_csv('gene_counts.csv', index_col=0)

# Load guide assignment matrix
guide_assignment = np.load('guide_assignment.npy')  # (n_cells, n_guides)

# Load guide metadata
guide_meta = pd.read_csv('guide_meta.csv')
```

### Step 2: Initialize Model

```python
model = bayesDREAM(
    meta=meta,
    counts=gene_counts,
    guide_assignment=guide_assignment,  # Binary matrix
    guide_meta=guide_meta,              # Guide annotations
    cis_gene='GFI1B',
    output_dir='./output',
    label='high_moi_run'
)

# Verify high MOI mode is active
print(f"High MOI mode: {model.is_high_moi}")  # Should print True
```

**Automatic cell subsetting**:
- Cells with **only NTC guides** → kept as controls
- Cells with **any guides targeting cis_gene** → kept for cis fitting
- Cells with **guides targeting other genes only** → excluded

### Step 3: Run Standard Pipeline

The high MOI implementation works seamlessly with the standard three-step pipeline:

```python
# Step 1: Technical fitting (NTC cells only)
model.set_technical_groups(['cell_line'])
model.fit_technical()

# Step 2: Adjust sum factors for guide effects
model.adjust_ntc_sum_factor(
    sum_factor_col_old='sum_factor',
    sum_factor_col_adj='sum_factor_adj',
    covariates=['cell_line']
)

# Step 3: Cis fitting (with additive guide effects)
model.fit_cis(
    sum_factor_col='sum_factor_adj',
    technical_covariates=['cell_line']
)

# Step 4: Remove cis gene contribution from sum factors
model.refit_sumfactor(
    sum_factor_col_old='sum_factor_adj',
    sum_factor_col_refit='sum_factor_refit',
    covariates=['cell_line']
)

# Step 5: Trans fitting
model.fit_trans(
    sum_factor_col='sum_factor_refit',
    function_type='additive_hill'
)
```

### Step 4: Interpret Results

```python
# Access posterior samples
posterior_cis = model.posterior_samples_cis

# Per-guide effects (shape: [n_samples, n_guides])
x_eff_g = posterior_cis['x_eff_g']

# Get posterior means for each guide
guide_effects_mean = x_eff_g.mean(dim=0).cpu().numpy()

# Map to guide names
for i, guide_name in enumerate(model.guide_meta['guide']):
    target = model.guide_meta.loc[i, 'target']
    effect = guide_effects_mean[i]
    print(f"{guide_name} (targets {target}): {effect:.3f}")
```

## Key Differences from Single-Guide Mode

### 1. Data Input

| Feature | Single-Guide | High MOI |
|---------|-------------|----------|
| Cell metadata | Has 'guide' and 'target' columns | No 'guide' or 'target' columns |
| Guide info | In `meta` DataFrame | Separate `guide_assignment` matrix + `guide_meta` |
| Guides per cell | Exactly 1 | 0 to many |

### 2. Guide Effects Model

**Single-guide mode:**
```python
# Each cell has exactly one guide
x_true[cell_i] = x_eff_g[guide_i]
```

**High MOI mode (additive):**
```python
# Cell with multiple guides: effects sum
# Example: cell has guides 0, 1, and 2
x_true[cell_i] = x_eff_g[0] + x_eff_g[1] + x_eff_g[2]
```

This is implemented efficiently via matrix multiplication:
```python
x_true = guide_assignment @ x_eff_g
```

### 3. NTC Cell Identification

**Single-guide mode:**
```python
# Cell is NTC if its guide targets 'ntc'
is_ntc = (meta['target'] == 'ntc')
```

**High MOI mode:**
```python
# Cell is NTC only if ALL its guides are NTC guides
is_ntc = (all guides in cell are ntc guides)
```

### 4. Posterior Samples

**Single-guide mode:**
- `x_eff_g`: Shape `(n_samples, n_guides)` where `n_guides` = number of unique guides

**High MOI mode:**
- `x_eff_g`: Shape `(n_samples, n_guides)` where `n_guides` = number of total guides (from `guide_meta`)
- Includes all guides even if some cells have multiple

## Advanced Examples

### Example 1: Creating Guide Assignment from Barcode Data

If you have single-cell barcode counts, convert them to binary assignment:

```python
import pandas as pd
import numpy as np

# Load barcode counts (cells × guides)
barcode_counts = pd.read_csv('barcode_counts.csv', index_col=0)

# Threshold to determine presence (e.g., > 5 UMIs)
threshold = 5
guide_assignment = (barcode_counts.values > threshold).astype(int)

print(f"Guide assignment shape: {guide_assignment.shape}")
print(f"Cells with 2+ guides: {(guide_assignment.sum(axis=1) >= 2).sum()}")

# Create guide metadata from barcode names
guide_meta = pd.DataFrame({
    'guide': barcode_counts.columns,
    'target': barcode_counts.columns.str.extract(r'sg([A-Z0-9]+)_')[0]  # Extract target from name
})
```

### Example 2: Multi-Gene High MOI Screen

Screen with guides targeting multiple genes:

```python
# Guide metadata for multi-gene screen
guide_meta = pd.DataFrame({
    'guide': [
        'sgGFI1B_1', 'sgGFI1B_2', 'sgGFI1B_3',
        'sgTET2_1', 'sgTET2_2',
        'sgMYB_1', 'sgMYB_2',
        'sgNTC_1', 'sgNTC_2', 'sgNTC_3'
    ],
    'target': [
        'GFI1B', 'GFI1B', 'GFI1B',
        'TET2', 'TET2',
        'MYB', 'MYB',
        'ntc', 'ntc', 'ntc'
    ]
})

# When fitting for GFI1B:
model_gfi1b = bayesDREAM(
    meta=meta,
    counts=gene_counts,
    guide_assignment=guide_assignment,
    guide_meta=guide_meta,
    cis_gene='GFI1B',  # Focus on GFI1B
    output_dir='./output_gfi1b',
    label='gfi1b_high_moi'
)

# Cells will be categorized as:
# - NTC: only have sgNTC guides
# - GFI1B-targeting: have any sgGFI1B guides (even if also have sgMYB guides)
# - Other: only have non-GFI1B, non-NTC guides (excluded from fitting)
```

### Example 3: Analyzing Guide Combinations

Identify cells with specific guide combinations:

```python
# Find cells with both sgGFI1B_1 and sgGFI1B_2
guide_idx_1 = list(guide_meta['guide']).index('sgGFI1B_1')
guide_idx_2 = list(guide_meta['guide']).index('sgGFI1B_2')

has_both = (guide_assignment[:, guide_idx_1] == 1) & (guide_assignment[:, guide_idx_2] == 1)
print(f"Cells with both guides: {has_both.sum()}")

# After fitting, compare their effects
posterior_cis = model.posterior_samples_cis
x_eff_1 = posterior_cis['x_eff_g'][:, guide_idx_1].mean()
x_eff_2 = posterior_cis['x_eff_g'][:, guide_idx_2].mean()
print(f"Guide 1 effect: {x_eff_1:.3f}")
print(f"Guide 2 effect: {x_eff_2:.3f}")
print(f"Expected combined effect: {x_eff_1 + x_eff_2:.3f}")
```

### Example 4: Quality Control

Check guide assignment quality before fitting:

```python
import matplotlib.pyplot as plt

# Guides per cell distribution
guides_per_cell = guide_assignment.sum(axis=1)
plt.hist(guides_per_cell, bins=range(0, guides_per_cell.max() + 2))
plt.xlabel('Guides per cell')
plt.ylabel('Number of cells')
plt.title('Guide Assignment Distribution')
plt.show()

print(f"Mean guides per cell: {guides_per_cell.mean():.2f}")
print(f"Cells with 0 guides: {(guides_per_cell == 0).sum()}")
print(f"Cells with 1 guide: {(guides_per_cell == 1).sum()}")
print(f"Cells with 2+ guides: {(guides_per_cell >= 2).sum()}")

# Cells per guide distribution
cells_per_guide = guide_assignment.sum(axis=0)
for i, guide_name in enumerate(guide_meta['guide']):
    print(f"{guide_name}: {cells_per_guide[i]} cells")
```

## Best Practices

### 1. Guide Assignment Threshold

When converting barcode counts to binary assignment, choose threshold carefully:

**Too low** → False positives (ambient RNA, doublets)
**Too high** → False negatives (low MOI guides missed)

**Recommended approach:**
```python
# Use guide barcode UMI counts
barcode_umi = pd.read_csv('barcode_umi.csv', index_col=0)

# Set threshold based on distribution
threshold = np.percentile(barcode_umi[barcode_umi > 0], 25)  # 25th percentile of detected
print(f"Using threshold: {threshold} UMIs")

guide_assignment = (barcode_umi.values >= threshold).astype(int)
```

### 2. NTC Guide Representation

Ensure adequate NTC guides for normalization:
- Include multiple NTC guides (3-5 recommended)
- Ensure NTC guides are well-represented across cells
- Check that pure NTC cells (no targeting guides) are sufficient (>100 recommended)

```python
# Check NTC representation
ntc_guide_mask = guide_meta['target'] == 'ntc'
ntc_guide_indices = np.where(ntc_guide_mask)[0]

# Cells with only NTC guides
has_any_guide = guide_assignment.sum(axis=1) > 0
has_only_ntc = (guide_assignment[:, ntc_guide_indices].sum(axis=1) ==
                guide_assignment.sum(axis=1))
pure_ntc_cells = has_any_guide & has_only_ntc

print(f"Pure NTC cells: {pure_ntc_cells.sum()}")
if pure_ntc_cells.sum() < 100:
    print("WARNING: Fewer than 100 pure NTC cells. Consider adding more.")
```

### 3. Validate Additive Effects

After fitting, verify that additive model makes sense:

```python
# Get posterior samples
x_eff_g_samples = model.posterior_samples_cis['x_eff_g']
x_true_samples = model.posterior_samples_cis['x_true']

# For cells with multiple guides, check additivity
# Example: cells with guides 0 and 1
has_guide_0 = guide_assignment[:, 0] == 1
has_guide_1 = guide_assignment[:, 1] == 1
has_both = has_guide_0 & has_guide_1

if has_both.sum() > 0:
    # Expected (sum of guide effects)
    expected = (x_eff_g_samples[:, 0].mean() +
                x_eff_g_samples[:, 1].mean())

    # Observed (actual x_true for these cells)
    observed = x_true_samples[:, has_both].mean()

    print(f"Expected (additive): {expected:.3f}")
    print(f"Observed: {observed:.3f}")
    print(f"Difference: {abs(expected - observed):.3f}")
```

### 4. Handling Missing Guides

Some cells may have no guides detected:

```python
# Identify cells with no guides
no_guides = guide_assignment.sum(axis=1) == 0

if no_guides.sum() > 0:
    print(f"WARNING: {no_guides.sum()} cells have no guides assigned")
    print("Consider filtering these cells before bayesDREAM initialization")

    # Remove cells with no guides
    keep_cells = ~no_guides
    meta_filtered = meta[keep_cells]
    guide_assignment_filtered = guide_assignment[keep_cells, :]
    counts_filtered = gene_counts.loc[:, meta_filtered['cell']]
```

### 5. Memory Considerations

High MOI mode stores guide assignment matrix for all cells:

**Memory usage**: `n_cells × n_guides × 4 bytes` (float32)

For large screens (100K cells, 1000 guides):
- Matrix size: ~400 MB
- Consider subsetting to high-quality cells before initialization
- Use sparse representation if needed (convert to dense for bayesDREAM)

```python
from scipy.sparse import csr_matrix

# If starting with sparse matrix
guide_assignment_sparse = csr_matrix(guide_assignment_binary)

# Convert to dense for bayesDREAM
guide_assignment_dense = guide_assignment_sparse.toarray()

# Check memory
import sys
print(f"Memory: {sys.getsizeof(guide_assignment_dense) / 1e6:.1f} MB")
```

## Troubleshooting

### Error: "Both guide_assignment and guide_meta must be provided together"

**Cause**: One of the parameters is provided but not the other

**Solution**: Provide both `guide_assignment` and `guide_meta` to enable high MOI mode

```python
# WRONG
model = bayesDREAM(meta=meta, counts=counts, guide_assignment=guide_assignment)

# CORRECT
model = bayesDREAM(
    meta=meta,
    counts=counts,
    guide_assignment=guide_assignment,
    guide_meta=guide_meta
)
```

### Error: "guide_assignment must be a 2D matrix"

**Cause**: Guide assignment is not a 2D NumPy array

**Solution**: Ensure guide_assignment is a 2D array with shape `(n_cells, n_guides)`

```python
# Check shape
print(guide_assignment.shape)  # Should print (n_cells, n_guides)

# If 1D, reshape
if guide_assignment.ndim == 1:
    guide_assignment = guide_assignment.reshape(-1, 1)
```

### Error: "guide_meta has X rows but guide_assignment has Y columns"

**Cause**: Mismatch between number of guides in matrix and metadata

**Solution**: Ensure dimensions match

```python
n_cells, n_guides = guide_assignment.shape
print(f"guide_assignment: {n_cells} cells × {n_guides} guides")
print(f"guide_meta: {len(guide_meta)} guides")

# They must match
assert n_guides == len(guide_meta), "Dimension mismatch!"
```

### Error: "guide_meta missing required columns: {'guide', 'target'}"

**Cause**: Guide metadata is missing required columns

**Solution**: Ensure guide_meta has both 'guide' and 'target' columns

```python
# Check columns
print(guide_meta.columns.tolist())

# Add missing columns if needed
if 'guide' not in guide_meta.columns:
    guide_meta['guide'] = guide_meta.index  # Use index as guide names
```

### Warning: Too few NTC cells

**Cause**: After subsetting, very few pure NTC cells remain

**Solution**:
- Increase number of NTC guides in screen design
- Lower guide assignment threshold to capture more NTC cells
- Verify NTC guides are being correctly labeled in guide_meta

### Unexpected guide effects

**Symptom**: Guide effects don't match expectations or show unexpected patterns

**Debugging**:
1. Check guide assignment quality (see Quality Control section)
2. Verify NTC guides are correctly marked in guide_meta
3. Ensure sum factors are appropriate for your data
4. Check for batch effects or technical covariates

```python
# Inspect guide effects
x_eff_g_mean = model.posterior_samples_cis['x_eff_g'].mean(dim=0)

for i, guide_name in enumerate(model.guide_meta['guide']):
    target = model.guide_meta.loc[i, 'target']
    effect = x_eff_g_mean[i].item()
    print(f"{guide_name} → {target}: {effect:.3f}")

# NTC guides should have effects near 0
ntc_guides = model.guide_meta['target'] == 'ntc'
print(f"\nNTC guide effects: {x_eff_g_mean[ntc_guides].numpy()}")
```

## Comparison with Single-Guide Mode

### When to Use Each Mode

| Feature | Single-Guide | High MOI |
|---------|-------------|----------|
| **Experimental design** | Low MOI (one guide/cell) | High MOI (multiple guides/cell) |
| **Data structure** | Simple (guide in metadata) | Complex (binary matrix) |
| **Combinatorial effects** | Not supported | Additive combination |
| **Computational cost** | Lower | Slightly higher |
| **Setup complexity** | Simple | Moderate |

### Converting Between Modes

**Single-guide to high MOI** (for mixed designs):

```python
# If you have traditional single-guide data but want to use high MOI infrastructure
meta_single = pd.DataFrame({
    'cell': ['c1', 'c2', 'c3'],
    'guide': ['g1', 'g2', 'ntc'],
    'target': ['GFI1B', 'GFI1B', 'ntc']
})

# Convert to high MOI format
unique_guides = meta_single['guide'].unique()
guide_meta = pd.DataFrame({
    'guide': unique_guides,
    'target': meta_single.groupby('guide')['target'].first().loc[unique_guides].values
})

# Create binary assignment matrix
n_cells = len(meta_single)
n_guides = len(unique_guides)
guide_assignment = np.zeros((n_cells, n_guides), dtype=int)

guide_to_idx = {g: i for i, g in enumerate(unique_guides)}
for i, guide in enumerate(meta_single['guide']):
    guide_assignment[i, guide_to_idx[guide]] = 1

# Remove guide/target from meta
meta_high_moi = meta_single.drop(columns=['guide', 'target'])

# Use high MOI mode
model = bayesDREAM(
    meta=meta_high_moi,
    counts=counts,
    guide_assignment=guide_assignment,
    guide_meta=guide_meta,
    cis_gene='GFI1B'
)
```

## Related Documentation

- **[HIGH_MOI_DESIGN.md](HIGH_MOI_DESIGN.md)** - Technical design document and implementation details
- **[API_REFERENCE.md](API_REFERENCE.md)** - Complete API documentation
- **[QUICKSTART_MULTIMODAL.md](QUICKSTART_MULTIMODAL.md)** - Quick start guide
- **[FIT_TRANS_GUIDE.md](FIT_TRANS_GUIDE.md)** - Trans fitting guide
- **[tests/test_high_moi.py](../tests/test_high_moi.py)** - Test suite with examples
