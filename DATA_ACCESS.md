# Data Access Guide

Guide to accessing and working with data in bayesDREAM.

## Table of Contents

- [Accessing Modalities](#accessing-modalities)
- [Working with Count Data](#working-with-count-data)
- [Accessing Metadata](#accessing-metadata)
- [Subsetting Data](#subsetting-data)
- [Converting Data Formats](#converting-data-formats)
- [Accessing Model Results](#accessing-model-results)
- [Working with Posterior Samples](#working-with-posterior-samples)

---

## Accessing Modalities

### List All Modalities

```python
# Get summary of all modalities
summary = model.list_modalities()
print(summary)
```

Output:
```
                  name distribution  n_features  n_cells  n_categories
0                 gene     negbinom        1000      500           NaN
1   splicing_donor  multinomial          50      500          10.0
2               spliz       normal        1000      500           NaN
```

### Get Specific Modality

```python
# Get a modality by name
donor_mod = model.get_modality('splicing_donor')

# Access modality attributes
print(donor_mod.name)           # 'splicing_donor'
print(donor_mod.distribution)   # 'multinomial'
print(donor_mod.dims)           # {'n_features': 50, 'n_cells': 500, 'n_categories': 10}
```

### Check if Modality Exists

```python
if 'splicing_donor' in model.modalities:
    donor_mod = model.get_modality('splicing_donor')
else:
    print("Modality not found")
```

---

## Working with Count Data

### Access Raw Counts

```python
# Get counts from primary modality (genes)
gene_counts = model.counts  # pandas DataFrame (genes × cells)

# Get counts from specific modality
donor_mod = model.get_modality('splicing_donor')
donor_counts = donor_mod.counts  # numpy array (donors × cells × acceptors)

# Check shape
print(gene_counts.shape)    # (1000, 500)
print(donor_counts.shape)   # (50, 500, 10)
```

### 2D Count Data (negbinom, normal, binomial)

```python
# Gene counts (negbinom)
gene_mod = model.get_modality('gene')
gene_counts = gene_mod.counts  # shape: (n_genes, n_cells)

# Access specific gene
gene_idx = 0
gene_expression = gene_counts[gene_idx, :]  # expression across all cells

# Access specific cell
cell_idx = 10
cell_profile = gene_counts[:, cell_idx]  # all genes in this cell
```

### 3D Count Data (multinomial, mvnormal)

```python
# Donor usage (multinomial)
donor_mod = model.get_modality('splicing_donor')
donor_counts = donor_mod.counts  # shape: (n_donors, n_cells, n_acceptors)

# Access specific donor
donor_idx = 5
donor_usage = donor_counts[donor_idx, :, :]  # cells × acceptors

# Access specific cell
cell_idx = 10
cell_donors = donor_counts[:, cell_idx, :]  # donors × acceptors for this cell

# SpliZVD (mvnormal)
splizvd_mod = model.get_modality('splizvd')
splizvd_data = splizvd_mod.counts  # shape: (n_genes, n_cells, 3)
# 3rd dimension: [z0, z1, z2]
```

### Binomial with Denominator

```python
# Exon skipping
exon_mod = model.get_modality('splicing_exon_skip')
inclusion = exon_mod.counts         # numerator (events × cells)
total = exon_mod.denominator        # denominator (events × cells)

# Calculate PSI (proportion spliced in)
psi = inclusion / (total + 1e-9)  # add small constant to avoid division by zero
```

---

## Accessing Metadata

### Cell Metadata

```python
# Access full cell metadata
meta = model.meta  # pandas DataFrame

# Key columns
print(meta['cell'])         # Cell barcodes
print(meta['guide'])        # Guide RNA names
print(meta['target'])       # Target genes
print(meta['cell_line'])    # Cell line identifiers
print(meta['sum_factor'])   # Normalization factors

# Check for adjusted sum factors
if 'sum_factor_adj' in meta.columns:
    print(meta['sum_factor_adj'])
```

### Feature Metadata

```python
# Gene metadata
gene_mod = model.get_modality('gene')
gene_meta = gene_mod.feature_meta
print(gene_meta.index)  # Gene names

# Donor metadata
donor_mod = model.get_modality('splicing_donor')
donor_meta = donor_mod.feature_meta
print(donor_meta.columns)
# Output: ['chrom', 'strand', 'donor', 'acceptors', 'n_acceptors']

# Access specific donor info
print(donor_meta.loc[0, 'donor'])      # Donor position
print(donor_meta.loc[0, 'acceptors'])  # List of acceptor positions
print(donor_meta.loc[0, 'n_acceptors']) # Number of acceptors

# Exon skipping metadata
exon_mod = model.get_modality('splicing_exon_skip')
exon_meta = exon_mod.feature_meta
print(exon_meta.columns)
# Output: ['trip_id', 'chrom', 'strand', 'd1', 'a2', 'd2', 'a3', 'sj_inc1', 'sj_inc2', 'sj_skip']
```

### Guide-Level Metadata

```python
# Access guide-level summaries
guide_meta = model.meta.groupby('guide').first()
print(guide_meta['target'])  # Target for each guide
```

---

## Subsetting Data

### Subset by Features

```python
# Get specific genes
gene_names = ['GFI1B', 'GATA1', 'TAL1']
gene_mod = model.get_modality('gene')

# Find indices
gene_indices = [i for i, name in enumerate(gene_mod.feature_meta.index)
                if name in gene_names]

# Subset modality
subset_mod = gene_mod.get_feature_subset(gene_indices)
print(subset_mod.counts.shape)  # (3, n_cells)
```

### Subset by Cells

```python
# Get cells from specific guide
guide_name = 'gRNA1'
cell_indices = model.meta[model.meta['guide'] == guide_name].index

# Subset modality
donor_mod = model.get_modality('splicing_donor')
subset_mod = donor_mod.get_cell_subset(cell_indices)
print(subset_mod.counts.shape)  # (n_donors, n_guide_cells, n_acceptors)
```

### Filter by Cell Line

```python
# Get cells from K562 cell line
k562_cells = model.meta[model.meta['cell_line'] == 'K562']
cell_indices = k562_cells.index

# Subset all modalities
for mod_name in model.modalities.keys():
    mod = model.get_modality(mod_name)
    subset = mod.get_cell_subset(cell_indices)
    print(f"{mod_name}: {subset.dims}")
```

### Filter by NTC vs Perturbed

```python
# NTC cells
ntc_cells = model.meta[model.meta['target'] == 'ntc']
ntc_indices = ntc_cells.index

# Perturbed cells (targeting cis gene)
cis_cells = model.meta[model.meta['target'] == model.cis_gene]
cis_indices = cis_cells.index

# Get gene counts for each
gene_mod = model.get_modality('gene')
ntc_counts = gene_mod.get_cell_subset(ntc_indices).counts
cis_counts = gene_mod.get_cell_subset(cis_indices).counts
```

---

## Converting Data Formats

### Convert to PyTorch Tensors

```python
# Convert counts to tensor
gene_mod = model.get_modality('gene')
gene_tensor = gene_mod.to_tensor(device='cuda')  # or 'cpu'
print(type(gene_tensor))  # torch.Tensor
print(gene_tensor.device)  # cuda:0 or cpu

# Convert 3D data
donor_mod = model.get_modality('splicing_donor')
donor_tensor = donor_mod.to_tensor(device='cpu')
print(donor_tensor.shape)  # torch.Size([50, 500, 10])
```

### Convert to Pandas DataFrame

```python
# Counts are already pandas for 2D primary modality
gene_counts = model.counts  # pandas DataFrame

# Convert numpy array to DataFrame
donor_mod = model.get_modality('splicing_donor')
# For 2D slice
donor_cell0 = donor_mod.counts[:, 0, :]  # donors × acceptors
donor_df = pd.DataFrame(
    donor_cell0,
    index=donor_mod.feature_meta.index,
    columns=[f'acceptor_{i}' for i in range(donor_cell0.shape[1])]
)
```

### Convert to AnnData (for scanpy)

```python
import anndata

# Create AnnData from gene modality
gene_mod = model.get_modality('gene')
adata = anndata.AnnData(
    X=gene_mod.counts.T,  # AnnData expects (cells × genes)
    obs=model.meta,       # Cell metadata
    var=gene_mod.feature_meta  # Gene metadata
)

# Add additional layers from other modalities
spliz_mod = model.get_modality('spliz')
adata.layers['spliz'] = spliz_mod.counts.T
```

---

## Accessing Model Results

### Technical Model Results

```python
# After running fit_technical()
if hasattr(model, 'alpha_y_prefit'):
    alpha_y = model.alpha_y_prefit  # Overdispersion parameters
    print(f"Shape: {alpha_y.shape}")  # (n_trans_genes, n_cell_lines)

    # Convert to pandas for easier viewing
    alpha_df = pd.DataFrame(
        alpha_y.cpu().numpy(),
        index=model.trans_genes,
        columns=model.meta['cell_line'].unique()
    )
    print(alpha_df.head())
```

### Cis Model Results

```python
# After running fit_cis()
if hasattr(model, 'x_true'):
    x_true = model.x_true  # Cis expression per guide
    print(f"Shape: {x_true.shape}")  # (n_guides,)

    # Create DataFrame with guide info
    guide_expr = pd.DataFrame({
        'guide': model.meta.groupby('guide').first().index,
        'x_true': x_true.cpu().numpy()
    })
    print(guide_expr)
```

### Trans Model Results

```python
# After running fit_trans()
if hasattr(model, 'posterior_samples_trans'):
    posterior = model.posterior_samples_trans

    # Available parameters depend on function_type
    # For 'additive_hill':
    print(posterior.keys())
    # ['params_pos', 'params_neg', 'pi_y', ...]

    # Get positive Hill parameters
    params_pos = posterior['params_pos']  # (n_trans_genes, 3) for [B, K, EC50]
    print(f"B (magnitude): {params_pos[:, 0]}")
    print(f"K (Hill coefficient): {params_pos[:, 1]}")
    print(f"EC50: {params_pos[:, 2]}")
```

---

## Working with Posterior Samples

### Extract Point Estimates

```python
# Cis posterior
x_true_samples = model.posterior_samples_cis['x_true']  # (n_samples, n_guides)
x_true_mean = x_true_samples.mean(dim=0)  # Mean across samples
x_true_std = x_true_samples.std(dim=0)    # Std across samples

# Trans posterior (additive_hill)
params_pos = model.posterior_samples_trans['params_pos']  # (n_samples, n_genes, 3)
B_mean = params_pos[:, :, 0].mean(dim=0)  # Mean B across samples
B_std = params_pos[:, :, 0].std(dim=0)    # Std B across samples
```

### Credible Intervals

```python
import torch

# 95% credible interval for x_true
x_true_samples = model.posterior_samples_cis['x_true']
x_lower = torch.quantile(x_true_samples, 0.025, dim=0)
x_upper = torch.quantile(x_true_samples, 0.975, dim=0)

# Create summary DataFrame
x_summary = pd.DataFrame({
    'guide': model.meta.groupby('guide').first().index,
    'mean': x_true_mean.cpu().numpy(),
    'lower': x_lower.cpu().numpy(),
    'upper': x_upper.cpu().numpy()
})
```

### Save Posterior Samples

```python
import torch

# Save all posteriors
torch.save({
    'cis': model.posterior_samples_cis,
    'trans': model.posterior_samples_trans
}, 'posteriors.pt')

# Load later
posteriors = torch.load('posteriors.pt')
```

### Reconstruct Predictions

```python
from bayesDREAM import Hill_based_positive, Hill_based_negative

# For additive Hill model
x_grid = torch.linspace(
    model.x_true.min(),
    model.x_true.max(),
    100
)

params_pos = model.posterior_samples_trans['params_pos'].mean(dim=0)  # (n_genes, 3)
params_neg = model.posterior_samples_trans['params_neg'].mean(dim=0)

# Predict for first gene
gene_idx = 0
y_pos = Hill_based_positive(x_grid, params_pos[gene_idx])
y_neg = Hill_based_negative(x_grid, params_neg[gene_idx])
y_pred = y_pos + y_neg

# Plot
import matplotlib.pyplot as plt
plt.plot(x_grid.cpu(), y_pred.cpu())
plt.xlabel('Cis expression (x)')
plt.ylabel('Trans expression (y)')
plt.title(f'Dose-response: {model.trans_genes[gene_idx]}')
```

---

## Common Data Access Patterns

### Compare NTC vs Perturbed Expression

```python
# Get gene expression for NTC and perturbed cells
gene_mod = model.get_modality('gene')
gene_counts = gene_mod.counts

ntc_mask = model.meta['target'] == 'ntc'
cis_mask = model.meta['target'] == model.cis_gene

ntc_expr = gene_counts.loc[:, ntc_mask].mean(axis=1)
cis_expr = gene_counts.loc[:, cis_mask].mean(axis=1)

# Calculate log fold change
log2fc = np.log2((cis_expr + 1) / (ntc_expr + 1))

# Find top changed genes
top_genes = log2fc.abs().nlargest(20)
print(top_genes)
```

### Extract Guide-Level Summaries

```python
# Average expression per guide
guide_summary = []
for guide in model.meta['guide'].unique():
    guide_cells = model.meta[model.meta['guide'] == guide]['cell']
    guide_expr = model.counts[guide_cells].mean(axis=1)
    guide_summary.append({
        'guide': guide,
        'target': model.meta[model.meta['guide'] == guide]['target'].iloc[0],
        'n_cells': len(guide_cells),
        'mean_expression': guide_expr
    })

guide_df = pd.DataFrame(guide_summary)
```

### Access Splicing Changes

```python
# Get donor usage for NTC vs perturbed
donor_mod = model.get_modality('splicing_donor')
donor_counts = donor_mod.counts  # (donors × cells × acceptors)

ntc_mask = model.meta['target'] == 'ntc'
cis_mask = model.meta['target'] == model.cis_gene

# Average usage across cells
ntc_usage = donor_counts[:, ntc_mask, :].mean(axis=1)  # (donors × acceptors)
cis_usage = donor_counts[:, cis_mask, :].mean(axis=1)

# Calculate usage difference
usage_diff = cis_usage - ntc_usage

# Find most changed donors
max_diff = np.abs(usage_diff).max(axis=1)
top_donors = np.argsort(max_diff)[-10:]  # Top 10

# Get metadata for top donors
print(donor_mod.feature_meta.iloc[top_donors])
```

---

## Tips and Best Practices

### 1. Always Check Data Dimensions

```python
mod = model.get_modality('my_modality')
print(f"Distribution: {mod.distribution}")
print(f"Dimensions: {mod.dims}")
print(f"Counts shape: {mod.counts.shape}")
```

### 2. Handle Missing Values

```python
# Check for NaN
print(f"NaN in counts: {np.isnan(mod.counts).sum()}")

# Check for Inf
print(f"Inf in counts: {np.isinf(mod.counts).sum()}")

# Filter out features with too many zeros
zero_frac = (mod.counts == 0).mean(axis=1)
keep_features = zero_frac < 0.9  # Keep features with <90% zeros
```

### 3. Memory-Efficient Access

```python
# For large datasets, don't load all data at once
# Instead, iterate over features
gene_mod = model.get_modality('gene')
n_genes = gene_mod.dims['n_features']

for i in range(0, n_genes, 100):  # Process 100 genes at a time
    subset_counts = gene_mod.counts[i:i+100, :]
    # ... process subset ...
```

### 4. Verify Cell-Modality Alignment

```python
# Check that all modalities have same cells
cell_counts = {}
for mod_name in model.modalities.keys():
    mod = model.get_modality(mod_name)
    cell_counts[mod_name] = mod.dims['n_cells']

print(f"All modalities aligned: {len(set(cell_counts.values())) == 1}")
```

### 5. Save and Load Modalities

```python
import pickle

# Save a modality
donor_mod = model.get_modality('splicing_donor')
with open('donor_mod.pkl', 'wb') as f:
    pickle.dump(donor_mod, f)

# Load later
with open('donor_mod.pkl', 'rb') as f:
    loaded_mod = pickle.load(f)
```
