# Multi-Modal bayesDREAM Quick Start

## Installation

```bash
# Navigate to repository
cd bayesDREAM_forClaude

# Install in development mode
pip install -e .

# Ensure R dependencies are installed
R -e "install.packages('data.table')"
```

## Basic Usage

### 1. Gene Expression Only (Original Workflow)

```python
from bayesDREAM import MultiModalBayesDREAM
import pandas as pd

# Load data
meta = pd.read_csv('meta.csv')
gene_counts = pd.read_csv('gene_counts.csv', index_col=0)

# Create model (exactly like original bayesDREAM)
model = MultiModalBayesDREAM(
    meta=meta,
    counts=gene_counts,
    cis_gene='GFI1B',
    output_dir='./output',
    label='my_run'
)

# Run 3-step pipeline
model.fit_technical(covariates=['cell_line'])
model.fit_cis(sum_factor_col='sum_factor')
model.fit_trans(sum_factor_col='sum_factor_adj', function_type='additive_hill')
```

### 2. Adding Splicing Data

```python
from bayesDREAM import MultiModalBayesDREAM
import pandas as pd

# Load data
meta = pd.read_csv('meta.csv')
gene_counts = pd.read_csv('gene_counts.csv', index_col=0)
sj_counts = pd.read_csv('SJ_counts.csv', index_col=0)
sj_meta = pd.read_csv('SJ_meta.csv')

# Create model
model = MultiModalBayesDREAM(
    meta=meta,
    counts=gene_counts,
    cis_gene='GFI1B',
    output_dir='./output',
    label='with_splicing'
)

# Add splicing modalities
model.add_splicing_modality(
    sj_counts=sj_counts,
    sj_meta=sj_meta,
    splicing_types=['sj', 'donor', 'acceptor', 'exon_skip'],
    min_cell_total=1,
    min_total_exon=2
)

# Check what was added
print(model.list_modalities())
# Output:
#   modality           distribution  n_features  n_cells  n_categories  is_primary
# 0 gene               negbinom      1000        500      NaN           True
# 1 splicing_sj        binomial      200         500      NaN           False
# 2 splicing_donor     multinomial   50          500      8.0           False
# 3 splicing_acceptor  multinomial   45          500      6.0           False
# 4 splicing_exon_skip binomial      20          500      NaN           False

# Run pipeline (operates on gene modality)
model.fit_technical(covariates=['cell_line'])
model.fit_cis()
model.fit_trans(function_type='additive_hill')

# Access splicing data
donor_mod = model.get_modality('splicing_donor')
print(f"Donor sites: {donor_mod.dims['n_features']}")
print(f"Metadata columns: {donor_mod.feature_meta.columns.tolist()}")
print(donor_mod.feature_meta.head())
```

### 3. Adding Transcript Data

```python
# Add transcript counts as independent features (negative binomial)
model.add_transcript_modality(
    transcript_counts=tx_counts,  # transcripts × cells
    transcript_meta=tx_meta,      # must have: transcript_id, gene
    use_isoform_usage=False,
    name='transcript_counts'
)

# OR: Add as isoform usage (multinomial - proportions within gene)
model.add_transcript_modality(
    transcript_counts=tx_counts,
    transcript_meta=tx_meta,
    use_isoform_usage=True,
    name='isoform_usage'
)
```

### 4. Adding Custom Modalities

```python
# SpliZ scores (normal distribution)
spliz_scores = pd.read_csv('spliz_scores.csv', index_col=0)
gene_meta = pd.DataFrame({'gene': spliz_scores.index})

model.add_custom_modality(
    name='spliz',
    counts=spliz_scores,
    feature_meta=gene_meta,
    distribution='normal'
)

# SpliZVD (multivariate normal - 3D)
import numpy as np
z0 = pd.read_csv('splizvd_z0.csv', index_col=0)
z1 = pd.read_csv('splizvd_z1.csv', index_col=0)
z2 = pd.read_csv('splizvd_z2.csv', index_col=0)

# Stack into 3D array: (genes, cells, 3)
splizvd_array = np.stack([z0.values, z1.values, z2.values], axis=2)
gene_meta = pd.DataFrame({'gene': z0.index})

model.add_custom_modality(
    name='splizvd',
    counts=splizvd_array,
    feature_meta=gene_meta,
    distribution='mvnormal'
)
```

## Required Data Formats

### Cell Metadata (`meta.csv`)
Required columns:
- `cell`: Cell barcode/ID
- `guide`: Guide RNA ID
- `target`: Target gene (use 'ntc' for non-targeting controls)
- `sum_factor`: Size factor for normalization
- `cell_line`: Cell line (or other covariates)

Optional columns:
- `lane`: Sequencing lane (for batch effects)
- Any other covariates for `guide_covariates`

### Gene Counts (`gene_counts.csv`)
- Rows: Genes
- Columns: Cell barcodes (matching `meta.cell`)
- Values: Raw counts (integers)

### Splice Junction Metadata (`SJ_meta.csv`)
Required columns:
- `coord.intron`: Junction ID (e.g., "chr1:12345:67890:+")
- `chrom`: Chromosome
- `intron_start`: Junction start coordinate
- `intron_end`: Junction end coordinate
- `strand`: Strand ('+', '-', 1, or 2)
- `gene_name_start`: Gene name at start of junction
- `gene_name_end`: Gene name at end of junction

Optional columns (for Ensembl ID support):
- `gene_id_start`: Ensembl gene ID at start of junction
- `gene_id_end`: Ensembl gene ID at end of junction

### Splice Junction Counts (`SJ_counts.csv`)
- Rows: Junction IDs (matching `SJ_meta.coord.intron`)
- Columns: Cell barcodes (matching `meta.cell`)
- Values: Read counts

### Transcript Metadata (`tx_meta.csv`)
Required columns:
- `transcript_id`: Transcript ID (matching row names in transcript counts)
- `gene`: Parent gene name

## Working with Modalities

### List All Modalities
```python
df = model.list_modalities()
print(df)
```

### Access Specific Modality
```python
mod = model.get_modality('splicing_donor')
print(mod)  # Modality(name='splicing_donor', distribution='multinomial', ...)

# Get counts as numpy array
counts = mod.counts  # 3D array: (donors, cells, acceptors)

# Get counts as torch tensor
tensor = mod.to_tensor(device=model.device)

# Get feature metadata
meta = mod.feature_meta
print(meta.columns)  # ['chrom', 'strand', 'donor', 'acceptors', 'n_acceptors', ...]
```

### Subset Modality
```python
# Subset to specific features
subset = mod.get_feature_subset(['feature1', 'feature2', 'feature3'])

# Subset to specific cells
subset = mod.get_cell_subset(['cell1', 'cell2', 'cell3'])
```

## Understanding Distributions

| Distribution | Use Case | Data Shape | Example |
|--------------|----------|-----------|---------|
| `negbinom` | Count data | 2D: (features, cells) | Gene counts, transcript counts |
| `multinomial` | Proportional usage | 3D: (features, cells, categories) | Donor usage, isoform usage |
| `binomial` | Binary outcomes | 2D: (features, cells) + denominator | SJ counts (with gene denominator), Exon skipping PSI |
| `normal` | Continuous scores | 2D: (features, cells) | SpliZ scores |
| `mvnormal` | Multivariate continuous | 3D: (features, cells, dims) | SpliZVD (z0, z1, z2) |

### Splicing Modality Types

| Type | Description | Distribution | Output |
|------|-------------|--------------|--------|
| `sj` | Raw splice junction counts | binomial | SJ reads / gene expression (per-junction) |
| `donor` | Donor usage | multinomial | Which acceptor for each donor (5'SS) |
| `acceptor` | Acceptor usage | multinomial | Which donor for each acceptor (3'SS) |
| `exon_skip` | Exon skipping | binomial | Inclusion reads / total reads (cassette exons) |

## Common Workflows

### Workflow 1: Genes + Splicing
```python
# 1. Create model with genes
model = MultiModalBayesDREAM(meta=meta, counts=gene_counts, cis_gene='GFI1B')

# 2. Add splicing (all types including raw SJ counts)
model.add_splicing_modality(
    sj_counts=sj_counts,
    sj_meta=sj_meta,
    splicing_types=['sj', 'donor', 'acceptor', 'exon_skip']
)

# 3. Run pipeline
model.fit_technical(covariates=['cell_line'])
model.fit_cis()
model.fit_trans(function_type='additive_hill')

# 4. Analyze splicing
sj_mod = model.get_modality('splicing_sj')  # Raw SJ counts
donor_mod = model.get_modality('splicing_donor')  # Donor usage
# ... downstream analysis ...
```

### Workflow 2: Genes + Transcripts + Splicing
```python
# 1. Create model
model = MultiModalBayesDREAM(meta=meta, counts=gene_counts, cis_gene='GFI1B')

# 2. Add transcripts
model.add_transcript_modality(tx_counts, tx_meta, use_isoform_usage=True)

# 3. Add splicing
model.add_splicing_modality(sj_counts, sj_meta, ['exon_skip'])

# 4. Run pipeline
model.fit_technical(covariates=['cell_line'])
model.fit_cis()
model.fit_trans(function_type='polynomial')

# 5. Compare modalities
print(model.list_modalities())
gene_mod = model.get_modality('gene')
tx_mod = model.get_modality('isoform_usage')
exon_mod = model.get_modality('splicing_exon_skip')
```

### Workflow 3: Custom Modalities Only
```python
# Create modalities manually
from bayesDREAM import Modality

gene_mod = Modality(
    name='gene',
    counts=gene_counts,
    feature_meta=pd.DataFrame({'gene': gene_counts.index}),
    distribution='negbinom',
    cells_axis=1
)

spliz_mod = Modality(
    name='spliz',
    counts=spliz_scores,
    feature_meta=pd.DataFrame({'gene': spliz_scores.index}),
    distribution='normal',
    cells_axis=1
)

# Initialize with pre-built modalities
model = MultiModalBayesDREAM(
    meta=meta,
    modalities={'gene': gene_mod, 'spliz': spliz_mod},
    cis_gene='GFI1B',
    primary_modality='gene'
)
```

## Troubleshooting

### Issue: "R script failed"
**Solution**: Ensure R and data.table are installed:
```bash
R -e "install.packages('data.table')"
```

### Issue: "Cannot subset by name: feature_names not available"
**Solution**: Provide a DataFrame instead of numpy array, or create Modality with explicit feature names

### Issue: "Modality 'xyz' not found"
**Solution**: Check `model.list_modalities()` to see available modalities

### Issue: "feature_meta has X rows but counts has Y features"
**Solution**: Ensure feature_meta index/length matches the feature dimension of counts

### Issue: "binomial distribution requires denominator"
**Solution**: For binomial (exon skipping), provide both counts and denominator arrays

## Examples

See `examples/multimodal_example.py` for complete working examples covering:
1. Gene-only (backward compatible)
2. Genes + transcripts
3. Genes + splicing
4. Custom modalities (SpliZ, SpliZVD)
5. Pre-constructed modalities
6. Subsetting operations

## Next Steps

- **Current**: Store multiple modalities, model genes only
- **Future**: Develop modality-specific probabilistic models
- **Future**: Cross-modality joint modeling
- **Future**: Modality-specific normalization strategies

For more details, see:
- `MULTIMODAL_IMPLEMENTATION.md`: Technical implementation details
- `CLAUDE.md`: Full architecture documentation
- `examples/multimodal_example.py`: Complete code examples
