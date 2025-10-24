<p align="center">
  <img src="bayesDREAM logo.pdf" alt="bayesDREAM Logo" width="600"/>
</p>

# bayesDREAM: Bayesian Dosage Response Effects Across Modalities

A Bayesian framework for modeling perturbation effects across multiple molecular modalities using PyTorch and Pyro.

## Overview

bayesDREAM models how perturbations propagate through molecular layers using a three-step Bayesian framework:

1. **Technical fit**: Model technical variation in non-targeting controls
2. **Cis fit**: Model direct effects on targeted features (genes, ATAC regions, etc.)
3. **Trans fit**: Model downstream effects as dose-response functions

Supports multiple molecular modalities including genes, transcripts, splicing, ATAC-seq, and custom measurements.

### Key Design: Separate 'cis' Modality

bayesDREAM uses a dedicated **'cis' modality** for modeling direct perturbation effects:

**Initialization Behavior:**
- The 'cis' modality is extracted during `bayesDREAM()` initialization from your primary modality
- The primary modality contains only **trans** features (cis feature excluded)
- When you add additional modalities later (e.g., `add_atac_modality()`), NO cis extraction occurs
- All modalities are automatically subset to cells present in the 'cis' modality

**Fitting Behavior:**
- `fit_technical()` on primary modality: Includes BOTH cis and trans features for cell-line effect estimation
- `fit_cis()`: Always uses the 'cis' modality - consistent interface regardless of data type
- `fit_trans()`: Uses the primary modality (trans features only)

**Example**: If you specify `cis_gene='GFI1B'`, you get:
- **'cis' modality**: Just GFI1B (for cis modeling)
- **'gene' modality**: All other genes (for trans modeling)
- **fit_technical**: Fits all 92 genes (including GFI1B), extracts alpha_x for GFI1B, stores alpha_y for remaining 91 genes

**Parameters for Specifying Cis Feature:**
- `cis_gene`: For gene modality (e.g., `cis_gene='GFI1B'`)
- `cis_feature`: Generic parameter for any modality type (e.g., `cis_feature='chr9:132283881-132284881'` for ATAC)
- Note: `cis_gene` is an alias for `cis_feature` when `primary_modality='gene'`

## Features

- 🧬 **Gene expression modeling** with negative binomial likelihood
- 📊 **Transcript-level analysis** (isoform usage or independent counts)
- ✂️ **Splicing analysis** (donor usage, acceptor usage, exon skipping)
- 📈 **Custom modalities** (SpliZ, SpliZVD, user-defined measurements)
- 🔄 **Multiple dose-response functions** (Hill equations, polynomials)
- 🎯 **Guide-level inference** with technical variation modeling
- 🔀 **Permutation testing** for statistical significance
- 💾 **Save/load pipeline stages** with modality-specific control
- 🚀 **GPU support** via PyTorch

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/bayesDREAM.git
cd bayesDREAM

# Install dependencies
pip install torch pyro-ppl pandas numpy scipy scikit-learn matplotlib

# Install in development mode
pip install -e .
```

## Quick Start

### Basic Usage (Gene Expression)

```python
from bayesDREAM import bayesDREAM
import pandas as pd

# Load data
meta = pd.read_csv('meta.csv')
gene_counts = pd.read_csv('gene_counts.csv', index_col=0)

# Create model
model = bayesDREAM(
    meta=meta,
    counts=gene_counts,
    gene_meta=gene_meta,  # Optional: gene annotations (gene, gene_name, gene_id)
    cis_gene='GFI1B',
    output_dir='./output',
    label='my_run'
)

# Run 3-step pipeline
model.set_technical_groups(['cell_line'])
model.fit_technical(sum_factor_col='sum_factor')
model.fit_cis(sum_factor_col='sum_factor')
model.fit_trans(sum_factor_col='sum_factor_adj', function_type='additive_hill')

# Save results
model.save_technical_fit()
model.save_cis_fit()
model.save_trans_fit()
```

### Multi-Modal Analysis

```python
from bayesDREAM import bayesDREAM

# Initialize with genes (gene_meta optional but recommended)
model = bayesDREAM(
    meta=meta,
    counts=gene_counts,
    gene_meta=gene_meta,  # Optional: gene, gene_name, gene_id
    cis_gene='GFI1B'
)

# Add splicing modalities
model.add_splicing_modality(
    sj_counts=sj_counts,
    sj_meta=sj_meta,
    splicing_types=['donor', 'acceptor', 'exon_skip']
)

# Add custom modalities
model.add_custom_modality('spliz', spliz_scores, gene_meta, 'normal')

# View all modalities
print(model.list_modalities())

# Run pipeline (operates on primary gene modality)
model.set_technical_groups(['cell_line'])
model.fit_technical(sum_factor_col='sum_factor')
model.fit_cis(sum_factor_col='sum_factor')
model.fit_trans(sum_factor_col='sum_factor_adj', function_type='additive_hill')

# Access other modalities
donor_mod = model.get_modality('splicing_donor')
```

### Staged Pipeline with Save/Load

```python
# Stage 1: Technical fit
model.set_technical_groups(['cell_line'])
model.fit_technical(sum_factor_col='sum_factor')
model.save_technical_fit()

# Stage 2: Cis fit (in new session)
model2 = bayesDREAM(meta=meta, counts=gene_counts, cis_gene='GFI1B')
model2.load_technical_fit()
model2.fit_cis(sum_factor_col='sum_factor')
model2.save_cis_fit()

# Stage 3: Trans fit (in new session)
model3 = bayesDREAM(meta=meta, counts=gene_counts, cis_gene='GFI1B')
model3.load_technical_fit()
model3.load_cis_fit()
model3.fit_trans(sum_factor_col='sum_factor_adj')
model3.save_trans_fit()

# Selective modality saving (model-level saved automatically when primary modality included)
model.save_technical_fit(modalities=['gene', 'atac'])
model.load_technical_fit(modalities=['gene'])
```

See **[docs/SAVE_LOAD_GUIDE.md](docs/SAVE_LOAD_GUIDE.md)** for complete save/load documentation.


## Documentation

### User Guides
- **[docs/QUICKSTART_MULTIMODAL.md](docs/QUICKSTART_MULTIMODAL.md)** - Quick reference guide
- **[docs/PLOTTING_GUIDE.md](docs/PLOTTING_GUIDE.md)** - Comprehensive visualization guide
- **[docs/SAVE_LOAD_GUIDE.md](docs/SAVE_LOAD_GUIDE.md)** - Save/load pipeline stages with modality-specific control
- **[docs/API_REFERENCE.md](docs/API_REFERENCE.md)** - Complete API reference with all functions
- **[docs/DATA_ACCESS.md](docs/DATA_ACCESS.md)** - Guide to accessing and working with data
- **[examples/](examples/)** - Example scripts for staged pipeline execution

### Technical Documentation
- **[CLAUDE.md](CLAUDE.md)** - Complete architecture documentation for Claude Code
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System architecture and design patterns
- **[docs/archive/](docs/archive/)** - Historical design documents and refactoring summaries

## Data Requirements

### Cell Metadata
- Required columns: `cell`, `guide`, `target`, `sum_factor`, `cell_line`
- Use `'ntc'` for non-targeting control guides

### Count Matrices
- Genes: rows = genes, columns = cells
- Splice junctions: rows = junctions, columns = cells
- Transcripts: rows = transcripts, columns = cells

### Gene Metadata (Optional but Recommended)
- Rows correspond to genes in counts matrix
- Recommended columns: `gene`, `gene_name`, `gene_id`
- Can use index as gene identifier if named
- Enables gene-level annotations for downstream analysis

See [docs/QUICKSTART_MULTIMODAL.md](docs/QUICKSTART_MULTIMODAL.md) for detailed format specifications.

## Supported Distributions

| Distribution | Use Case | Data Structure |
|--------------|----------|----------------|
| `negbinom` | Gene/transcript counts | 2D: (features, cells) |
| `multinomial` | Isoform/donor/acceptor usage | 3D: (features, cells, categories) |
| `binomial` | Exon skipping PSI | 2D + denominator |
| `normal` | Continuous scores (SpliZ) | 2D: (features, cells) |
| `mvnormal` | Multivariate scores (SpliZVD) | 3D: (features, cells, dims) |

## Citation

If you use bayesDREAM in your research, please cite:

```
[Citation to be added]
```

## Development

### Running Tests

```bash
# Test multi-modal infrastructure
/opt/anaconda3/envs/pyroenv/bin/python tests/test_multimodal_fitting.py

# Test filtering
/opt/anaconda3/envs/pyroenv/bin/python tests/test_filtering_simple.py

# Test per-modality fitting
/opt/anaconda3/envs/pyroenv/bin/python tests/test_per_modality_fitting.py
```

See [tests/README.md](tests/README.md) for complete testing documentation.

### Project Structure

```
bayesDREAM/
├── bayesDREAM/           # Core package
│   ├── model.py          # Main bayesDREAM class (~311 lines)
│   ├── core.py           # _BayesDREAMCore base class (~909 lines)
│   ├── modality.py       # Modality data structure
│   ├── distributions.py  # Distribution-specific samplers
│   ├── splicing.py       # Splicing processing (pure Python)
│   ├── fitting/          # Modular fitting methods
│   │   ├── helpers.py    # Helper functions
│   │   ├── technical.py  # TechnicalFitter
│   │   ├── cis.py        # CisFitter
│   │   └── trans.py      # TransFitter
│   ├── io/               # Save/load functionality
│   │   ├── save.py       # ModelSaver
│   │   └── load.py       # ModelLoader
│   ├── modalities/       # Modality-specific methods
│   │   ├── transcript.py
│   │   ├── splicing_modality.py
│   │   ├── atac.py
│   │   └── custom.py
│   └── __init__.py       # Package exports
├── examples/             # Usage examples
├── tests/                # Test suite
├── docs/                 # Documentation
└── toydata/              # Test datasets
```

**Note**: The codebase was refactored from a single 4,537-line file into a modular structure (93% reduction in model.py). This improves maintainability while preserving complete backward compatibility. See [docs/archive/](docs/archive/) for refactoring history.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

[Add license information]

## Contact

[Add contact information]

## Acknowledgments

This work uses:
- [PyTorch](https://pytorch.org/) for tensor computations
- [Pyro](https://pyro.ai/) for probabilistic programming
