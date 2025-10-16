# bayesDREAM: Bayesian Dosage Response Effects Across Modalities

A Bayesian framework for modeling CRISPR perturbation effects across multiple molecular modalities using PyTorch and Pyro.

## Overview

bayesDREAM models how CRISPR perturbations propagate through molecular layers using a three-step Bayesian framework:

1. **Technical fit**: Model technical variation in non-targeting controls
2. **Cis fit**: Model direct effects on targeted genes
3. **Trans fit**: Model downstream effects as dose-response functions

**New in v0.2.0**: Multi-modal support for transcripts, splicing, and custom measurements!

## Features

- 🧬 **Gene expression modeling** with negative binomial likelihood
- 📊 **Transcript-level analysis** (isoform usage or independent counts)
- ✂️ **Splicing analysis** (donor usage, acceptor usage, exon skipping)
- 📈 **Custom modalities** (SpliZ, SpliZVD, user-defined measurements)
- 🔄 **Multiple dose-response functions** (Hill equations, polynomials)
- 🎯 **Guide-level inference** with technical variation modeling
- 🔀 **Permutation testing** for statistical significance
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

### Single Modality (Gene Expression)

```python
from bayesDREAM import MultiModalBayesDREAM
import pandas as pd

# Load data
meta = pd.read_csv('meta.csv')
gene_counts = pd.read_csv('gene_counts.csv', index_col=0)

# Create model
model = MultiModalBayesDREAM(
    meta=meta,
    counts=gene_counts,
    cis_gene='GFI1B',
    output_dir='./output',
    label='my_run'
)

# Run 3-step pipeline
model.fit_technical(covariates=['cell_line'], sum_factor_col='sum_factor')
model.fit_cis(sum_factor_col='sum_factor')
model.fit_trans(sum_factor_col='sum_factor_adj', function_type='additive_hill')
```

### Multi-Modal Analysis

```python
from bayesDREAM import MultiModalBayesDREAM

# Initialize with genes
model = MultiModalBayesDREAM(meta=meta, counts=gene_counts, cis_gene='GFI1B')

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
model.fit_technical(covariates=['cell_line'], sum_factor_col='sum_factor')
model.fit_cis(sum_factor_col='sum_factor')
model.fit_trans(sum_factor_col='sum_factor_adj', function_type='additive_hill')

# Access other modalities
donor_mod = model.get_modality('splicing_donor')
```

## Documentation

### User Guides
- **[docs/QUICKSTART_MULTIMODAL.md](docs/QUICKSTART_MULTIMODAL.md)** - Quick reference guide
- **[docs/API_REFERENCE.md](docs/API_REFERENCE.md)** - Complete API reference with all functions
- **[docs/DATA_ACCESS.md](docs/DATA_ACCESS.md)** - Guide to accessing and working with data
- **[examples/multimodal_example.py](examples/multimodal_example.py)** - Complete examples

### Technical Documentation
- **[CLAUDE.md](CLAUDE.md)** - Complete architecture documentation for Claude Code
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System architecture and design patterns
- **[docs/PER_MODALITY_FITTING_PLAN.md](docs/PER_MODALITY_FITTING_PLAN.md)** - Per-modality fitting implementation

## Data Requirements

### Cell Metadata
- Required columns: `cell`, `guide`, `target`, `sum_factor`, `cell_line`
- Use `'ntc'` for non-targeting control guides

### Count Matrices
- Genes: rows = genes, columns = cells
- Splice junctions: rows = junctions, columns = cells
- Transcripts: rows = transcripts, columns = cells

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
│   ├── model.py          # Base bayesDREAM class (~2250 lines)
│   ├── multimodal.py     # Multi-modal wrapper
│   ├── modality.py       # Modality data structure
│   ├── distributions.py  # Distribution-specific samplers
│   ├── splicing.py       # Splicing processing (pure Python)
│   └── __init__.py       # Package exports
├── examples/             # Usage examples
├── tests/                # Test suite
├── docs/                 # Documentation
└── toydata/              # Test datasets
```

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
