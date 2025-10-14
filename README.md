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

- **[QUICKSTART_MULTIMODAL.md](QUICKSTART_MULTIMODAL.md)** - Quick reference guide
- **[CLAUDE.md](CLAUDE.md)** - Complete architecture documentation
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Visual architecture diagrams
- **[MULTIMODAL_IMPLEMENTATION.md](MULTIMODAL_IMPLEMENTATION.md)** - Implementation details
- **[MULTIMODAL_FITTING_INFRASTRUCTURE.md](MULTIMODAL_FITTING_INFRASTRUCTURE.md)** - Distribution-specific fitting
- **[examples/multimodal_example.py](examples/multimodal_example.py)** - Complete examples

## Data Requirements

### Cell Metadata
- Required columns: `cell`, `guide`, `target`, `sum_factor`, `cell_line`
- Use `'ntc'` for non-targeting control guides

### Count Matrices
- Genes: rows = genes, columns = cells
- Splice junctions: rows = junctions, columns = cells
- Transcripts: rows = transcripts, columns = cells

See [QUICKSTART_MULTIMODAL.md](QUICKSTART_MULTIMODAL.md) for detailed format specifications.

## Distribution-Flexible Fitting

bayesDREAM supports multiple statistical distributions for different data types:

```python
# Fit with different distributions
# For negbinom (gene counts)
model.fit_technical(covariates=['cell_line'], sum_factor_col='sum_factor', distribution='negbinom')
model.fit_trans(sum_factor_col='sum_factor', distribution='negbinom', function_type='additive_hill')

# For normal (continuous measurements like SpliZ)
model.fit_technical(covariates=['cell_line'], distribution='normal')
model.fit_trans(distribution='normal', function_type='polynomial')

# For binomial (exon skipping PSI)
model.fit_trans(distribution='binomial', denominator=total_counts, function_type='single_hill')
```

See **[MULTIMODAL_FITTING_INFRASTRUCTURE.md](MULTIMODAL_FITTING_INFRASTRUCTURE.md)** for complete details on distribution-specific fitting.

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
/opt/anaconda3/envs/pyroenv/bin/python test_multimodal_fitting.py

# Test trans model backward compatibility
/opt/anaconda3/envs/pyroenv/bin/python test_negbinom_compat.py

# Test technical model backward compatibility
/opt/anaconda3/envs/pyroenv/bin/python test_technical_compat.py
```

### Project Structure

```
bayesDREAM_forClaude/
├── bayesDREAM/           # Core package
│   ├── model.py          # Base bayesDREAM class (~2500 lines)
│   ├── multimodal.py     # Multi-modal wrapper
│   ├── modality.py       # Modality data structure
│   ├── distributions.py  # Distribution-specific observation samplers
│   ├── splicing.py       # Splicing processing (pure Python)
│   └── utils.py          # Helper functions (Hill, polynomial, etc.)
├── examples/             # Usage examples
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
