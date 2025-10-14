# Multi-Modal bayesDREAM Implementation

## Summary

I've successfully extended bayesDREAM to support multiple molecular modalities while maintaining full backward compatibility with the existing single-modality (gene expression) workflow.

## New Files Created

### 1. `bayesDREAM/modality.py` (267 lines)
**Purpose**: Core data structure for multi-modal data

**Key Class**: `Modality`
- Stores counts/measurements with feature-level metadata
- Supports 5 distribution types: `negbinom`, `multinomial`, `binomial`, `normal`, `mvnormal`
- Handles different data structures:
  - 2D arrays for gene/transcript counts, SpliZ scores
  - 3D arrays for isoform usage, donor/acceptor usage (multinomial)
  - 3D arrays for SpliZVD (multivariate normal: z0, z1, z2)
  - Binomial with separate denominator arrays (exon skipping)
- Methods for subsetting by features or cells
- Validation of data shapes and distribution requirements

### 2. `bayesDREAM/splicing.py` (~750 lines)
**Purpose**: Process splice junction data with pure Python implementations

**Key Functions**:
- `process_sj_counts()`: Raw SJ counts with gene expression denominator → binomial modality
- `process_donor_usage()`: Compute donor site usage → 3D multinomial array
- `process_acceptor_usage()`: Compute acceptor site usage → 3D multinomial array
- `process_exon_skipping()`: Detect cassette exons → 2D binomial (inclusion/total)
- `create_splicing_modality()`: High-level function returning ready-to-use Modality objects
- `_build_sj_index()`: Builds junction index with donor/acceptor annotations
- `_find_cassette_triplets_strand()`: Finds cassette exons using strand-aware coordinates
- `_find_cassette_triplets_genomic()`: Finds cassette exons using genomic coordinates (fallback)

**No R Dependencies**:
- Pure Python/NumPy/pandas implementation
- No subprocess calls or R integration required
- Fully self-contained splicing analysis

### 3. `bayesDREAM/multimodal.py` (~530 lines)
**Purpose**: Multi-modal wrapper extending the base bayesDREAM class

**Key Class**: `MultiModalBayesDREAM` (extends `bayesDREAM`)
- **Backward compatible**: Works exactly like original bayesDREAM when only gene counts provided
- Stores modalities in `self.modalities` dictionary
- Designates one modality as "primary" for cis/trans modeling
- Automatically subsets all modalities to match filtered cells

**Key Methods**:
- `add_modality()`: Add any pre-constructed Modality
- `add_transcript_modality()`: Add transcripts as counts and/or usage (both negbinom AND multinomial in one call)
- `add_splicing_modality()`: Add multiple splicing types (sj/donor/acceptor/exon_skip) in one call
- `add_custom_modality()`: Add user-defined modalities (SpliZ, SpliZVD, etc.)
- `get_modality()`: Retrieve a specific modality
- `list_modalities()`: Summary table of all modalities

### 4. `bayesDREAM/distributions.py` (~700 lines) **NEW**
**Purpose**: Distribution-specific observation models for multi-modal fitting

**Key Components**:
- Observation samplers for 5 distributions: `negbinom`, `multinomial`, `binomial`, `normal`, `mvnormal`
- Separate implementations for `technical` and `trans` models
- `DISTRIBUTION_REGISTRY`: Maps distribution names to sampler functions
- Helper functions: `get_observation_sampler()`, `requires_denominator()`, `is_3d_distribution()`

**Design Philosophy**:
- Function types (Hill, polynomial) are **shared** across modalities
- Observation likelihoods are **distribution-specific**
- Ready for integration into `_model_technical` and `_model_y` Pyro models

### 5. `bayesDREAM/__init__.py` (updated)
**Purpose**: Package exports for easy importing

Exports:
- `bayesDREAM` (original class)
- `MultiModalBayesDREAM` (new multi-modal wrapper)
- `Modality` (data structure)
- Splicing functions
- Distribution registry and helpers

### 6. `examples/multimodal_example.py` (398 lines)
**Purpose**: Comprehensive usage examples

Includes 6 complete examples:
1. Gene counts only (backward compatibility)
2. Genes + transcripts (isoform usage)
3. Genes + splicing (donor/acceptor/exon skipping)
4. Custom modalities (SpliZ, SpliZVD)
5. Pre-constructed modalities
6. Subsetting and filtering

### 7. `CLAUDE.md` (updated)
**Purpose**: Documentation for future Claude Code instances

Added comprehensive "Multi-Modal Architecture" section covering:
- Modality class and supported distributions
- MultiModalBayesDREAM usage
- Splicing processing details
- Current limitations
- Example workflows

## Architecture Design

### Data Flow

```
Input Data
    ├── Gene counts (2D DataFrame)
    ├── Transcript counts (2D DataFrame)
    ├── SJ counts (2D DataFrame) + SJ metadata
    ├── SpliZ scores (2D DataFrame)
    └── SpliZVD (3 separate 2D DataFrames)
                    ↓
        Create Modality objects
                    ↓
    ┌─────────────────────────────────┐
    │  MultiModalBayesDREAM           │
    │                                 │
    │  modalities = {                 │
    │    'gene': Modality(negbinom)   │ ← Primary modality
    │    'transcript': Modality(...)  │
    │    'splicing_donor': ...        │
    │    'splicing_acceptor': ...     │
    │    'splicing_exon_skip': ...    │
    │    'spliz': Modality(normal)    │
    │    'splizvd': Modality(mvnorm)  │
    │  }                              │
    └─────────────────────────────────┘
                    ↓
        fit_technical() ─────────────→ Uses primary modality
        fit_cis() ───────────────────→ Uses primary modality
        fit_trans() ─────────────────→ Uses primary modality
                    ↓
    Access other modalities for downstream analysis
```

### Distribution to Data Structure Mapping

| Distribution | Data Shape | Example Use Case | Denominator |
|--------------|-----------|------------------|-------------|
| `negbinom` | 2D: (features, cells) | Gene counts, transcript counts | No |
| `multinomial` | 3D: (features, cells, categories) | Isoform usage, donor usage | No |
| `binomial` | 2D: (features, cells) | Raw SJ counts, Exon skipping PSI | Yes (2D) |
| `normal` | 2D: (features, cells) | SpliZ scores | No |
| `mvnormal` | 3D: (features, cells, dims) | SpliZVD (z0, z1, z2) | No |

### Splicing Data Structures

**Raw SJ Counts** (binomial):
```python
# Shape: (n_junctions, n_cells)
# Numerator: SJ read counts
# Denominator: Gene expression (matched by gene annotation)
# Example: sj_counts[10, 5] = junction 10 reads in cell 5
#          gene_denom[10, 5] = gene expression for junction 10's gene in cell 5

feature_meta:
  - All SJ metadata columns (coord.intron, chrom, strand, positions, etc.)
  - gene: Assigned gene identifier (from gene_name_start/end or gene_id_start/end)
```

**Donor Usage** (multinomial):
```python
# Shape: (n_donors, n_cells, max_acceptors)
# Example: donor_counts[5, 10, 2] = count of cells using donor 5 → acceptor 2 in cell 10

feature_meta:
  - chrom, strand, donor (position)
  - acceptors: list of acceptor positions
  - n_acceptors: number of alternative acceptors
```

**Acceptor Usage** (multinomial):
```python
# Shape: (n_acceptors, n_cells, max_donors)
# Example: acceptor_counts[3, 15, 1] = count of cells using donor 1 → acceptor 3 in cell 15

feature_meta:
  - chrom, strand, acceptor (position)
  - donors: list of donor positions
  - n_donors: number of alternative donors
```

**Exon Skipping** (binomial):
```python
# Shape: (n_events, n_cells) for both inclusion and total
# Example: inclusion[2, 8] = inclusion reads for event 2 in cell 8
#          total[2, 8] = total reads (inclusion + skipping) for event 2 in cell 8

feature_meta:
  - trip_id, chrom, strand
  - d1, a2, d2, a3: coordinates of triplet (donor1, acceptor2, donor2, acceptor3)
  - sj_inc1, sj_inc2, sj_skip: junction IDs
```

## Key Design Decisions

### 1. Inheritance vs Composition
**Decision**: `MultiModalBayesDREAM` extends `bayesDREAM` via inheritance

**Rationale**:
- Full backward compatibility: users can still use `bayesDREAM` directly
- `MultiModalBayesDREAM` "is-a" `bayesDREAM` with additional modality support
- All existing methods (fit_technical, fit_cis, fit_trans) work unchanged
- Primary modality transparently passed to parent class

### 2. Modality as Separate Class
**Decision**: Create standalone `Modality` class rather than nested dicts

**Rationale**:
- Type safety and validation at construction time
- Reusable across different contexts
- Clear API for subsetting and manipulation
- Extensible for future features (e.g., modality-specific transformations)

### 3. Pure Python Splicing Implementation
**Decision**: Implement splicing analysis entirely in Python (NumPy/pandas)

**Rationale**:
- No R dependency simplifies installation and deployment
- Better performance (no subprocess overhead)
- Easier to debug and maintain
- Full control over data flow and validation
- Can leverage Python's rich data science ecosystem

### 4. Distribution-Aware Storage
**Decision**: Store distribution type and validate data shapes

**Rationale**:
- Different distributions require different data structures
- Early validation prevents runtime errors
- Enables future development of distribution-specific models
- Documents expected data format

### 5. Primary Modality Concept
**Decision**: Designate one modality as "primary" for cis/trans analysis

**Rationale**:
- Current statistical models (model_x, model_y) designed for single modality
- Allows incremental enhancement: store multi-modal data now, model later
- Clear which modality drives perturbation effects
- Future: can extend to multi-modal joint models

## Current Status & Implementation Roadmap

### ✅ Completed (v0.2.0+)

1. **Multi-modal data storage**: All modalities can be added and stored in a unified framework
2. **Distribution-specific observation models**: Implemented in `distributions.py` for all 5 distribution types
3. **Backward compatibility**: `MultiModalBayesDREAM` works exactly like `bayesDREAM` for gene expression
4. **Cell alignment**: Automatic subsetting ensures all modalities have consistent cell sets
5. **Flexible gene identifiers**: Support for gene names, gene_name, and gene_id (Ensembl IDs)
6. **Distribution-flexible fitting** (MAJOR UPDATE):
   - ✅ `_model_technical` now accepts `distribution` parameter and uses appropriate sampler from `distributions.py`
   - ✅ `_model_y` now accepts `distribution` parameter and uses appropriate sampler
   - ✅ `fit_technical()` supports all distributions with validation
   - ✅ `fit_trans()` supports all distributions with validation
   - ✅ Parameter handling for different distributions:
     - Negative binomial: overdispersion (`phi_y`), sum factors
     - Multinomial: no overdispersion, 3D data (K categories)
     - Binomial: denominator array, 2D data
     - Normal: variance parameter (`sigma_y`), no sum factors
     - Multivariate normal: covariance matrix, 3D data (D dimensions)
   - ✅ Cell-line covariate effects:
     - Negative binomial: multiplicative on mu
     - Normal/MVNormal: additive on mu
     - Binomial: logit-scale effects
     - Multinomial: not yet supported (complex)
7. **Backward compatibility tests**: test_negbinom_compat.py, test_technical_compat.py verify negbinom still works

### 🚧 In Progress (Next Steps)

1. **Modality-specific preprocessing**: Handle modality-specific transformations before fitting

### 🔮 Future Enhancements

1. **Cross-modality modeling**:
   - Model transcript usage as function of gene expression
   - Model splicing changes as function of cis perturbation
   - Joint models across modalities

2. **Normalization**:
   - Implement modality-specific normalization (currently only gene counts use sum factors)
   - Support for total-count normalization (multinomial)
   - VST or arcsinh transforms for continuous data

3. **Performance**:
   - Option to cache splicing metrics
   - Parallel processing of multiple modalities
   - Optimize cassette exon detection for large datasets

## Usage Examples

### Minimal Example (Backward Compatible)
```python
from bayesDREAM import MultiModalBayesDREAM

# Works exactly like original bayesDREAM (negbinom is default)
model = MultiModalBayesDREAM(
    meta=meta,
    counts=gene_counts,
    cis_gene='GFI1B'
)
model.fit_technical(covariates=['cell_line'], sum_factor_col='sum_factor')
model.fit_cis(sum_factor_col='sum_factor')
model.fit_trans(sum_factor_col='sum_factor_adj', function_type='additive_hill')
```

### Full Multi-Modal Example
```python
from bayesDREAM import MultiModalBayesDREAM

# Initialize with gene counts
model = MultiModalBayesDREAM(
    meta=meta,
    counts=gene_counts,
    cis_gene='GFI1B',
    primary_modality='gene'
)

# Add transcripts as both counts and usage
model.add_transcript_modality(
    transcript_counts=tx_counts,
    transcript_meta=tx_meta,
    modality_types=['counts', 'usage']
)

# Add all splicing modalities (including raw SJ counts)
model.add_splicing_modality(
    sj_counts=sj_counts,
    sj_meta=sj_meta,
    splicing_types=['sj', 'donor', 'acceptor', 'exon_skip']
)

# Add SpliZ and SpliZVD
model.add_custom_modality('spliz', spliz_scores, gene_meta, 'normal')
model.add_custom_modality('splizvd', splizvd_3d, gene_meta, 'mvnormal')

# View all modalities
print(model.list_modalities())

# Run pipeline on primary modality (gene counts = negbinom)
model.fit_technical(covariates=['cell_line'], sum_factor_col='sum_factor', distribution='negbinom')
model.fit_cis(sum_factor_col='sum_factor')
model.fit_trans(sum_factor_col='sum_factor_adj', function_type='additive_hill', distribution='negbinom')

# Access other modalities for analysis
donor_mod = model.get_modality('splicing_donor')
donor_tensor = donor_mod.to_tensor(device=model.device)
```

### Distribution-Flexible Fitting Example
```python
from bayesDREAM import MultiModalBayesDREAM

# For continuous measurements (SpliZ scores - normal distribution)
model = MultiModalBayesDREAM(meta=meta, counts=spliz_scores, cis_gene='GFI1B')
model.fit_technical(covariates=['cell_line'], distribution='normal')
model.fit_trans(distribution='normal', function_type='polynomial')

# For binomial data (exon skipping PSI)
model = MultiModalBayesDREAM(meta=meta, counts=inclusion_counts, cis_gene='GFI1B')
model.fit_trans(
    distribution='binomial',
    denominator=total_counts,  # inclusion + skipping
    function_type='single_hill'
)

# For multinomial data (donor usage)
model = MultiModalBayesDREAM(meta=meta, counts=donor_usage_3d, cis_gene='GFI1B')
model.fit_trans(distribution='multinomial', function_type='additive_hill')
```

## Testing Recommendations

1. **Unit tests** for Modality class:
   - Test each distribution type
   - Validate shape checking
   - Test subsetting operations

2. **Integration tests** for splicing:
   - Test R function calls with toy data
   - Verify array shapes and metadata
   - Test with missing junctions

3. **End-to-end tests**:
   - Run full pipeline with toydata
   - Compare MultiModalBayesDREAM vs bayesDREAM (should be identical for gene-only)
   - Test all add_*_modality methods

4. **Documentation tests**:
   - Run all examples in multimodal_example.py
   - Verify code snippets in CLAUDE.md

## Migration Guide for Existing Code

**Old code (still works)**:
```python
from bayesDREAM.model import bayesDREAM
model = bayesDREAM(meta=meta, counts=gene_counts, cis_gene='GFI1B')
```

**New code (same functionality)**:
```python
from bayesDREAM import MultiModalBayesDREAM
model = MultiModalBayesDREAM(meta=meta, counts=gene_counts, cis_gene='GFI1B')
```

**New code (with additional modalities)**:
```python
from bayesDREAM import MultiModalBayesDREAM
model = MultiModalBayesDREAM(meta=meta, counts=gene_counts, cis_gene='GFI1B')
model.add_splicing_modality(
    sj_counts=sj_counts,
    sj_meta=sj_meta,
    splicing_types=['sj', 'donor', 'acceptor']
)
```

All existing pipeline scripts (run_technical.py, run_cis.py, run_trans.py) continue to work unchanged with the base `bayesDREAM` class. New scripts can optionally use `MultiModalBayesDREAM` for multi-modal data storage and future multi-modal modeling.
