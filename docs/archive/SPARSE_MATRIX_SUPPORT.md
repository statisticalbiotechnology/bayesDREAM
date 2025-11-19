# Sparse Matrix Support in bayesDREAM

## Current Status

bayesDREAM now has **partial sparse matrix support** to reduce memory usage when working with large single-cell datasets.

### What Works ✅

1. **Modality class**: Fully supports sparse matrices
   - Accepts `scipy.sparse` matrices in `__init__`
   - Preserves sparsity during cell/feature subsetting
   - Only densifies when calling `to_tensor()` for PyTorch operations
   - Stores `is_sparse` flag to track sparsity

2. **Cis/trans modality extraction**: Works with sparse input
   - `_extract_cis_from_gene()`: Can extract from sparse matrices
   - `_extract_cis_generic()`: Can extract from sparse matrices
   - Modality objects preserve sparsity after extraction

3. **model.py creation methods**: Accept sparse input
   - `_create_gene_modality()`: Accepts sparse
   - `_create_negbinom_modality()`: Accepts sparse

### Current Limitations ⚠️

**Base class initialization (`core.py`)** still requires DataFrame, causing temporary densification:
- During `bayesDREAM.__init__()`, sparse matrices are converted to DataFrame
- This happens at line 82-90 in `model.py` via `_counts_to_dataframe()`
- Memory spike occurs during init, then dense DataFrame is stored in `self.counts`
- Fitting functions (`fit_technical`, `fit_cis`, `fit_trans`) could work with Modality objects directly but currently use `self.counts`

**Why this limitation exists**:
- `core.py` extensively uses DataFrame operations (`.index`, `.columns`, `.loc[]`, etc.)
- Full refactoring of core.py would require changing ~50+ operations
- This is a large architectural change that needs careful testing

### Memory Usage Pattern

With current implementation:
1. **User passes sparse matrix** → ✅ No densification
2. **Modality objects created** → ✅ Stored as sparse
3. **Base class init** → ⚠️ Temporary densification to DataFrame
4. **Fitting functions** → ⚠️ Work with dense `self.counts` from base class

**Recommended usage for large datasets**:
- Use sparse input to benefit from Modality-level sparsity
- Understand that base class will densify during init (one-time cost)
- Future updates will eliminate the init densification

### Example Usage

```python
from scipy import sparse
import pandas as pd
from bayesDREAM import bayesDREAM

# Load large sparse matrix (e.g., 30k genes × 100k cells)
gene_counts_sparse = sparse.load_npz('gene_counts.npz')  # Stays sparse!

# Create model - WARNING: will densify during init
model = bayesDREAM(
    meta=cell_metadata,
    counts=gene_counts_sparse,  # Accepts sparse!
    feature_meta=gene_metadata,
    cis_gene='GFI1B'
)

# Modalities are stored as sparse
cis_mod = model.get_modality('cis')
print(cis_mod.is_sparse)  # True

gene_mod = model.get_modality('gene')
print(gene_mod.is_sparse)  # True

# But self.counts is dense DataFrame (from base class)
print(type(model.counts))  # pandas.DataFrame
```

### Future Work

To achieve full sparse support:
1. Modify `core.py.__init__` to accept sparse with metadata
2. Store gene_names and cell_names separately
3. Provide sparse-aware subsetting methods
4. Update fitting functions to densify only necessary subsets
5. Remove dependency on `self.counts` DataFrame in fitting methods

### Files Modified

- `bayesDREAM/modality.py`: Added sparse detection and sparse-aware operations
  - `__init__`: Detects sparse, stores `is_sparse` flag
  - `to_tensor()`: Densifies only when converting to PyTorch
  - `get_feature_subset()`, `get_cell_subset()`: Maintain sparsity

- `bayesDREAM/model.py`: Added sparse handling in creation methods
  - `_counts_to_dataframe()`: Added `keep_sparse` option (not yet used)
  - Cis extraction methods work with sparse input

- `bayesDREAM/core.py`: Added sparse import (no functional changes yet)

## Testing

To test sparse support:

```python
from scipy import sparse
import numpy as np
import pandas as pd
from bayesDREAM import bayesDREAM

# Create sparse test data
n_genes, n_cells = 1000, 500
density = 0.1  # 10% non-zero
gene_counts_sparse = sparse.random(n_genes, n_cells, density=density, format='csr')

gene_meta = pd.DataFrame({'gene': [f'Gene{i}' for i in range(n_genes)]})
cell_meta = pd.DataFrame({
    'cell': [f'Cell{i}' for i in range(n_cells)],
    'guide': ['g1'] * (n_cells // 2) + ['ntc'] * (n_cells // 2),
    'target': ['GFI1B'] * (n_cells // 2) + ['ntc'] * (n_cells // 2),
    'sum_factor': np.ones(n_cells)
})

model = bayesDREAM(
    meta=cell_meta,
    counts=gene_counts_sparse,
    feature_meta=gene_meta,
    cis_gene='Gene0'
)

# Check modality sparsity
print(f"Cis modality sparse: {model.get_modality('cis').is_sparse}")
print(f"Gene modality sparse: {model.get_modality('gene').is_sparse}")
```
