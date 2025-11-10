# Cell Names with Numpy Arrays

When adding modalities to bayesDREAM, you can provide count data as either:
- **pandas DataFrame**: Cell names are automatically extracted from column names
- **numpy array**: Cell names must be explicitly provided via the `cell_names` parameter

## Why Cell Names Matter

Cell names are used for:
- **Cell subsetting**: Match cells across modalities during initialization
- **Cell alignment**: Ensure consistency when adding multiple modalities
- **Debugging**: Identify cells when troubleshooting
- **Export**: Include cell identifiers in summary CSVs

## Usage

### Option 1: DataFrame (Automatic)

When using a DataFrame, cell names are automatically extracted from columns:

```python
import pandas as pd
import numpy as np

# Create DataFrame with cell names as columns
counts_df = pd.DataFrame(
    np.random.randn(100, 50),  # 100 features, 50 cells
    index=[f'feature_{i}' for i in range(100)],
    columns=[f'cell_{i}' for i in range(50)]  # Cell names
)

model.add_custom_modality(
    name='my_modality',
    counts=counts_df,  # Cell names extracted from columns
    feature_meta=feature_meta,
    distribution='normal'
)
```

### Option 2: Numpy Array (Explicit)

When using a numpy array, provide cell names explicitly:

```python
import numpy as np

# Cell names (must match number of cells in array)
cell_names = [f'cell_{i}' for i in range(50)]

# Create numpy array
counts_array = np.random.randn(100, 50)  # 100 features, 50 cells

model.add_custom_modality(
    name='my_modality',
    counts=counts_array,
    feature_meta=feature_meta,
    distribution='normal',
    cell_names=cell_names  # Explicitly provide cell names
)
```

### Option 3: Numpy Array (No Cell Names)

Cell names are optional - modalities will work without them:

```python
counts_array = np.random.randn(100, 50)

model.add_custom_modality(
    name='my_modality',
    counts=counts_array,
    feature_meta=feature_meta,
    distribution='normal'
    # No cell_names - will be None
)
```

**Note**: Without cell names, cell subsetting by name will not work, but subsetting by index will still function.

## Supported Methods

The `cell_names` parameter is available in:

- `add_custom_modality()` ✅
- `add_atac_modality()` ✅
- `add_transcript_modality()` ✅
- `add_splicing_modality()` ✅

## Cell Alignment

When initializing bayesDREAM, all modalities are automatically subset to match the cells in the 'cis' modality:

```python
# Initialize with DataFrame (automatic cell names)
model = bayesDREAM(
    meta=meta,  # Must have 'cell' column matching count columns
    counts=gene_counts_df,  # Columns = cell names
    cis_gene='GFI1B'
)

# Add numpy array with matching cell names
model.add_custom_modality(
    name='custom',
    counts=custom_array,
    feature_meta=feature_meta,
    distribution='normal',
    cell_names=gene_counts_df.columns.tolist()  # Must match
)
```

bayesDREAM will automatically:
1. Identify which cells are in both modalities
2. Subset to the intersection of cells
3. Maintain consistent cell order

## Cell Subsetting

Cell names enable subsetting by name:

```python
# Get modality
mod = model.get_modality('my_modality')

# Subset by cell names
subset_cells = ['cell_0', 'cell_5', 'cell_10']
subset_mod = mod.get_cell_subset(subset_cells)

print(f"Original: {mod.dims['n_cells']} cells")
print(f"Subset: {subset_mod.dims['n_cells']} cells")
print(f"Subset cell_names: {subset_mod.cell_names}")
```

Without cell names, you can still subset by index:

```python
# Subset by index (works even without cell_names)
subset_indices = [0, 5, 10]
subset_mod = mod.get_cell_subset(subset_indices)
```

## Example: Multi-Modal with Numpy Arrays

```python
from bayesDREAM import bayesDREAM
import numpy as np
import pandas as pd

# Define cell names
n_cells = 100
cell_names = [f'cell_{i}' for i in range(n_cells)]

# Cell metadata
meta = pd.DataFrame({
    'cell': cell_names,
    'guide': np.random.choice(['guide_1', 'guide_2', 'guide_3'], n_cells),
    'target': ['GFI1B'] * 60 + ['ntc'] * 40,
    'sum_factor': np.random.lognormal(0, 0.2, n_cells)
})

# Gene counts (DataFrame)
gene_counts = pd.DataFrame(
    np.random.negative_binomial(10, 0.5, (50, n_cells)),
    columns=cell_names
)

# Initialize model
model = bayesDREAM(
    meta=meta,
    counts=gene_counts,
    cis_gene='GFI1B'
)

# Add custom modality #1: SpliZ scores (numpy array)
spliz_array = np.random.randn(50, n_cells)
spliz_meta = pd.DataFrame({'gene': [f'gene_{i}' for i in range(50)]})

model.add_custom_modality(
    name='spliz',
    counts=spliz_array,
    feature_meta=spliz_meta,
    distribution='normal',
    cell_names=cell_names  # Provide cell names
)

# Add custom modality #2: ATAC peaks (numpy array)
atac_array = np.random.negative_binomial(5, 0.3, (200, n_cells))
atac_meta = pd.DataFrame({'peak': [f'peak_{i}' for i in range(200)]})

model.add_custom_modality(
    name='atac',
    counts=atac_array,
    feature_meta=atac_meta,
    distribution='negbinom',
    cell_names=cell_names  # Provide cell names
)

# Verify all modalities have matching cells
print(model.list_modalities())
for name in ['gene', 'spliz', 'atac']:
    mod = model.get_modality(name)
    print(f"{name}: {mod.dims['n_cells']} cells, cell_names={mod.cell_names[:3]}...")
```

## Best Practices

1. **Use DataFrames when possible**: Automatic cell name extraction reduces errors
2. **Consistent naming**: Use the same cell identifiers across all modalities
3. **Match metadata**: Ensure `meta['cell']` matches count column names
4. **Provide cell_names for numpy**: Always include cell_names when using numpy arrays for better traceability
5. **Validate alignment**: Check that all modalities have the expected number of cells after initialization

## Testing

The functionality is tested in `tests/test_cell_names_numpy.py`:

```bash
python tests/test_cell_names_numpy.py
```

Tests validate:
- ✅ cell_names parameter works with numpy arrays
- ✅ cell_names auto-extracted from DataFrames
- ✅ Cell subsetting preserves cell_names
- ✅ Legacy behavior (no cell_names) still works

## See Also

- [QUICKSTART_MULTIMODAL.md](QUICKSTART_MULTIMODAL.md) - Multi-modal analysis guide
- [API_REFERENCE.md](API_REFERENCE.md) - Complete API documentation
