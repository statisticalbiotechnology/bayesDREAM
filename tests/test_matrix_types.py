"""
Test bayesDREAM with different matrix types:
- Dense numpy arrays
- Sparse scipy matrices
- Pandas DataFrames

Ensures that all initialization and fitting works uniformly.
"""

import numpy as np
import pandas as pd
from scipy import sparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bayesDREAM import bayesDREAM

def create_test_data(n_genes=50, n_cells=100, matrix_type='dataframe'):
    """Create test data in different matrix formats."""
    np.random.seed(42)

    # Create metadata
    meta = pd.DataFrame({
        'cell': [f'Cell_{i}' for i in range(n_cells)],
        'guide': ['guide_ntc'] * (n_cells // 2) + ['guide_GFI1B'] * (n_cells // 2),
        'target': ['ntc'] * (n_cells // 2) + ['GFI1B'] * (n_cells // 2),
        'sum_factor': np.random.uniform(0.8, 1.2, n_cells),
        'cell_line': ['line1'] * (n_cells // 2) + ['line2'] * (n_cells // 2)
    })

    # Create count matrix
    counts_array = np.random.randint(0, 200, (n_genes, n_cells))

    # Convert to requested format and create appropriate gene_meta
    if matrix_type == 'dataframe':
        # DataFrame: use custom gene names with GFI1B as first gene
        gene_names = ['GFI1B'] + [f'Gene_{i}' for i in range(1, n_genes)]
        cell_names = meta['cell'].tolist()
        counts = pd.DataFrame(
            counts_array,
            index=gene_names,
            columns=cell_names
        )
        # gene_meta with gene names as index (matching counts index)
        gene_meta = pd.DataFrame({
            'gene_name': gene_names,
            'gene_id': [f'ENSG{i:08d}' for i in range(n_genes)]
        }, index=gene_names)
    elif matrix_type == 'dense':
        # Dense array: use NUMERIC index with actual gene names in metadata
        counts = counts_array
        # gene_meta with NUMERIC index (0, 1, 2, ...) and actual gene names
        # Row 0 is GFI1B, rest are Gene_1, Gene_2, etc.
        gene_names_for_meta = ['GFI1B'] + [f'Gene_{i}' for i in range(1, n_genes)]
        gene_meta = pd.DataFrame({
            'gene_name': gene_names_for_meta,
            'gene_id': [f'ENSG{i:08d}' for i in range(n_genes)],
            'note': ['This is GFI1B' if i == 0 else '' for i in range(n_genes)]
        }, index=range(n_genes))  # NUMERIC index [0, 1, 2, ...]
    elif matrix_type == 'sparse':
        # Sparse matrix: use NUMERIC index with actual gene names in metadata
        counts = sparse.csr_matrix(counts_array.astype(float))
        # gene_meta with NUMERIC index (0, 1, 2, ...) and actual gene names
        # Row 0 is GFI1B, rest are Gene_1, Gene_2, etc.
        gene_names_for_meta = ['GFI1B'] + [f'Gene_{i}' for i in range(1, n_genes)]
        gene_meta = pd.DataFrame({
            'gene_name': gene_names_for_meta,
            'gene_id': [f'ENSG{i:08d}' for i in range(n_genes)],
            'note': ['This is GFI1B' if i == 0 else '' for i in range(n_genes)]
        }, index=range(n_genes))  # NUMERIC index [0, 1, 2, ...]
    else:
        raise ValueError(f"Unknown matrix_type: {matrix_type}")

    return meta, counts, gene_meta


def test_matrix_type(matrix_type, verbose=True):
    """Test bayesDREAM initialization with a specific matrix type."""
    if verbose:
        print(f"\n{'='*80}")
        print(f"TEST: {matrix_type.upper()} MATRIX")
        print('='*80)

    try:
        # Create data
        meta, counts, gene_meta = create_test_data(n_genes=50, n_cells=100, matrix_type=matrix_type)

        if verbose:
            print(f"Created test data:")
            print(f"  counts type: {type(counts)}")
            print(f"  counts shape: {counts.shape}")
            if hasattr(counts, 'nnz'):
                sparsity = 1 - counts.nnz / (counts.shape[0] * counts.shape[1])
                print(f"  sparsity: {sparsity:.2%}")

        # Initialize model
        # For matrices/arrays, we need to specify which row is the cis feature using numeric index
        # For DataFrames, we can use gene names directly
        if matrix_type == 'dataframe':
            model = bayesDREAM(
                meta=meta,
                counts=counts,
                feature_meta=gene_meta,
                cis_gene='GFI1B',
                output_dir=f'./test_output_{matrix_type}',
                label=f'test_{matrix_type}',
                device='cpu'
            )
        else:
            # For dense/sparse: pass feature_meta with numeric index and specify cis gene by name in that meta
            model = bayesDREAM(
                meta=meta,
                counts=counts,
                feature_meta=gene_meta,
                cis_gene='GFI1B',  # This will be found in gene_meta['gene_name']
                output_dir=f'./test_output_{matrix_type}',
                label=f'test_{matrix_type}',
                device='cpu'
            )

        if verbose:
            print(f"\n✓ Model initialized successfully")
            print(f"  Counts shape: {model.counts.shape}")
            print(f"  Device: {model.device}")
            print(f"  Modalities: {model.list_modalities()}")

            # Check counts type preservation
            if matrix_type == 'sparse':
                assert sparse.issparse(model.counts), "Sparse matrix not preserved!"
                print(f"  ✓ Sparse matrix preserved (type: {type(model.counts).__name__})")
            elif matrix_type == 'dense':
                assert isinstance(model.counts, np.ndarray), "Dense array not preserved!"
                print(f"  ✓ Dense array preserved (type: {type(model.counts).__name__})")
            elif matrix_type == 'dataframe':
                assert isinstance(model.counts, pd.DataFrame), "DataFrame not preserved!"
                print(f"  ✓ DataFrame preserved (type: {type(model.counts).__name__})")

        # Check modality types
        cis_mod = model.get_modality('cis')
        gene_mod = model.get_modality('gene')

        if verbose:
            print(f"\n  Cis modality:")
            print(f"    - counts type: {type(cis_mod.counts)}")
            print(f"    - counts shape: {cis_mod.counts.shape}")
            print(f"    - is_sparse: {cis_mod.is_sparse}")

            print(f"\n  Gene modality:")
            print(f"    - counts type: {type(gene_mod.counts)}")
            print(f"    - counts shape: {gene_mod.counts.shape}")
            print(f"    - is_sparse: {gene_mod.is_sparse}")

        # Verify numeric indexing works
        assert len(gene_mod.feature_meta) == gene_mod.counts.shape[0], \
            f"Feature metadata rows ({len(gene_mod.feature_meta)}) != counts rows ({gene_mod.counts.shape[0]})"

        if verbose:
            print(f"\n  ✓ Numeric indexing verified")

        return True, "Success"

    except Exception as e:
        if verbose:
            print(f"\n✗ FAILED: {str(e)}")
        return False, str(e)


def test_all_matrix_types():
    """Test all matrix types."""
    print("="*80)
    print("TESTING BAYESDREAM WITH DIFFERENT MATRIX TYPES")
    print("="*80)

    results = {}

    for matrix_type in ['dataframe', 'dense', 'sparse']:
        success, message = test_matrix_type(matrix_type, verbose=True)
        results[matrix_type] = (success, message)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    all_passed = True
    for matrix_type, (success, message) in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{matrix_type:12s}: {status}")
        if not success:
            print(f"             {message}")
            all_passed = False

    if all_passed:
        print("\n" + "="*80)
        print("ALL TESTS PASSED!")
        print("="*80)
        return 0
    else:
        print("\n" + "="*80)
        print("SOME TESTS FAILED")
        print("="*80)
        return 1


if __name__ == '__main__':
    exit_code = test_all_matrix_types()
    sys.exit(exit_code)
