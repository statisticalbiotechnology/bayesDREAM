"""
Test cell_names parameter for add_custom_modality with numpy arrays.

This test validates that cell names can be explicitly provided when adding
modalities using numpy arrays instead of DataFrames.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bayesDREAM import bayesDREAM

def test_cell_names_with_numpy():
    """Test add_custom_modality with cell_names parameter."""

    print("=" * 80)
    print("Testing cell_names Parameter with Numpy Arrays")
    print("=" * 80)

    # Create minimal toy data
    print("\n1. Creating toy data...")
    np.random.seed(42)
    n_genes = 10
    n_cells = 50
    n_guides = 5

    # Cell names
    cell_names = [f'cell_{i}' for i in range(n_cells)]

    # Cell metadata
    meta = pd.DataFrame({
        'cell': cell_names,
        'guide': np.random.choice([f'guide_{i}' for i in range(n_guides)], n_cells),
        'cell_line': np.random.choice(['K562', 'HEL'], n_cells),
        'target': np.concatenate([['GFI1B'] * 30, ['ntc'] * 20]),
        'sum_factor': np.random.lognormal(0, 0.2, n_cells)
    })

    # Gene counts (DataFrame)
    genes = [f'gene_{i}' for i in range(n_genes)] + ['GFI1B']
    gene_counts_df = pd.DataFrame(
        np.random.negative_binomial(10, 0.5, (len(genes), n_cells)),
        index=genes,
        columns=cell_names
    )

    print(f"   - Created {n_cells} cells")
    print(f"   - Created {len(genes)} genes")

    # Initialize model
    print("\n2. Initializing bayesDREAM model...")
    model = bayesDREAM(
        meta=meta,
        counts=gene_counts_df,
        cis_gene='GFI1B',
        output_dir='./test_output/cell_names_test',
        label='cell_names_test',
        device='cpu'
    )

    print(f"   ✓ Model created")
    print(f"   - Gene modality has {model.get_modality('gene').dims['n_cells']} cells")
    print(f"   - Gene modality cell_names: {model.get_modality('gene').cell_names[:3]}... (first 3)")

    # Test 1: Add custom modality with numpy array + cell_names
    print("\n3. Testing add_custom_modality with numpy array + cell_names...")

    # Create numpy array for custom modality
    custom_counts_array = np.random.randn(15, n_cells)  # 15 features, 50 cells
    custom_feature_meta = pd.DataFrame({
        'feature': [f'custom_feature_{i}' for i in range(15)]
    })

    try:
        model.add_custom_modality(
            name='custom_array',
            counts=custom_counts_array,
            feature_meta=custom_feature_meta,
            distribution='normal',
            cell_names=cell_names  # Explicitly provide cell names
        )
        print(f"   ✓ Custom modality added with cell_names")

        # Verify cell_names were stored
        custom_mod = model.get_modality('custom_array')
        print(f"   - Custom modality has {custom_mod.dims['n_cells']} cells")
        print(f"   - Custom modality cell_names: {custom_mod.cell_names[:3]}... (first 3)")

        assert custom_mod.cell_names is not None, "cell_names should not be None"
        assert len(custom_mod.cell_names) == n_cells, f"Expected {n_cells} cell names"
        assert custom_mod.cell_names == cell_names, "cell_names should match provided list"
        print(f"   ✓ cell_names correctly stored and validated")

    except Exception as e:
        print(f"   ✗ Failed to add custom modality: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 2: Add custom modality with DataFrame (should auto-extract cell_names)
    print("\n4. Testing add_custom_modality with DataFrame (auto cell_names)...")

    custom_counts_df = pd.DataFrame(
        np.random.randn(10, n_cells),
        index=[f'df_feature_{i}' for i in range(10)],
        columns=cell_names
    )
    custom_feature_meta_df = pd.DataFrame({
        'feature': [f'df_feature_{i}' for i in range(10)]
    })

    try:
        model.add_custom_modality(
            name='custom_dataframe',
            counts=custom_counts_df,
            feature_meta=custom_feature_meta_df,
            distribution='normal'
            # No cell_names parameter - should extract from DataFrame columns
        )
        print(f"   ✓ Custom modality added from DataFrame")

        # Verify cell_names were extracted
        custom_df_mod = model.get_modality('custom_dataframe')
        print(f"   - DataFrame modality has {custom_df_mod.dims['n_cells']} cells")
        print(f"   - DataFrame modality cell_names: {custom_df_mod.cell_names[:3]}... (first 3)")

        assert custom_df_mod.cell_names is not None, "cell_names should not be None for DataFrame"
        assert len(custom_df_mod.cell_names) == n_cells, f"Expected {n_cells} cell names"
        assert custom_df_mod.cell_names == cell_names, "cell_names should match DataFrame columns"
        print(f"   ✓ cell_names correctly extracted from DataFrame")

    except Exception as e:
        print(f"   ✗ Failed to add DataFrame modality: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 3: Verify cell subsetting works with cell_names
    print("\n5. Testing cell subsetting with cell_names...")

    try:
        # Subset to first 20 cells using cell names
        subset_cells = cell_names[:20]
        custom_subset = custom_mod.get_cell_subset(subset_cells)

        print(f"   ✓ Cell subset created")
        print(f"   - Subset has {custom_subset.dims['n_cells']} cells")
        print(f"   - Subset cell_names: {custom_subset.cell_names[:3]}... (first 3)")

        assert custom_subset.cell_names is not None, "cell_names should be preserved in subset"
        assert len(custom_subset.cell_names) == 20, "Expected 20 cells in subset"
        assert custom_subset.cell_names == subset_cells, "Subset cell_names should match"
        print(f"   ✓ Cell subsetting preserves cell_names correctly")

    except Exception as e:
        print(f"   ✗ Cell subsetting failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 4: Add modality without cell_names (should still work)
    print("\n6. Testing add_custom_modality without cell_names (legacy behavior)...")

    custom_counts_no_names = np.random.randn(8, n_cells)
    custom_feature_meta_no_names = pd.DataFrame({
        'feature': [f'no_names_feature_{i}' for i in range(8)]
    })

    try:
        model.add_custom_modality(
            name='custom_no_names',
            counts=custom_counts_no_names,
            feature_meta=custom_feature_meta_no_names,
            distribution='normal'
            # No cell_names - should work but cell_names will be None
        )
        print(f"   ✓ Custom modality added without cell_names")

        no_names_mod = model.get_modality('custom_no_names')
        print(f"   - Modality has {no_names_mod.dims['n_cells']} cells")
        print(f"   - Modality cell_names: {no_names_mod.cell_names}")

        assert no_names_mod.cell_names is None, "cell_names should be None when not provided"
        print(f"   ✓ Legacy behavior (no cell_names) still works")

    except Exception as e:
        print(f"   ✗ Failed to add modality without cell_names: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED")
    print("=" * 80)
    print("\nSummary:")
    print("  ✓ cell_names parameter works with numpy arrays")
    print("  ✓ cell_names auto-extracted from DataFrames")
    print("  ✓ Cell subsetting preserves cell_names")
    print("  ✓ Legacy behavior (no cell_names) still works")

    return True

if __name__ == '__main__':
    success = test_cell_names_with_numpy()
    sys.exit(0 if success else 1)
