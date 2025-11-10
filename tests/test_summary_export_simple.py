"""
Simple test of summary export functionality using mock posterior data.

This test validates that all summary export methods work correctly
without requiring a full model fitting pipeline run.
"""

import pandas as pd
import numpy as np
import torch
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bayesDREAM import bayesDREAM

def test_summary_export_simple():
    """Test summary export with mock posterior data."""

    print("=" * 80)
    print("Testing Summary Export Functionality (Simple)")
    print("=" * 80)

    # Create minimal toy data
    print("\n1. Creating minimal toy data...")
    np.random.seed(42)
    n_genes = 10
    n_cells = 100
    n_guides = 6

    # Cell metadata
    meta = pd.DataFrame({
        'cell': [f'cell_{i}' for i in range(n_cells)],
        'guide': np.random.choice([f'guide_{i}' for i in range(n_guides)], n_cells),
        'cell_line': np.random.choice(['K562', 'HEL'], n_cells),
        'target': np.concatenate([['GFI1B'] * 60, ['ntc'] * 40]),
        'sum_factor': np.random.lognormal(0, 0.2, n_cells),
        'sum_factor_adj': np.random.lognormal(0, 0.2, n_cells)
    })

    # Gene counts (including cis gene)
    genes = [f'gene_{i}' for i in range(n_genes)] + ['GFI1B']
    gene_counts = pd.DataFrame(
        np.random.negative_binomial(10, 0.5, (len(genes), n_cells)),
        index=genes,
        columns=meta['cell']
    )

    print(f"   - Created {n_cells} cells")
    print(f"   - Created {len(genes)} genes")

    # Create output directory
    output_dir = './test_output/summary_export_simple'
    os.makedirs(output_dir, exist_ok=True)

    # Initialize model
    print("\n2. Initializing bayesDREAM model...")
    model = bayesDREAM(
        meta=meta,
        counts=gene_counts,
        cis_gene='GFI1B',
        guide_covariates=['cell_line'],
        output_dir=output_dir,
        label='summary_test_simple',
        device='cpu'
    )

    print(f"   ✓ Model created successfully")
    print(f"   - Cis gene: {model.cis_gene}")
    print(f"   - Primary modality: {model.primary_modality}")

    # Test 1: Technical summary with mock data
    print("\n3. Testing save_technical_summary() with mock data...")
    try:
        # Set technical groups
        model.set_technical_groups(['cell_line'])

        # Create mock technical fit posteriors
        n_features = model.modalities['gene'].dims['n_features']
        n_groups = 2  # K562, HEL

        # alpha_y_prefit should be [n_samples, n_groups, n_features]
        model.alpha_y_prefit = torch.randn((100, n_groups, n_features), device=model.device)

        # Export technical summary
        tech_df = model.save_technical_summary()
        print(f"   ✓ Technical summary exported")
        print(f"   - Shape: {tech_df.shape}")
        print(f"   - Columns: {tech_df.columns.tolist()}")
        print(f"\n   First few rows:")
        print(tech_df.head(3))

        # Validate
        assert 'feature' in tech_df.columns
        assert 'modality' in tech_df.columns
        assert 'distribution' in tech_df.columns
        assert any('alpha_y' in col for col in tech_df.columns)

        csv_path = os.path.join(output_dir, 'technical_feature_summary_gene.csv')
        assert os.path.exists(csv_path)
        print(f"   ✓ CSV file created: {csv_path}")

    except Exception as e:
        print(f"   ✗ Technical summary export failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 2: Cis summary with mock data
    print("\n4. Testing save_cis_summary() with mock data...")
    try:
        # Create mock cis fit posteriors
        n_guides_unique = model.meta['guide'].nunique()

        model.x_true = torch.randn(n_guides_unique, device=model.device) + 5  # Around log2(32)
        model.posterior_samples_cis = {
            'x_true': torch.randn((100, n_guides_unique), device=model.device) + 5
        }

        # Export cis summary
        guide_df, cell_df = model.save_cis_summary()
        print(f"   ✓ Cis summary exported")
        print(f"\n   Guide-level summary:")
        print(f"   - Shape: {guide_df.shape}")
        print(f"   - Columns: {guide_df.columns.tolist()}")
        print(guide_df.head(3))

        print(f"\n   Cell-level summary:")
        print(f"   - Shape: {cell_df.shape}")
        print(f"   - First few rows:")
        print(cell_df.head(3))

        # Validate
        assert 'guide' in guide_df.columns
        assert 'target' in guide_df.columns
        assert 'x_true_mean' in guide_df.columns
        assert 'x_true_lower' in guide_df.columns
        assert 'x_true_upper' in guide_df.columns

        guide_csv = os.path.join(output_dir, 'cis_guide_summary.csv')
        cell_csv = os.path.join(output_dir, 'cis_cell_summary.csv')
        assert os.path.exists(guide_csv)
        assert os.path.exists(cell_csv)
        print(f"   ✓ CSV files created")

    except Exception as e:
        print(f"   ✗ Cis summary export failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 3: Trans summary with mock data (additive_hill)
    print("\n5. Testing save_trans_summary() with mock data (additive_hill)...")
    try:
        # Create mock trans fit posteriors (additive_hill)
        n_features_trans = model.modalities['gene'].dims['n_features']
        n_samples = 100

        model.function_type = 'additive_hill'

        # params_pos should be [n_samples, n_features, 3] where last dim is [B, K, xc]
        params_pos = torch.zeros((n_samples, n_features_trans, 3), device=model.device)
        params_pos[:, :, 0] = torch.randn((n_samples, n_features_trans), device=model.device)  # B
        params_pos[:, :, 1] = torch.abs(torch.randn((n_samples, n_features_trans), device=model.device)) + 1.5  # K
        params_pos[:, :, 2] = torch.randn((n_samples, n_features_trans), device=model.device) + 5  # xc

        params_neg = torch.zeros((n_samples, n_features_trans, 3), device=model.device)
        params_neg[:, :, 0] = torch.randn((n_samples, n_features_trans), device=model.device)  # B
        params_neg[:, :, 1] = torch.abs(torch.randn((n_samples, n_features_trans), device=model.device)) + 1.5  # K
        params_neg[:, :, 2] = torch.randn((n_samples, n_features_trans), device=model.device) + 5  # xc

        model.posterior_samples_trans = {
            'params_pos': params_pos,
            'params_neg': params_neg,
            'pi_y': torch.rand((n_samples, n_features_trans), device=model.device) * 0.5 + 0.5
        }

        # Export trans summary
        trans_df = model.save_trans_summary(
            compute_inflection=True,
            compute_full_log2fc=True
        )
        print(f"   ✓ Trans summary exported")
        print(f"   - Shape: {trans_df.shape}")
        print(f"   - Sample columns: {trans_df.columns[:10].tolist()}...")
        print(f"\n   First row (example):")
        print(trans_df.iloc[0])

        # Validate
        assert 'feature' in trans_df.columns
        assert 'function_type' in trans_df.columns
        assert 'observed_log2fc' in trans_df.columns
        assert 'B_pos_mean' in trans_df.columns
        assert 'EC50_pos_mean' in trans_df.columns
        assert 'inflection_pos_mean' in trans_df.columns
        assert 'full_log2fc_mean' in trans_df.columns

        csv_path = os.path.join(output_dir, 'trans_feature_summary_gene.csv')
        assert os.path.exists(csv_path)
        print(f"   ✓ CSV file created: {csv_path}")

    except Exception as e:
        print(f"   ✗ Trans summary export failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 4: Trans summary with polynomial
    print("\n6. Testing save_trans_summary() with mock data (polynomial)...")
    try:
        # Create mock trans fit posteriors (polynomial)
        degree = 6
        model.function_type = 'polynomial'
        model.posterior_samples_trans = {
            'poly_coefs': torch.randn((n_samples, n_features_trans, degree), device=model.device)
        }

        # Export trans summary
        poly_df = model.save_trans_summary(
            compute_inflection=False,  # Not applicable for polynomial
            compute_full_log2fc=True
        )
        print(f"   ✓ Trans summary (polynomial) exported")
        print(f"   - Shape: {poly_df.shape}")

        # Validate
        assert 'coef_0_mean' in poly_df.columns
        assert 'full_log2fc_mean' in poly_df.columns

        print(f"   ✓ Polynomial summary structure validated")

    except Exception as e:
        print(f"   ✗ Trans summary (polynomial) export failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED")
    print("=" * 80)
    print(f"\nSummary files created in: {output_dir}")
    print("Files:")
    print("  - technical_feature_summary_gene.csv")
    print("  - cis_guide_summary.csv")
    print("  - cis_cell_summary.csv")
    print("  - trans_feature_summary_gene.csv")
    print("\nThese files are ready to use in R for plotting!")

    return True

if __name__ == '__main__':
    success = test_summary_export_simple()
    sys.exit(0 if success else 1)
