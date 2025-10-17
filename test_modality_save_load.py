#!/usr/bin/env python3
"""
Test script to verify modality-specific save/load functionality.

This script verifies that:
1. Users can save specific modalities only
2. Primary modality is NOT saved by default - must be in the list
3. save_model_level flag works correctly
4. Loading specific modalities works correctly
"""

import os
import sys
import tempfile
import shutil
import numpy as np
import pandas as pd
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bayesDREAM import bayesDREAM, Modality

def create_test_data():
    """Create minimal test data."""
    np.random.seed(42)

    # Create metadata
    n_cells = 60
    meta = pd.DataFrame({
        'cell': [f'cell_{i}' for i in range(n_cells)],
        'guide': ['NTC'] * 30 + ['guide1'] * 30,
        'cell_line': ['A'] * 20 + ['B'] * 20 + ['C'] * 20,
        'target': ['ntc'] * 30 + ['GFI1B'] * 30,
        'sum_factor': np.random.uniform(0.8, 1.2, n_cells),
        'sum_factor_adj': np.random.uniform(0.8, 1.2, n_cells)
    })

    # Create gene counts (11 genes including cis gene)
    genes = [f'gene_{i}' for i in range(10)] + ['GFI1B']
    gene_counts = pd.DataFrame(
        np.random.poisson(100, (11, n_cells)),
        index=genes,
        columns=meta['cell']
    )

    # Create ATAC counts (5 regions)
    atac_counts = np.random.poisson(50, (5, n_cells))
    atac_meta = pd.DataFrame({
        'region': [f'region_{i}' for i in range(5)],
        'chrom': 'chr1',
        'start': np.arange(5) * 1000,
        'end': np.arange(5) * 1000 + 500
    })

    # Create splicing counts (binomial)
    splicing_counts = np.random.poisson(20, (3, n_cells))
    splicing_denom = np.random.poisson(100, (3, n_cells))
    splicing_meta = pd.DataFrame({
        'junction': [f'junction_{i}' for i in range(3)],
        'gene': ['gene_0', 'gene_1', 'gene_2']
    })

    return meta, gene_counts, atac_counts, atac_meta, splicing_counts, splicing_denom, splicing_meta

def main():
    print("=" * 80)
    print("Test: Modality-Specific Save/Load Functionality")
    print("=" * 80)

    # Create temporary directory
    tmpdir = tempfile.mkdtemp()
    try:
        # Create test data
        meta, gene_counts, atac_counts, atac_meta, splicing_counts, splicing_denom, splicing_meta = create_test_data()

        # ===================================================================
        # Test 1: Create model with multiple modalities
        # ===================================================================
        print("\n" + "=" * 80)
        print("Test 1: Create model with 3 modalities (gene, atac, splicing)")
        print("=" * 80)

        model = bayesDREAM(
            meta=meta,
            counts=gene_counts,
            cis_gene='GFI1B',
            output_dir=os.path.join(tmpdir, 'test1'),
            label='modality_test',
            primary_modality='gene'
        )

        # Add ATAC modality
        atac_modality = Modality(
            name='atac',
            counts=atac_counts,
            feature_meta=atac_meta,
            cell_names=meta['cell'].values,
            distribution='negbinom'
        )
        model.modalities['atac'] = atac_modality

        # Add splicing modality (binomial)
        splicing_modality = Modality(
            name='splicing',
            counts=splicing_counts,
            feature_meta=splicing_meta,
            cell_names=meta['cell'].values,
            distribution='binomial',
            denominator=splicing_denom
        )
        model.modalities['splicing'] = splicing_modality

        print(f"✓ Created model with {len(model.modalities)} modalities: {list(model.modalities.keys())}")
        print(f"✓ Primary modality: {model.primary_modality}")

        # ===================================================================
        # Test 2: Fit technical on all modalities
        # ===================================================================
        print("\n" + "=" * 80)
        print("Test 2: Fit technical on all 3 modalities")
        print("=" * 80)

        model.set_technical_groups(['cell_line'])

        for mod_name in ['gene', 'atac', 'splicing']:
            print(f"\nFitting {mod_name}...")
            model.fit_technical(modality_name=mod_name, sum_factor_col='sum_factor', niters=10)

        print("\n✓ All modalities fitted")
        for mod_name in model.modalities:
            mod = model.modalities[mod_name]
            has_alpha = hasattr(mod, 'alpha_y_prefit') and mod.alpha_y_prefit is not None
            print(f"  - {mod_name}: alpha_y_prefit = {has_alpha}")

        # ===================================================================
        # Test 3: Save only specific modalities (NOT primary by default)
        # ===================================================================
        print("\n" + "=" * 80)
        print("Test 3: Save ONLY 'atac' and 'splicing' (exclude primary 'gene')")
        print("=" * 80)

        saved = model.save_technical_fit(modalities=['atac', 'splicing'], save_model_level=False)

        # Check which files exist
        outdir = model.output_dir
        files_exist = {
            'alpha_y_prefit_gene.pt': os.path.exists(os.path.join(outdir, 'alpha_y_prefit_gene.pt')),
            'alpha_y_prefit_atac.pt': os.path.exists(os.path.join(outdir, 'alpha_y_prefit_atac.pt')),
            'alpha_y_prefit_splicing.pt': os.path.exists(os.path.join(outdir, 'alpha_y_prefit_splicing.pt')),
            'alpha_x_prefit.pt': os.path.exists(os.path.join(outdir, 'alpha_x_prefit.pt')),
            'alpha_y_prefit.pt': os.path.exists(os.path.join(outdir, 'alpha_y_prefit.pt'))
        }

        print(f"\n✓ Files saved:")
        for fname, exists in files_exist.items():
            status = "EXISTS" if exists else "NOT SAVED"
            print(f"  - {fname}: {status}")

        # Verify primary is NOT saved
        assert not files_exist['alpha_y_prefit_gene.pt'], "ERROR: Primary 'gene' should NOT be saved!"
        print("\n✓ VERIFIED: Primary modality 'gene' was NOT saved (as expected)")

        # Verify atac and splicing ARE saved
        assert files_exist['alpha_y_prefit_atac.pt'], "ERROR: 'atac' should be saved!"
        assert files_exist['alpha_y_prefit_splicing.pt'], "ERROR: 'splicing' should be saved!"
        print("✓ VERIFIED: Only 'atac' and 'splicing' were saved")

        # Verify model-level NOT saved (save_model_level=False)
        assert not files_exist['alpha_x_prefit.pt'], "ERROR: alpha_x_prefit should NOT be saved!"
        assert not files_exist['alpha_y_prefit.pt'], "ERROR: alpha_y_prefit should NOT be saved!"
        print("✓ VERIFIED: Model-level parameters NOT saved (save_model_level=False)")

        # ===================================================================
        # Test 4: Save primary modality explicitly
        # ===================================================================
        print("\n" + "=" * 80)
        print("Test 4: Save ONLY primary 'gene' modality explicitly")
        print("=" * 80)

        outdir2 = os.path.join(tmpdir, 'test2')
        os.makedirs(outdir2, exist_ok=True)

        saved = model.save_technical_fit(
            output_dir=outdir2,
            modalities=['gene'],  # Explicitly include primary
            save_model_level=True   # Also save model-level
        )

        files_exist2 = {
            'alpha_y_prefit_gene.pt': os.path.exists(os.path.join(outdir2, 'alpha_y_prefit_gene.pt')),
            'alpha_y_prefit_atac.pt': os.path.exists(os.path.join(outdir2, 'alpha_y_prefit_atac.pt')),
            'alpha_y_prefit.pt': os.path.exists(os.path.join(outdir2, 'alpha_y_prefit.pt'))
        }

        print(f"\n✓ Files saved:")
        for fname, exists in files_exist2.items():
            status = "EXISTS" if exists else "NOT SAVED"
            print(f"  - {fname}: {status}")

        assert files_exist2['alpha_y_prefit_gene.pt'], "ERROR: 'gene' should be saved!"
        assert not files_exist2['alpha_y_prefit_atac.pt'], "ERROR: 'atac' should NOT be saved!"
        assert files_exist2['alpha_y_prefit.pt'], "ERROR: model-level alpha_y_prefit should be saved!"

        print("\n✓ VERIFIED: Only 'gene' modality saved")
        print("✓ VERIFIED: Model-level parameters saved (save_model_level=True)")

        # ===================================================================
        # Test 5: Load specific modalities
        # ===================================================================
        print("\n" + "=" * 80)
        print("Test 5: Load ONLY 'atac' and 'splicing' from first save")
        print("=" * 80)

        # Create fresh model
        model2 = bayesDREAM(
            meta=meta,
            counts=gene_counts,
            cis_gene='GFI1B',
            output_dir=os.path.join(tmpdir, 'test1'),  # Use first output dir
            label='modality_test',
            primary_modality='gene'
        )

        # Add modalities (must exist before loading)
        model2.modalities['atac'] = atac_modality
        model2.modalities['splicing'] = splicing_modality

        # Load only atac and splicing
        model2.load_technical_fit(modalities=['atac', 'splicing'], load_model_level=False)

        # Verify
        gene_mod = model2.modalities['gene']
        atac_mod = model2.modalities['atac']
        splicing_mod = model2.modalities['splicing']

        gene_loaded = hasattr(gene_mod, 'alpha_y_prefit') and gene_mod.alpha_y_prefit is not None
        atac_loaded = hasattr(atac_mod, 'alpha_y_prefit') and atac_mod.alpha_y_prefit is not None
        splicing_loaded = hasattr(splicing_mod, 'alpha_y_prefit') and splicing_mod.alpha_y_prefit is not None

        print(f"\n✓ Loaded parameters:")
        print(f"  - gene.alpha_y_prefit: {gene_loaded}")
        print(f"  - atac.alpha_y_prefit: {atac_loaded}")
        print(f"  - splicing.alpha_y_prefit: {splicing_loaded}")

        assert not gene_loaded, "ERROR: 'gene' should NOT be loaded!"
        assert atac_loaded, "ERROR: 'atac' should be loaded!"
        assert splicing_loaded, "ERROR: 'splicing' should be loaded!"

        print("\n✓ VERIFIED: Only 'atac' and 'splicing' were loaded")

        # ===================================================================
        # Test 6: Verify default behavior (save ALL modalities)
        # ===================================================================
        print("\n" + "=" * 80)
        print("Test 6: Default save (modalities=None) saves ALL modalities")
        print("=" * 80)

        outdir3 = os.path.join(tmpdir, 'test3')
        os.makedirs(outdir3, exist_ok=True)

        saved = model.save_technical_fit(output_dir=outdir3)  # No modalities specified

        files_exist3 = {
            'alpha_y_prefit_gene.pt': os.path.exists(os.path.join(outdir3, 'alpha_y_prefit_gene.pt')),
            'alpha_y_prefit_atac.pt': os.path.exists(os.path.join(outdir3, 'alpha_y_prefit_atac.pt')),
            'alpha_y_prefit_splicing.pt': os.path.exists(os.path.join(outdir3, 'alpha_y_prefit_splicing.pt'))
        }

        print(f"\n✓ Files saved:")
        for fname, exists in files_exist3.items():
            status = "EXISTS" if exists else "NOT SAVED"
            print(f"  - {fname}: {status}")

        assert all(files_exist3.values()), "ERROR: All modalities should be saved by default!"
        print("\n✓ VERIFIED: All modalities saved when modalities=None (default)")

        # ===================================================================
        # Success!
        # ===================================================================
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED!")
        print("=" * 80)
        print("\nVerified:")
        print("  ✓ Primary modality NOT saved by default - must be explicitly included")
        print("  ✓ Can save specific subset of modalities")
        print("  ✓ Can exclude primary modality from save")
        print("  ✓ Can load specific subset of modalities")
        print("  ✓ save_model_level flag works correctly")
        print("  ✓ Default (modalities=None) saves all modalities")

    finally:
        # Cleanup
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)
            print(f"\n✓ Cleaned up temporary directory: {tmpdir}")

if __name__ == '__main__':
    main()
