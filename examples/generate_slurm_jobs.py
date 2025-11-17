#!/usr/bin/env python3
"""
Example: Generate SLURM job scripts for bayesDREAM on Berzelius

This script demonstrates how to use SlurmJobGenerator to create
optimized SLURM submission scripts for large datasets.

Usage:
    python generate_slurm_jobs.py

The generator will:
1. Analyze your dataset (size, sparsity, etc.)
2. Estimate memory and time requirements
3. Select appropriate resources (fat/thin/cpu nodes)
4. Generate SLURM scripts with job dependencies
5. Create a master submission script

Output:
    ./slurm_jobs/
    ├── 01_fit_technical.sh
    ├── 02_fit_cis.sh
    ├── 03_fit_trans.sh
    ├── submit_all.sh
    └── README.md
"""

import pandas as pd
import numpy as np
from scipy import sparse
from bayesDREAM.slurm_jobgen import SlurmJobGenerator

# =============================================================================
# Step 1: Load or prepare your data
# =============================================================================

print("Loading data...")

# Option A: Load from saved files
meta = pd.read_csv('../toydata/gene_meta.csv')
counts = pd.read_csv('../toydata/gene_counts.csv', index_col=0)

# Option B: Prepare from AnnData
# import scanpy as sc
# adata = sc.read_h5ad('data.h5ad')
# meta = adata.obs.copy()
# counts = adata.X.T  # Features × cells

# Option C: Load sparse matrix
# from scipy.io import mmread
# counts = mmread('counts.mtx').T.tocsr()
# meta = pd.read_csv('meta.csv')

print(f"Loaded: {counts.shape[0]} genes × {counts.shape[1]} cells")

# =============================================================================
# Step 2: Specify cis genes to fit
# =============================================================================

cis_genes = ['GFI1B', 'TET2', 'MYB', 'NFE2']

# Or auto-detect from targets
# cis_genes = meta[meta['target'] != 'ntc']['target'].unique().tolist()

print(f"Cis genes: {cis_genes}")

# =============================================================================
# Step 3: Save data to location accessible by Berzelius
# =============================================================================

data_path = "/proj/berzelius-aiics-real/users/x_learo/bayesdream_data/run_20250117"

# Create directory if needed (do this on Berzelius)
import os
os.makedirs(data_path, exist_ok=True)

# Save data
meta.to_csv(f"{data_path}/meta.csv", index=False)
counts.to_csv(f"{data_path}/counts.csv")

print(f"Data saved to: {data_path}")

# =============================================================================
# Step 4: Generate SLURM scripts
# =============================================================================

print("\nGenerating SLURM scripts...")

generator = SlurmJobGenerator(
    meta=meta,
    counts=counts,
    cis_genes=cis_genes,
    output_dir='./slurm_jobs',
    label='perturb_seq_20250117',

    # Experiment type
    low_moi=True,  # Set to False for high MOI
    use_all_cells_technical=False,  # True for high MOI with all-cells mode

    # Data characteristics
    distribution='negbinom',
    sparsity=None,  # Auto-detect
    n_groups=None,  # Auto-detect from meta

    # Resource allocation
    partition_preference='auto',  # 'auto', 'fat', 'thin', or 'cpu'
    max_concurrent_jobs=50,  # Throttle job arrays
    time_multiplier=1.0,  # Scale time estimates (e.g., 1.5 for extra safety)

    # Paths (adjust for your setup)
    python_env='/proj/berzelius-aiics-real/users/x_learo/mambaforge/envs/pyroenv/bin/python',
    bayesdream_path='/proj/berzelius-aiics-real/users/x_learo/bayesDREAM',
    data_path=data_path,

    # Fitting parameters
    nsamples=1000,
)

# Generate all scripts
generator.generate_all_scripts()

print("\n" + "="*70)
print("SLURM scripts generated successfully!")
print("="*70)
print("\nNext steps:")
print("1. Transfer scripts to Berzelius:")
print("   scp -r slurm_jobs/ berzelius:/path/to/your/work/directory/")
print("")
print("2. On Berzelius, submit jobs:")
print("   cd slurm_jobs")
print("   bash submit_all.sh")
print("")
print("3. Monitor progress:")
print("   squeue -u $USER")
print("   tail -f logs/tech_*.out")

# =============================================================================
# Alternative: Customize for high MOI experiments
# =============================================================================

# Example: High MOI with all-cells technical fitting
if False:  # Set to True to generate
    generator_highmoi = SlurmJobGenerator(
        meta=meta,
        counts=counts,
        cis_genes=cis_genes,
        output_dir='./slurm_jobs_highmoi',
        label='perturb_seq_highmoi',

        low_moi=False,  # High MOI mode
        use_all_cells_technical=True,  # Use all cells for technical fit

        # Other parameters same as above
        python_env='/proj/berzelius-aiics-real/users/x_learo/mambaforge/envs/pyroenv/bin/python',
        bayesdream_path='/proj/berzelius-aiics-real/users/x_learo/bayesDREAM',
        data_path=data_path,
    )

    generator_highmoi.generate_all_scripts()
    print("\nHigh MOI scripts generated in: ./slurm_jobs_highmoi/")

# =============================================================================
# Alternative: Force specific resource allocation
# =============================================================================

# Example: Force CPU for testing
if False:  # Set to True to generate
    generator_cpu = SlurmJobGenerator(
        meta=meta,
        counts=counts,
        cis_genes=cis_genes[:2],  # Test with 2 genes
        output_dir='./slurm_jobs_cpu',
        label='perturb_seq_cpu_test',

        partition_preference='cpu',  # Force CPU
        time_multiplier=2.0,  # 2× time for CPU (slower)

        python_env='/proj/berzelius-aiics-real/users/x_learo/mambaforge/envs/pyroenv/bin/python',
        bayesdream_path='/proj/berzelius-aiics-real/users/x_learo/bayesDREAM',
        data_path=data_path,
    )

    generator_cpu.generate_all_scripts()
    print("\nCPU test scripts generated in: ./slurm_jobs_cpu/")
