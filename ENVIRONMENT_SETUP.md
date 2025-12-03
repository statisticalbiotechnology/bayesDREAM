# Environment Setup Guide

bayesDREAM provides three conda environment files for different hardware configurations:

## Quick Start

### 1. Choose Your Environment

| Environment File | Use Case | Hardware |
|-----------------|----------|----------|
| `environment_cpu.yml` | CPU-only (no GPU) | Local machines, login nodes, M1/M2 Mac |
| `environment_cuda.yml` | NVIDIA GPUs | Clusters with NVIDIA GPUs (A100, V100, etc.) |
| `environment_rocm.yml` | AMD GPUs | Clusters with AMD GPUs (MI250X, MI300, etc.) |

### 2. Installation

```bash
# Navigate to bayesDREAM directory
cd /path/to/bayesDREAM_forClaude

# Create the appropriate environment
conda env create -f environment_[cpu|cuda|rocm].yml

# Activate the environment
conda activate bayesdream_[cpu|cuda|rocm]

# Install bayesDREAM in development mode
pip install -e .
```

### 3. Verify Installation

```python
# Test basic imports
python -c "import torch; import pyro; import bayesDREAM; print('✓ Installation successful')"

# Check GPU availability (for cuda/rocm environments)
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"
```

## Detailed Instructions

### CPU-Only Environment

**When to use:**
- Local development on machines without GPU
- HPC login nodes (where GPU access is restricted)
- M1/M2 Mac computers (MPS backend not fully supported by Pyro)
- Quick testing and debugging

**Installation:**
```bash
conda env create -f environment_cpu.yml
conda activate bayesdream_cpu
pip install -e .
```

**Usage in bayesDREAM:**
```python
from bayesDREAM import bayesDREAM

model = bayesDREAM(
    meta=meta,
    counts=counts,
    device='cpu'  # Explicitly use CPU
)
```

---

### CUDA Environment (NVIDIA GPUs)

**When to use:**
- HPC clusters with NVIDIA GPUs (A100, V100, H100, etc.)
- Workstations with NVIDIA GPUs
- Cloud instances (AWS p3/p4, GCP A2, Azure NC-series)

**Check your CUDA version:**
```bash
nvcc --version
# OR
nvidia-smi
```

**Installation:**

For CUDA 12.x (most modern clusters):
```bash
conda env create -f environment_cuda.yml
conda activate bayesdream_cuda
pip install -e .
```

For CUDA 11.8 (older clusters):
```bash
# Edit environment_cuda.yml first:
# Change: pytorch::pytorch-cuda=12.1
# To: pytorch::pytorch-cuda=11.8

conda env create -f environment_cuda.yml
conda activate bayesdream_cuda
pip install -e .
```

**Verify GPU access:**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
```

**Usage in bayesDREAM:**
```python
from bayesDREAM import bayesDREAM

model = bayesDREAM(
    meta=meta,
    counts=counts,
    device='cuda'  # Use GPU
)
```

---

### ROCm Environment (AMD GPUs)

**When to use:**
- HPC clusters with AMD GPUs (MI250X, MI300A, MI300X, etc.)
- Clusters that have migrated from NVIDIA to AMD
- AMD-based workstations

**Check your ROCm version:**
```bash
rocm-smi --version
# OR
/opt/rocm/bin/rocminfo
```

**Installation:**

For ROCm 6.0 (latest stable):
```bash
conda env create -f environment_rocm.yml
conda activate bayesdream_rocm
pip install -e .
```

For older ROCm versions (5.7, 5.6, etc.):
```bash
# Edit environment_rocm.yml first:
# Change: pytorch::pytorch-rocm=6.0
# To: pytorch::pytorch-rocm=5.7  (or your version)

conda env create -f environment_rocm.yml
conda activate bayesdream_rocm
pip install -e .
```

**Verify GPU access:**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")  # Uses 'cuda' API for AMD too
print(f"ROCm version: {torch.version.hip}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
```

**Usage in bayesDREAM:**
```python
from bayesDREAM import bayesDREAM

model = bayesDREAM(
    meta=meta,
    counts=counts,
    device='cuda'  # PyTorch uses 'cuda' for both NVIDIA and AMD
)
```

**Important notes for ROCm:**
- ROCm support in PyTorch is generally good but less mature than CUDA
- Most bayesDREAM operations should work fine
- If you encounter issues, fall back to `environment_cpu.yml`
- Some advanced PyTorch features may have compatibility gaps

---

## Troubleshooting

### Environment Creation Fails

If conda takes too long or fails to resolve dependencies:

```bash
# Install mamba (faster solver)
conda install mamba -c conda-forge

# Use mamba instead
mamba env create -f environment_[cpu|cuda|rocm].yml
```

### Wrong GPU Type Detected

If your cluster changed GPU types (e.g., NVIDIA → AMD):

1. Remove old environment:
   ```bash
   conda remove -n bayesdream_cuda --all
   ```

2. Create new environment with correct GPU type:
   ```bash
   conda env create -f environment_rocm.yml
   ```

### Import Hangs on Cluster

This usually happens when using the wrong PyTorch build:

1. Check what GPU you have:
   ```bash
   nvidia-smi  # NVIDIA
   rocm-smi    # AMD
   ```

2. Remove and recreate environment with correct file

3. Or force CPU-only if on login node:
   ```bash
   export CUDA_VISIBLE_DEVICES=''
   python your_script.py
   ```

### M1/M2 Mac Issues

PyTorch's MPS backend (Apple Silicon GPU) is not fully compatible with Pyro. Use CPU-only:

```bash
conda env create -f environment_cpu.yml
```

---

## Cluster-Specific Notes

### If Your Cluster Provides PyTorch Modules

Some clusters provide optimized PyTorch through environment modules. If available:

```bash
# Load cluster-provided PyTorch
module load pytorch/2.2.0

# Create minimal environment without PyTorch
conda create -n bayesdream_cluster python=3.11 numpy scipy pandas scikit-learn pip
conda activate bayesdream_cluster

# Install remaining dependencies
pip install pyro-ppl matplotlib seaborn h5py

# Install bayesDREAM
cd /path/to/bayesDREAM_forClaude
pip install -e .
```

### Threading Environment Variables

If PyTorch hangs during import, set threading limits:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python your_script.py
```

Or add to your job script:
```bash
#!/bin/bash
#SBATCH --cpus-per-task=4

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

conda activate bayesdream_rocm
python run_pipeline/run_technical.py --cores $SLURM_CPUS_PER_TASK
```

---

## Legacy Environment Files

- `environment.yml`: Original file with CUDA 12.1 as default
- `environment_mac.yml`: Mac-specific configuration (if present)

These are kept for backward compatibility but the new architecture-specific files are recommended.

---

## Quick Reference

```bash
# CPU-only (safe default)
conda env create -f environment_cpu.yml

# NVIDIA GPU
conda env create -f environment_cuda.yml

# AMD GPU
conda env create -f environment_rocm.yml
```

After installation:
```bash
conda activate bayesdream_[cpu|cuda|rocm]
pip install -e .
```
