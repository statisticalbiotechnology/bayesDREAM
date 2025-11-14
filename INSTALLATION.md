# bayesDREAM Installation Guide

This guide provides instructions for installing bayesDREAM on different computing environments, including HPC clusters.

## Quick Start (Recommended)

### Using Conda (Recommended for most users)

**Note**: The default conda environment includes **GPU support with CUDA 12.1**, which works on most modern GPU clusters. If you need CPU-only installation, see the GPU Support section below.

1. **Create the conda environment:**
   ```bash
   conda env create -f environment.yml
   ```

2. **Activate the environment:**
   ```bash
   conda activate bayesdream
   ```

3. **Install bayesDREAM in development mode:**
   ```bash
   pip install -e .
   ```

4. **Register R kernel with Jupyter (for using R in notebooks):**
   ```bash
   R -e "IRkernel::installspec(user = FALSE)"
   ```

5. **Verify GPU support (if using GPU cluster):**
   ```bash
   python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
   ```

### Using pip only

1. **Create a virtual environment (optional but recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install bayesDREAM:**
   ```bash
   pip install -e .
   ```

## GPU Support

**Default**: The conda environment (`environment.yml`) installs PyTorch with **CUDA 12.1 support** by default. This works with CUDA 12.x on most modern GPU clusters (including CUDA 12.2+).

### Verify GPU is Working

After installation, verify CUDA is available:
```bash
conda activate bayesdream
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"
```

You should see:
```
CUDA available: True
CUDA version: 12.1
```

### Check Your Cluster's CUDA Version

```bash
nvidia-smi  # Shows CUDA driver version (e.g., 12.2)
nvcc --version  # Shows CUDA toolkit version (if installed)
```

**Note**: PyTorch CUDA 12.1 is compatible with CUDA 12.2+ drivers.

### For Older GPU Clusters (CUDA 11.8)

If your cluster only has CUDA 11.x, edit `environment.yml` and change:
```yaml
- pytorch::pytorch-cuda=12.1
```
to:
```yaml
- pytorch::pytorch-cuda=11.8
```

Or reinstall with pip:
```bash
pip uninstall torch -y
pip install torch>=2.2.0 --index-url https://download.pytorch.org/whl/cu118
```

### For CPU-Only Installation

If you don't have a GPU or want CPU-only for local development:

**Option 1: Edit environment.yml before creating environment**
```yaml
# Comment out:
# - pytorch::pytorch-cuda=12.1

# Uncomment:
- pytorch::cpuonly
```

**Option 2: Reinstall PyTorch as CPU-only after creating environment**
```bash
conda activate bayesdream
pip uninstall torch -y
conda install pytorch cpuonly -c pytorch
```

## HPC Cluster Installation

### SLURM Clusters (with GPU)

1. **Load required modules (if applicable):**
   ```bash
   module load anaconda3
   # or
   module load python/3.11
   ```

2. **Create environment:**
   ```bash
   conda env create -f environment.yml
   ```

3. **Activate and install:**
   ```bash
   conda activate bayesdream
   pip install -e .
   ```

4. **Verify GPU support:**
   ```bash
   python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
   ```

5. **In your SLURM job script (GPU job):**
   ```bash
   #!/bin/bash
   #SBATCH --job-name=bayesdream
   #SBATCH --nodes=1
   #SBATCH --ntasks=1
   #SBATCH --cpus-per-task=8
   #SBATCH --mem=32G
   #SBATCH --gres=gpu:1          # Request 1 GPU
   #SBATCH --time=24:00:00

   # Load modules if needed
   module load anaconda3

   # Activate environment
   conda activate bayesdream

   # Verify GPU is visible
   nvidia-smi
   echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

   # Run your script
   python your_script.py
   ```

### PBS/Torque Clusters

Similar to SLURM, but use PBS directives:
```bash
#!/bin/bash
#PBS -N bayesdream
#PBS -l nodes=1:ppn=8
#PBS -l mem=32gb
#PBS -l walltime=24:00:00

cd $PBS_O_WORKDIR
conda activate bayesdream
python your_script.py
```

## Verification

Test your installation:
```bash
python -c "from bayesDREAM import bayesDREAM; print('✓ bayesDREAM imported successfully')"
```

Run the test suite:
```bash
cd tests
python test_trans_quick.py
```

Start JupyterLab for interactive analysis:
```bash
jupyter lab
# Or for classic notebook interface:
jupyter notebook
```

Verify R kernel is available in Jupyter:
```bash
jupyter kernelspec list
# Should show both 'python3' and 'ir' kernels
```

## Dependencies

### Required
- Python 3.11 (recommended for best package compatibility)
- NumPy ≥ 1.24.0
- SciPy ≥ 1.10.0
- pandas ≥ 2.0.0
- scikit-learn ≥ 1.3.0
- PyTorch ≥ 2.2.0
- Pyro ≥ 1.9.0
- matplotlib ≥ 3.7
- seaborn ≥ 0.12
- h5py ≥ 3.8.0

**Note**: Version constraints have been relaxed for better cross-platform compatibility. The conda environment uses Python 3.11 (better package support than 3.12 on Linux clusters).

### For Preprocessing (included in conda environment)
- R ≥ 4.0
- Bioconductor scran ≥ 1.20 (for calculating sum factors)
- r-data.table (for fast data manipulation in R)

### For Interactive Analysis (included in conda environment)
- JupyterLab ≥ 4.0
- ipykernel (Python kernel for Jupyter)
- IRkernel (R kernel for Jupyter)
- notebook (classic Jupyter notebook)

**Note**: If using the conda environment (recommended), R, scran, and Jupyter components are installed automatically. After installation, register the R kernel with: `R -e "IRkernel::installspec(user = FALSE)"`

### Optional
- pytest ≥ 7.0.0 (for running tests)

## Troubleshooting

### Conda environment creation fails (LibMambaUnsatisfiableError)

If you get dependency conflicts when creating the environment:

**Option 1: Use mamba (faster and better at solving)**
```bash
conda install mamba -c conda-forge
mamba env create -f environment.yml
```

**Option 2: Try with fewer channels**
Edit `environment.yml` and comment out the pytorch channel:
```yaml
channels:
  # - pytorch  # Comment this out
  - conda-forge
  - bioconda
  - defaults
```
Then create the environment again.

**Option 3: Create minimal environment and add packages**
```bash
# Create minimal environment
conda create -n bayesdream python=3.11 numpy scipy pandas scikit-learn pytorch -c conda-forge

# Activate it
conda activate bayesdream

# Install Pyro via pip
pip install pyro-ppl

# Add R and bioconda packages
conda install r-base bioconductor-scran r-irkernel r-data.table -c bioconda -c conda-forge

# Add Jupyter and visualization
conda install jupyterlab ipykernel notebook ipywidgets matplotlib seaborn h5py -c conda-forge

# Register R kernel
R -e "IRkernel::installspec(user = FALSE)"

# Install bayesDREAM
cd /path/to/bayesDREAM_forClaude
pip install -e .
```

### Import errors
If you see `ModuleNotFoundError: No module named 'bayesDREAM'`:
- Make sure you ran `pip install -e .` from the repository root
- Check that the environment is activated: `conda activate bayesdream`

### PyTorch GPU not detected
```python
import torch
print(torch.cuda.is_available())  # Should be True if GPU is available
print(torch.cuda.device_count())  # Number of GPUs
```

If False:
- Verify CUDA is installed: `nvcc --version`
- Reinstall PyTorch with the correct CUDA version (see GPU Support above)

### Out of memory errors
- Reduce batch sizes in your scripts
- Use CPU instead of GPU for smaller datasets
- Request more memory in your SLURM/PBS script

### Slow performance
- Enable OpenBLAS or MKL for faster linear algebra:
  ```bash
  conda install openblas
  # or
  conda install mkl
  ```
- Use GPU if available
- Increase `--cpus-per-task` in SLURM scripts

## Support

For issues:
1. Check the documentation in `docs/`
2. Review test files in `tests/` for usage examples
3. Check `docs/OUTSTANDING_TASKS.md` for known issues

## Development Installation

For contributing to bayesDREAM:
```bash
conda env create -f environment.yml
conda activate bayesdream
pip install -e ".[dev]"  # Installs development dependencies
```

This installs additional tools:
- pytest and pytest-cov for testing
- black for code formatting
- flake8 for linting
