# bayesDREAM Installation Guide

This guide provides instructions for installing bayesDREAM on different computing environments, including HPC clusters.

## Quick Start (Recommended)

### Using Conda (Recommended for most users)

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

### For CUDA 11.8

Edit `environment.yml` and replace the PyTorch lines with:
```yaml
- pytorch>=2.2.0
- pytorch::pytorch-cuda=11.8
```

Or for pip installation:
```bash
pip install torch>=2.2.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```

### For CUDA 12.1

Edit `environment.yml` and replace the PyTorch lines with:
```yaml
- pytorch>=2.2.0
- pytorch::pytorch-cuda=12.1
```

Or for pip installation:
```bash
pip install torch>=2.2.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

### Check your CUDA version

```bash
nvcc --version
# or
nvidia-smi
```

## HPC Cluster Installation

### SLURM Clusters

1. **Load required modules (if applicable):**
   ```bash
   module load anaconda3
   # or
   module load python/3.12
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

4. **In your SLURM job script:**
   ```bash
   #!/bin/bash
   #SBATCH --job-name=bayesdream
   #SBATCH --nodes=1
   #SBATCH --ntasks=1
   #SBATCH --cpus-per-task=8
   #SBATCH --mem=32G
   #SBATCH --time=24:00:00

   # Load modules if needed
   module load anaconda3

   # Activate environment
   conda activate bayesdream

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
- Python ≥ 3.10 (3.12 recommended)
- NumPy ≥ 1.26.0
- SciPy ≥ 1.16.0
- pandas ≥ 2.3.0
- PyTorch ≥ 2.2.0
- Pyro ≥ 1.9.0
- matplotlib ≥ 3.10.0
- seaborn ≥ 0.13.0
- h5py ≥ 3.14.0

### For Preprocessing (included in conda environment)
- R ≥ 4.0
- Bioconductor scran ≥ 1.20 (for calculating sum factors)

### For Interactive Analysis (included in conda environment)
- JupyterLab ≥ 4.0
- ipykernel (Python kernel for Jupyter)
- IRkernel (R kernel for Jupyter)
- notebook (classic Jupyter notebook)

**Note**: If using the conda environment (recommended), R, scran, and Jupyter components are installed automatically. After installation, register the R kernel with: `R -e "IRkernel::installspec(user = FALSE)"`

### Optional
- pytest ≥ 7.0.0 (for running tests)

## Troubleshooting

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
