# Environment Setup Quick Reference

**See [INSTALLATION.md](INSTALLATION.md) for the complete installation guide.**

## Which Environment File Should I Use?

| Environment File | Use For |
|-----------------|---------|
| `environment_cpu.yml` | CPU-only (local dev, login nodes, M1/M2 Mac) |
| `environment_cuda.yml` | NVIDIA GPUs (CUDA 12.1 default) |
| `environment_rocm.yml` | AMD GPUs (ROCm 6.0 default) |

## Quick Install

```bash
# Choose one:
conda env create -f environment_cpu.yml      # CPU-only
conda env create -f environment_cuda.yml     # NVIDIA GPU
conda env create -f environment_rocm.yml     # AMD GPU

# Activate (adjust name to match your choice)
conda activate bayesdream_cpu   # or bayesdream_cuda, bayesdream_rocm

# Install bayesDREAM
pip install -e .
```

## Check Your Hardware

```bash
# NVIDIA GPU
nvidia-smi

# AMD GPU
rocm-smi --version

# Verify after installation
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"
```

## Customizing Versions

Edit the environment file before creating:

**For CUDA 11.8 (older clusters):**
```yaml
# In environment_cuda.yml, change:
- pytorch::pytorch-cuda=12.1
# To:
- pytorch::pytorch-cuda=11.8
```

**For ROCm 5.7 (older AMD clusters):**
```yaml
# In environment_rocm.yml, change:
- pytorch::pytorch-rocm=6.0
# To:
- pytorch::pytorch-rocm=5.7
```

---

**For detailed instructions, troubleshooting, and HPC cluster setup, see [INSTALLATION.md](INSTALLATION.md).**
