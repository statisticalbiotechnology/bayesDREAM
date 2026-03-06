# Snakemake: bayesDREAM Technical + Cis Pipeline

This workflow runs the Typer CLI stages in order:

1. `fit-technical`
2. `fit-cis`

It supports:
- container execution via Apptainer (including ORAS refs)
- local execution via your installed Python environment
- default GPU container usage when NVIDIA (`nvidia-smi`) or AMD (`rocm-smi`) is detected

## Files

- `examples/snakemake/Snakefile`
- `examples/snakemake/config.yaml`
- `examples/gfi1b_cli_config_smk.yaml` (Typer CLI config used by the workflow)

## Run

From repo root:

```bash
snakemake -s examples/snakemake/Snakefile \
  --configfile examples/snakemake/config.yaml \
  -j 1
```

Dry run:

```bash
snakemake -s examples/snakemake/Snakefile \
  --configfile examples/snakemake/config.yaml \
  -n
```

## Execution modes

Set in `examples/snakemake/config.yaml`:

- `execution.mode: auto` (default)
  - uses container if `apptainer` is available, otherwise local Python
- `execution.mode: container`
  - always container
- `execution.mode: local`
  - always local Python (`execution.local_python`)

## GPU behavior

With `execution.mode` resolving to container and `execution.prefer_gpu: true`:

- if `nvidia-smi` detects a GPU:
  - `execution.image_gpu`
  - `apptainer exec --nv ...`
- else if `rocm-smi` detects a GPU:
  - `execution.image_rocm`
  - `apptainer exec --rocm ...`
- otherwise it uses `execution.image_cpu`

## Expected outputs

Under `output/<label>/` from your bayesDREAM config:

- `alpha_x_prefit.pt`
- `posterior_samples_technical_gene.pt`
- `x_true.pt`
- `posterior_samples_cis.pt`
- logs:
  - `.snakemake_fit_technical.log`
  - `.snakemake_fit_cis.log`
