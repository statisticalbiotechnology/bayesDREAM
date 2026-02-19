# Apptainer Support

This folder provides a CPU-focused Apptainer recipe for `bayesDREAM`.

## Build

```bash
apptainer build bayesdream_cpu.sif apptainer/bayesdream_cpu.def
```

If your cluster requires rootless builds, use your site's remote builder (if configured):

```bash
apptainer build --remote bayesdream_cpu.sif apptainer/bayesdream_cpu.def
```

## Run bayesDREAM CLI

```bash
apptainer exec --bind $PWD:/work bayesdream_cpu.sif \
  bayesdream run --config /work/config.yaml
```

You can also use run mode (runs `bayesdream` directly):

```bash
apptainer run --bind $PWD:/work bayesdream_cpu.sif --help
```

## Snakemake integration

Use the generated `.sif` directly in rule definitions:

```python
rule bayesdream_pipeline:
    input:
        cfg="config.yaml"
    output:
        done="results/.done"
    container:
        "bayesdream_cpu.sif"
    shell:
        """
        bayesdream run --config {input.cfg}
        touch {output.done}
        """
```

Run with container support enabled:

```bash
snakemake --use-singularity --singularity-args "--bind $PWD:/work"
```

## Notes

- This recipe is CPU-first for portability and reproducibility.
- For GPU execution later, add a CUDA/ROCm-specific definition and run with `apptainer exec --nv` (NVIDIA) or your cluster's ROCm support setup.

## GitHub Actions + GHCR

A workflow is provided at `.github/workflows/apptainer-ghcr.yml` that builds and publishes on each push.

Published refs:

- `ghcr.io/<owner>/<repo>-apptainer:cpu-amd64-sha-<12charsha>`
- `ghcr.io/<owner>/<repo>-apptainer:cpu-arm64-sha-<12charsha>`
- `ghcr.io/<owner>/<repo>-apptainer:cpu-amd64-<branch>`
- `ghcr.io/<owner>/<repo>-apptainer:cpu-arm64-<branch>`
- `ghcr.io/<owner>/<repo>-apptainer:cpu-amd64-latest` (default branch only)
- `ghcr.io/<owner>/<repo>-apptainer:cpu-arm64-latest` (default branch only)
