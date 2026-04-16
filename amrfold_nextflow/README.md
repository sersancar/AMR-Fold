# AMR-Fold Nextflow Pipeline

Portable AMR-Fold inference pipeline with:

- local CPU via `conda`
- local GPU via `conda_gpu`
- SLURM CPU via `slurm`
- SLURM GPU via `slurm_gpu`
- optional **Apptainer/Singularity** support for `slurm` and `slurm_gpu`

## Runtime behavior

### Local profiles
- `conda`: CPU with `envs/amrfold_infer_cpu.yml`
- `conda_gpu`: GPU with `envs/amrfold_infer_gpu.yml`

### SLURM profiles
If no container is given:
- `slurm` falls back to `envs/amrfold_infer_cpu.yml`
- `slurm_gpu` falls back to `envs/amrfold_infer_gpu.yml`

If a container is given:
- `slurm` uses `--container_cpu /absolute/path/to/amrfold_cpu.sif`
- `slurm_gpu` uses `--container_gpu /absolute/path/to/amrfold_gpu.sif`

## Important container note

Use **absolute paths** for `.sif` files.

Relative paths may be interpreted as remote image names.

## SLURM GPU with Apptainer/Singularity

```bash

nextflow run main.nf -resume -profile slurm_gpu \
  --input test/test_500.faa.gz \
  --models_dir models \
  --outdir /data3/ssanchez/amr_fold/amrfold_nextflow/results \
  --prostt5_dir /data3/ssanchez/amr_fold/amrfold_nextflow/models/ProstT5 \
  --local_files_only true \
  --container_gpu /data3/ssanchez/amr_fold/amrfold_nextflow/containers/amrfold_gpu.sif \
  --slurm_partition normal \
  --slurm_gpu_partition gpu
```

## SLURM CPU with Apptainer/Singularity

```bash

nextflow run main.nf -resume -profile slurm \
  --input test/test_50.faa.gz \
  --models_dir models \
  --outdir /data3/ssanchez/amr_fold/amrfold_nextflow/results \
  --local_files_only true \
  --container_cpu /data3/ssanchez/amr_fold/amrfold_nextflow/containers/amrfold_cpu.sif \
  --slurm_partition normal
```

## Optional Singularity settings

- `--singularity_cache_dir`
- `--singularity_run_options`
- `--singularity_auto_mounts`

Example:

```bash
nextflow run main.nf -profile slurm_gpu \
  --input test/test_500.faa.gz \
  --models_dir models \
  --outdir results \
  --container_gpu /data3/ssanchez/amr_fold/amrfold_nextflow/containers/amrfold_gpu.sif \
  --singularity_cache_dir /data4/ssanchez/.singularity-cache \
  --slurm_partition normal \
  --slurm_gpu_partition gpu \
  --gpus_per_task 1 \
  --local_files_only true
```

## Large artifacts (models and containers)

Large files are stored on the Hugging Face Hub dataset
`sersancar/amr-fold-artifacts` and are **not** tracked in this repo.

To download all required artifacts after cloning:

```bash
bash scripts/download_artifacts.sh
```

## ProstT5 resolution order

The pipeline resolves ProstT5 in this order:

1. `--prostt5_dir`
2. `${models_dir}/ProstT5`
3. `Rostlab/ProstT5`

For offline runs:

```bash
--models_dir models --local_files_only true
```

## Notes

- `light` jobs do not use the GPU container
- only `extract` and `score` use the GPU container in `slurm_gpu`
- GPU container jobs add `--nv`
