#!/bin/bash
#SBATCH --job-name=getFeatures
#SBATCH --output=/home/ssanchez/amr_fold/logs/%x_%j.out
#SBATCH --error=/home/ssanchez/amr_fold/logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=250:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sergio.sanchezcarrillo@universityofgalway.ie

# Load modules
module load cuda/12.1.1
module load cudnn_for_cuda12/8.9.1

# Conda
source /data3/ssanchez/miniforge3/etc/profile.d/conda.sh
conda activate prostt5

echo "Job started on $(date)"
echo "Running on node(s): $SLURM_NODELIST"
echo "Current working directory: $(pwd)"

# Directories
INPUTDIR=/data3/ssanchez/amr_fold/data
OUTPUTDIR=/data3/ssanchez/amr_fold/features       # or /features/prostt5_features if you prefer
HF_HOME=/data3/ssanchez/amr_fold/hf_cache
CODEDIR=/home/ssanchez/amr_fold

# Create dirs if needed
mkdir -p "$OUTPUTDIR"
mkdir -p "$HF_HOME"

# Export so they are visible inside Python
export INPUTDIR OUTPUTDIR HF_HOME
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Make sure Python can find features.py in CODEDIR (if it lives there)
export PYTHONPATH="$CODEDIR:$PYTHONPATH"

# Use env vars inside Python (heredoc single-quoted to avoid shell interpolation)
python - << 'EOF'
import os
from features import ProstT5FeatureExtractor, ProstT5Config

input_dir = os.environ["INPUTDIR"]
output_dir = os.environ["OUTPUTDIR"]

# For testing small subset:
#fasta_name = "DB_200.faa.gz"
# Full DB:
fasta_name = "DB.faa.gz"

input_fasta = os.path.join(input_dir, fasta_name)
out_dir = output_dir

cfg = ProstT5Config(
    device="cuda",
    half_precision=True,
)
extractor = ProstT5FeatureExtractor(cfg)
extractor.encode_fasta(
    fasta_path=input_fasta,
    out_dir=out_dir,
    overwrite=False,
    progress=True,
)
EOF

echo "Job finished on $(date)"

# RUN:  sbatch getFeatures.sh

