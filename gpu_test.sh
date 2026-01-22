#!/bin/bash
#SBATCH --job-name=gpu_test
#SBATCH --output=gpu_test_%j.out
#SBATCH --error=gpu_test_%j.err
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00

# Load CUDA modules (usually OK; if there is a conflict we can comment them out)
module load cuda/12.1.1
module load cudnn_for_cuda12/8.9.1

source /data3/ssanchez/miniforge3/etc/profile.d/conda.sh
conda activate prostt5

echo "Node: $SLURM_NODELIST"
echo "=== nvidia-smi ==="
nvidia-smi || echo "nvidia-smi failed"

python - << 'EOF'
import torch
print("torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(0))
    print("torch.version.cuda:", torch.version.cuda)
EOF

