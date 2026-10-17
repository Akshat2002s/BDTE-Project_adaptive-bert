#!/bin/bash
# Fine-tune one static baseline on DAS-5.
# UNTESTED template: every run in this project was trained locally (see README).
#   sbatch slurm/train_baseline.sbatch bert-base
#   sbatch slurm/train_baseline.sbatch distilbert
#SBATCH --job-name=abert-baseline
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
NAME=${1:?usage: sbatch slurm/train_baseline.sbatch <baseline-name>}

# Adapt to how the venv is set up on the cluster (module load ... / CUDA path):
source "$HOME/venvs/adaptive-bert/bin/activate"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
nvidia-smi
python scripts/train_baseline.py "$NAME"
