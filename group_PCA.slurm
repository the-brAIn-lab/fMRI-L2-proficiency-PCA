#!/bin/bash
#SBATCH --job-name=groupPCA_L1Ln_K20
#SBATCH --output=/home/x-onarayanamud/NewPCA/logs/groupPCA_%j.out
#SBATCH --error=/home/x-onarayanamud/NewPCA/logs/groupPCA_%j.err

#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

module purge
module load python/3.9.5

# Cache control (from your older script)
export PIP_CACHE_DIR=/anvil/scratch/x-onarayanamud/pip_cache

# ✅ Your environment (from your older script)
source /anvil/scratch/x-onarayanamud/pca_env39/bin/activate

# Create output dirs
mkdir -p /home/x-onarayanamud/NewPCA/logs
mkdir -p /home/x-onarayanamud/NewPCA/results/result1

# Parallel BLAS threading = your speedup
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Using python: $(which python)"
python -V
echo "OMP_NUM_THREADS = $OMP_NUM_THREADS"

PYTHON_SCRIPT=/home/x-onarayanamud/NewPCA/group_PCA.py

echo "Running: $PYTHON_SCRIPT"
python "$PYTHON_SCRIPT" \
  --n_components 20 \
  --batch_size 200 \
  --base_root /anvil/scratch/x-onarayanamud/PreprosessedData \
  --out_root /home/x-onarayanamud/NewPCA/results/result1 \
  --tasks compL1,compLn \
  --center

# Optional:
# add --zscore if needed

echo "End time: $(date)"
