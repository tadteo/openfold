#!/bin/bash

#SBATCH -A berzelius-2024-220
#SBATCH --gpus 2
#SBATCH -t 00:30:00
#SBATCH -C thin
#SBATCH --output=/proj/berzelius-2021-29/users/x_matta/logs/openfold/run_%j_%A_%a.out
#SBATCH --error=/proj/berzelius-2021-29/users/x_matta/logs/openfold/run_%j_%A_%a.err
#SBATCH --array=1-1



module load Mambaforge/23.3.1-1-hpc1-bdist
module load buildenv-gcccuda/12.1.1-gcc12.3.0
mamba activate /home/x_matta/.conda/envs/openfold-env

export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6+PTX"
export CUDA_LAUNCH_BLOCKING=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python3 -m unittest "$@" || \
echo -e "\nTest(s) failed. Make sure you've installed all Python dependencies."
