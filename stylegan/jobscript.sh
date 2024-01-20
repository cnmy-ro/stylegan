#!/bin/bash

#SBATCH --job-name=genarch
#SBATCH --output=/exports/lkeb-hpc/csrao/git-personal/stylegan/stylegan/training_output/shark-%j.log

# Compute and memory
#SBATCH --partition=LKEBgpu
# SBATCH --gres=gpu
#SBATCH --gres=gpu:RTX6000:1
# SBATCH --nodelist=res-hpc-lkeb05
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# Time
#SBATCH --time=3-00:00:00

# Load/add required modules
module purge
module load library/cuda/11.3/gcc.8.3.1

# Debugging info
hostname
echo "Cuda devices: $CUDA_VISIBLE_DEVICES"
nvidia-smi
echo

# Run python script
python_interpreter=/exports/lkeb-hpc/csrao/miniconda3/envs/mri/bin/python3
$python_interpreter train.py