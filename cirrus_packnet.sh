#!/bin/bash
# Slurm job options (name, compute nodes, job time)

#SBATCH --job-name=PackNet
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4

# We use the "standard" partition as we are running on CPU nodes
#SBATCH --partition=gpu-cascade
# We use the "standard" QoS as our runtime is less than 4 days
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1

# Load the default environment
#module load nvidia/cuda-11.2
source ~/.bashrc
conda activate KM

# Change to the target directory
#cd ~/proj/Kernel_market

# Set the number of threads to 1
#   This prevents any threaded system libraries from automatically
#   using threading.
export OMP_NUM_THREADS=1
sh PackNet_cifar100.sh