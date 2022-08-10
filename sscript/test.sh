#!/bin/bash
#SBATCH -t 03:00:00
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint=high-capacity
#SBATCH -p yanglab
#SBATCH -e ./sbatch/slurm-%j.out
#SBATCH -o ./sbatch/slurm-%j.out

echo this_is_test_message

exit 0;