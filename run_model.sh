#!/bin/bash
#SBATCH --account=def-ctetsass
# For staff use cc_debug
##SBATCH --partition=debug # Must have
#SBATCH --time=00-23:05
#SBATCH --mem=120G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=h100:1 # Must have, equivalent to --gres=gpu:h100:1
module load StdEnv/2023
module load cuda
module load python/3.12
source /home/advaitgore/PycharmProjects/GenDynDiff/.venv/bin/activate
python /home/advaitgore/PycharmProjects/GenDynDiff/gendyndiff/run.py
