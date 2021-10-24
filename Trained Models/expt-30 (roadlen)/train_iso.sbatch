#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=6:00:00
#SBATCH --mem=96GB
#SBATCH -p nvidia
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=roadlen2
#SBATCH --array=3-4
. ~/.bashrc
conda activate tensorflow2020
cd /scratch/btt1/tse_cnn_19/expt-30
python Train_TSECNN_isoModel-Slurm-$SLURM_ARRAY_TASK_ID.py