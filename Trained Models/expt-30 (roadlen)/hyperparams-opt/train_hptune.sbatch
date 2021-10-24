#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --mem=96GB
#SBATCH -p nvidia
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=hyperopt2
#SBATCH --mail-user=btt1@nyu.edu
#SBATCH --output=slurm_%j.out
#SBATCH --array=2-4
. ~/.bashrc
conda activate tensorflow2020
cd /scratch/btt1/tse_cnn_19/expt-30
python Train_TSECNN_Hypopt-Slurm-$SLURM_ARRAY_TASK_ID.py >> Hypopt-$SLURM_ARRAY_TASK_ID.out