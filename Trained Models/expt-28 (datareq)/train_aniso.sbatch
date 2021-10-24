#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --mem=96GB
#SBATCH -p nvidia
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=aniso_datareq
#SBATCH --array=9-10
. ~/.bashrc
conda activate tensorflow2020
cd /scratch/btt1/tse_cnn_19/expt-28
python Train_TSECNN_anisoModel-Slurm.py $SLURM_ARRAY_TASK_ID