#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=6:00:00
#SBATCH --mem=96GB
#SBATCH -p nvidia
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=tse_cnn
#SBATCH --mail-user=btt1@nyu.edu
#SBATCH --array=3
. ~/.bashrc
conda activate tf-gpu
cd /scratch/btt1/tse_cnn_19/expt-23
python Train_TSE_ImprovedCNN-$SLURM_ARRAY_TASK_ID-Slurm.py
