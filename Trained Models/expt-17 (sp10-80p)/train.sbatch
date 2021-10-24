#!/bin/bash
#SBATCH --nodes=4
#SBATCH --time=6:00:00
#SBATCH --mem=32GB
#SBATCH -p nvidia
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=tse_cnn
#SBATCH --mail-user=btt1@nyu.edu
#SBATCH --array=1-8
. ~/.bashrc
conda activate tf-gpu
cd /scratch/btt1/tse_cnn_19/expt-17
python Train_TSE_ImprovedCNN-Slurm-sp$SLURM_ARRAY_TASK_ID.py
