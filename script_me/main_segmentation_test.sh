#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --array=0
#SBATCH -J seg
#SBATCH -o slurm_logs/%x.%3a.%A.out
#SBATCH -e slurm_logs/%x.%3a.%A.err
#SBATCH --time=10:00:00
##SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=40G
#SBATCH --gres=gpu:a100:1


[ ! -d "slurm_logs" ] && echo "Create a directory slurm_logs" && mkdir -p slurm_logs

# module load cuda/11.1.1
# module load gcc

# echo "===> Anaconda env loaded"
# source ~/.bashrc
# source activate openpoints

# nvidia-smi
# nvcc --version

# hostname
# NUM_GPU_AVAILABLE=`nvidia-smi --query-gpu=name --format=csv,noheader | wc -l`
# echo $NUM_GPU_AVAILABLE


cfg=cfgs/radar/hpenet-l.yaml
# PY_ARGS=${@:2}
# echo ${PY_ARGS}
CUDA_VISIBLE_DEVICES=7 python examples/segmentation/main.py --cfg $cfg wandb.use_wandb=False mode=test \
--pretrained_path log/radar/radar-train-hpenet-l-ngpus1-20260515-013127-HXWMALkaAC4GiUWjNV5c3g/checkpoint/radar-train-hpenet-l-ngpus1-20260515-013127-HXWMALkaAC4GiUWjNV5c3g_ckpt_best.pth


# how to run
# using slurm, run with 1 GPU, by 3 times (array=0-2):
# sbatch --array=0-2 --gres=gpu:1 --time=12:00:00 script/main_segmentation.sh cfgs/s3dis/pointnext-s.yaml

# if using local machine with GPUs, run with ALL GPUs:
# bash script/main_segmentation.sh cfgs/s3dis/pointnext-s.yaml

# local machine, run with 1 GPU:
# CUDA_VISIBLE_DEVICES=0 bash script/main_segmentation.sh cfgs/s3dis/pointnext-s.yaml
