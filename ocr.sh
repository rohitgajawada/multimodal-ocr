#!/bin/bash
#SBATCH -A rohit.gajawada
#SBATCH --gres=gpu:1
#SBATCH -n 20
#SBATCH --mincpus=24
#SBATCH --mem-per-cpu=2048
#SBATCH --time=72:00:00
#SBATCH --mail-type=ALL

python crnn_main.py --random_sample --trainroot='../train/' --valroot='../val/' --cuda --adadelta --batchSize=64 
