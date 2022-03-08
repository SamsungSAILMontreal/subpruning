#!/bin/bash

# Slurm sbatch options
#SBATCH -o outputLogs/pruning_job-%A-%a.out
#SBATCH -a 1-1408 #Change according to max njobs allowed on cluster
#SBATCH -c 20
#SBATCH --gres=gpu:volta:1

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "STARTING AT `date`"

source activate venv

export PYTHONPATH="$PYTHONPATH:$HOME/subpruning/"
echo $PYTHONPATH

# run one layer pruning experiments needed for perlayer budget selection

python prune_onelayer.py -s all -b 4 -d MNIST -m LeNet -t neuron -o Adam --task $SLURM_ARRAY_TASK_ID --ntasks $SLURM_ARRAY_TASK_COUNT --job $SLURM_ARRAY_JOB_ID

python prune_onelayer.py -s all -b 4 -d CIFAR10 -m vgg11_bn_small -t neuron -o Adam --task $SLURM_ARRAY_TASK_ID --ntasks $SLURM_ARRAY_TASK_COUNT --job $SLURM_ARRAY_JOB_ID

python prune_onelayer.py -s all -b 4 -d CIFAR10 -m vgg11_bn_small -t channel -o Adam --task $SLURM_ARRAY_TASK_ID --ntasks $SLURM_ARRAY_TASK_COUNT --job $SLURM_ARRAY_JOB_ID

echo "FINISHING AT `date`"
