#!/bin/bash

# Slurm sbatch options
#SBATCH -o outputLogs/pruning_job-%A-%a.out
#SBATCH -a 1-960 #Change according to max njobs allowed on cluster
#SBATCH -c 20
#SBATCH --gres=gpu:volta:1

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "STARTING AT `date`"

source activate venv

export PYTHONPATH="$PYTHONPATH:$HOME/subpruning/"
echo $PYTHONPATH

LAYJOBID_MNIST = 1
LAYJOBID_CIFAR10_RESNET = 2
LAYJOBID_CIFAR10_VGG = 3

# Run experiments of Figure 1 and 3

python prune_channels.py -r 5 -s all -b 4 -d MNIST -m LeNet -l all -o Adam --task $SLURM_ARRAY_TASK_ID --ntasks $SLURM_ARRAY_TASK_COUNT --job $SLURM_ARRAY_JOB_ID --layjob $LAYJOBID_MNIST

python prune_channels.py  -r 5 -s all -b 4 -d CIFAR10 -m resnet56 -l all -o Adam --task $SLURM_ARRAY_TASK_ID --ntasks $SLURM_ARRAY_TASK_COUNT --job $SLURM_ARRAY_JOB_ID --layjob $LAYJOBID_CIFAR10_RESNET

python prune_channels.py  -r 5 -s all -b 4 -d CIFAR10 -m vgg11_bn_small -l all -o Adam --task $SLURM_ARRAY_TASK_ID --ntasks $SLURM_ARRAY_TASK_COUNT --job $SLURM_ARRAY_JOB_ID --layjob $LAYJOBID_CIFAR10_RESNET

# Run experiments of Figure 2

python prune_channels.py -r 5 -s all -b 4 -d MNIST -m LeNet -l all -fd False -ld -o Adam --task $SLURM_ARRAY_TASK_ID --ntasks $SLURM_ARRAY_TASK_COUNT --job $SLURM_ARRAY_JOB_ID --layjob $LAYJOBID_MNIST

python prune_channels.py  -r 5 -s all -b 4 -d CIFAR10 -m resnet56 -l all -fd False -ld -o Adam --task $SLURM_ARRAY_TASK_ID --ntasks $SLURM_ARRAY_TASK_COUNT --job $SLURM_ARRAY_JOB_ID --layjob $LAYJOBID_CIFAR10_RESNET

python prune_channels.py  -r 5 -s all -b 4 -d CIFAR10 -m vgg11_bn_small -l all -fd False -ld -o Adam --task $SLURM_ARRAY_TASK_ID --ntasks $SLURM_ARRAY_TASK_COUNT --job $SLURM_ARRAY_JOB_ID --layjob $LAYJOBID_CIFAR10_RESNET


echo "FINISHING AT `date`"


