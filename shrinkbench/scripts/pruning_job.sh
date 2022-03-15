#!/bin/bash

# Slurm sbatch options
#SBATCH -o outputLogs/pruning_job-%A-%a.out
#SBATCH -a 1-720 #Change according to max njobs allowed on cluster
#SBATCH -c 20
#SBATCH --gres=gpu:volta:1

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "STARTING AT `date`"

source activate venv

export PYTHONPATH="$PYTHONPATH:$HOME/subpruning/"
echo $PYTHONPATH

LAYJOBID_MNIST = 1
LAYJOBID_CIFAR10_NEURON = 2
LAYJOBID_CIFAR10_CHANNEL = 3

# Run experiment of Figure 1, 7, 11 
python prune_channels.py  -r 5 -s all -b 4 -d MNIST -m LeNet -t neuron -l all -o Adam --task $SLURM_ARRAY_TASK_ID --ntasks $SLURM_ARRAY_TASK_COUNT --job $SLURM_ARRAY_JOB_ID --layjob $LAYJOBID_MNIST

# Run experiment of Figure 2, 8, 12
python prune_channels.py  -r 5 -s all -b 4 -d CIFAR10 -m vgg11_bn_small -t neuron -l all -o Adam --task $SLURM_ARRAY_TASK_ID --ntasks $SLURM_ARRAY_TASK_COUNT --job $SLURM_ARRAY_JOB_ID --layjob $LAYJOBID_CIFAR10_NEURON

# Run experiment of Figure 3, 9, 13
python prune_channels.py  -r 5 -s all -b 4 -d CIFAR10 -m vgg11_bn_small -t channel -l  features.0,features.4 -o Adam --task $SLURM_ARRAY_TASK_ID --ntasks $SLURM_ARRAY_TASK_COUNT --job $SLURM_ARRAY_JOB_ID --layjob $LAYJOBID_CIFAR10_CHANNEL

# Run experiment of Figure 4, 10, 15
python prune_channels.py  -r 5 -s all -b 4 -d MNIST -m vgg11_bn_small_mnist -t channel -l  features.0,features.4 -o Adam --task $SLURM_ARRAY_TASK_ID --ntasks $SLURM_ARRAY_TASK_COUNT --job $SLURM_ARRAY_JOB_ID --layjob $LAYJOBID_CIFAR10_CHANNEL

# Run experiment of Figure 5, 14
python prune_channels.py  -r 5 -s all -b 4 -d CIFAR10 -m vgg11_bn_small -t channel -l  features.0,features.8,features.11,features.18 -o Adam --task $SLURM_ARRAY_TASK_ID --ntasks $SLURM_ARRAY_TASK_COUNT --job $SLURM_ARRAY_JOB_ID --layjob $LAYJOBID_CIFAR10_CHANNEL

# Run experiment of Figure 6, 16
python prune_channels.py  -r 5 -s all -b 4 -d MNIST -m vgg11_bn_small_mnist -t channel -l  features.0,features.8,features.11,features.18 -o Adam --task $SLURM_ARRAY_TASK_ID --ntasks $SLURM_ARRAY_TASK_COUNT --job $SLURM_ARRAY_JOB_ID --layjob $LAYJOBID_CIFAR10_CHANNEL

# Run experiment of Figure 17
python prune_channels.py  -r 5 -s all -b 4 -d MNIST -m vgg11_bn_small_mnist -t channel -l  features.0,features.22 -o Adam --task $SLURM_ARRAY_TASK_ID --ntasks $SLURM_ARRAY_TASK_COUNT --job $SLURM_ARRAY_JOB_ID --layjob $LAYJOBID_CIFAR10_CHANNEL

# Run experiment of Figure 18
python prune_channels.py  -r 5 -s all -b 4 -d MNIST -m vgg11_bn_small_mnist -t channel -l  features.0,features.22 -o Adam --task $SLURM_ARRAY_TASK_ID --ntasks $SLURM_ARRAY_TASK_COUNT --job $SLURM_ARRAY_JOB_ID 

echo "FINISHING AT `date`"


