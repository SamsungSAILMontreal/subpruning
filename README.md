# subpruning

Code to reproduce results of the paper "Data-Efficient Structured Pruning via Submodular Optimization" 

# Installation

To install dependencies:

cd subpruning/shrinkbench
## Create a python virtualenv or conda env and activate it
## With conda
conda install --file requirements.txt -y

## With pip
pip install -r requirements.txt 

##  add the path to subpruning folder to your `PYTHONPATH`.  For example:
export PYTHONPATH="$PYTHONPATH:$HOME/subpruning/"

To install the submodular maximization package from https://github.com/sbuschjaeger/SubmodularStreamingMaximization (code already included in SubmodularStreamingMaximization folder)
cd ../SubmodularStreamingMaximization/
pip install --upgrade pip
pip install --upgrade setuptools
pip install cmake
pip install -e .

# Test installation

cd subpruning/shrinkbench/scripts
python test.py

# To reproduce results in the paper

- move pretrained and data folders inside subpruning/shrinkbench and state_dicts folder inside subpruning/shrinkbench/models/cifar10_models
- run one layer pruning script (used for perlayer budget selection): subpruning/shrinkbench/scripts/lay_pruning_job.sh 
- run script for multiple layers pruning (to reproduce all results in the paper): subpruning/shrinkbench/scripts/pruning_job.sh 
- plot results using subpruning/shrinkbench/jupyter/visualize_results.ipynb notebook

Note: make sure to adapt the PYTHONPATH in both lay_pruning_job.sh and pruning_job.sh, and the one layer pruning job ids in pruning_job.sh

# To cite our paper


# Acknowledgements

- Our code builds on the open source ShrinkBench library: https://github.com/JJGO/shrinkbench
- We use the Greedy algorithm code from https://github.com/sbuschjaeger/SubmodularStreamingMaximization, with some modifications
- We use the implementation of VGG11 provided in https://github.com/huyvnphan/PyTorch_CIFAR10/tree/v3.0.1
- We use a progress bar from https://github.com/gipert/progressbar
