# Data-Efficient Structured Pruning via Submodular Optimization

Code to reproduce results of the paper [Data-Efficient Structured Pruning via Submodular Optimization](https://arxiv.org/abs/2203.04940)

## Installation

To install dependencies:

```bash
# Create a python virtualenv or conda env and activate it
# With conda
conda install --file requirements.txt -y

# With pip
pip install -r requirements.txt 

#  add the path to subpruning folder to your `PYTHONPATH`.  For example:
export PYTHONPATH="$PYTHONPATH:$HOME/subpruning/"
```

To install the submodular maximization package from https://github.com/sbuschjaeger/SubmodularStreamingMaximization (code already included in SubmodularStreamingMaximization folder)

```bash
cd subpruning/SubmodularStreamingMaximization/
pip install --upgrade pip
pip install --upgrade setuptools
pip install cmake
pip install -e .
```

Download data from https://drive.google.com/drive/folders/1ae-NCkGhu6gX3AwnCrytMQftywd62kyz and place data folder inside subpruning/shrinkbench folder 

Download pretrained lenet and resnet56 models from https://drive.google.com/drive/folders/16SsMrq_qp2CYgfbIU4rpkKpfV0DyVXzH?usp=sharing and place pretrained folder inside subpruning/shrinkbench folder 

Download pretrained vgg11 model from https://drive.google.com/drive/folders/16CcqDpfNkE046pfXbe_d1UE6ZpeaRRMl?usp=sharing and place state_dicts folder inside subpruning/shrinkbench/models/cifar10_models folder

# Test installation

```bash
cd subpruning/shrinkbench/scripts
python test.py
```

# To reproduce results in the paper

- run one layer pruning script (used for perlayer budget selection): subpruning/shrinkbench/scripts/lay_pruning_job.sh 
- run script for multiple layers pruning (to reproduce all results in the paper): subpruning/shrinkbench/scripts/pruning_job.sh 
- plot results using subpruning/shrinkbench/jupyter/visualize_results.ipynb notebook

Note: make sure to adapt the PYTHONPATH in both lay_pruning_job.sh and pruning_job.sh, and the one layer pruning job ids in pruning_job.sh

# To cite our paper
```
@InProceedings{elhalabi2022dataefficient,
      title={Data-Efficient Structured Pruning via Submodular Optimization}, 
      author={Marwa El Halabi and Suraj Srinivas and Simon Lacoste-Julien},
      booktitle = {Advances in Neural Information Processing Systems},
      year={2022},
}
```
# Acknowledgements

- Our code builds on the open source ShrinkBench library: https://github.com/JJGO/shrinkbench
- We use the Greedy algorithm code from https://github.com/sbuschjaeger/SubmodularStreamingMaximization, with some modifications
- Our implementation of the LayerSampling pruning method is adapted from the original code provided in https://github.com/lucaslie/torchprune
- We use the implementation of VGG11 provided in https://github.com/huyvnphan/PyTorch_CIFAR10/tree/v3.0.1
- We use a progress bar from https://github.com/gipert/progressbar
