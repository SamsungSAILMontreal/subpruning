import os
import time
import torch
import cProfile
import torchvision
from shrinkbench.experiment import PruningExperiment, StructuredPruningExperiment
from shrinkbench.plot import df_from_results, plot_df
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PySSM import Greedyint, SieveStreamingint, SieveStreamingPPint, ThreeSievesint, Salsaint
from shrinkbench.strategies.greedy_algos import greedy_algo
from shrinkbench.metrics import model_size

if __name__ == '__main__':

    os.environ['DATAPATH'] = '../data'
    os.environ['WEIGHTSPATH'] = '../pretrained/shrinkbench-models'
    os.environ['TORCH_HOME'] = '../pretrained/torchvision-models'

    print(torch.cuda.is_available())

    debug = True
    model = 'LeNet'
    dataset = 'MNIST'
#    model = 'resnet56'  # 'resnet56', 'vgg11_bn_small'
#    dataset = 'CIFAR10'
    structure = None
    fractions = [0.1]
    nbatches = 4
    strategies = [('RandomChannel', {})]
    

    for strategy, prune_kwargs in strategies:
        for reweight in [False]:
                #pr = cProfile.Profile()
                #pr.enable()
                exp = StructuredPruningExperiment(dataset=dataset,
                                              model=model,
                                              strategy=strategy,
                                              fractions=fractions,
                                              reweight=reweight,
                                              bias=False,
                                              structure=structure,
                                              prune_layers=['all'],
                                              nbatches=nbatches,
                                              prune_kwargs=prune_kwargs,
                                              pruned_path=None,
                                              limited_data=True,
                                              train_kwargs={'epochs': 2, 'optim': 'Adam', 'optim_kwargs': {'lr': 1e-3}} if
                                              dataset == 'MNIST' else {'epochs': 2, 'optim': 'Adam', 'scheduler': 'MultiStepLR',
                                                            'optim_kwargs': {'lr': 1e-3, 'weight_decay': 5e-4},
                                                            'scheduler_kwargs': {'gamma': 0.1, 'milestones': [10, 15]}},
                                              dl_kwargs={'num_workers': 0},
                                              # rootdir=f'results/onelayer-pruning/{dataset}-{model}',
                                              verif=True,
                                              debug=debug,
                                              pretrained=True,
                                              seed=42)
                exp.run()
                #pr.disable()

    structured = True
    cf_key = 'fraction' if structured else 'compression'
    df = df_from_results('results' + ('/tmp' if debug else ''), structured=structured, icml=True)
    df = df[(df['model'] == model) & (df['dataset'] == dataset)]

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)

    print("done")

   
