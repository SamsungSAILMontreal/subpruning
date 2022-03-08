import os
import torch
from shrinkbench.experiment import PruningExperiment, StructuredPruningExperiment
from shrinkbench.plot import df_from_results, plot_df
import pandas as pd


if __name__ == '__main__':

    os.environ['DATAPATH'] = '../data' 
    os.environ['WEIGHTSPATH'] = '../pretrained/shrinkbench-models'
    os.environ['TORCH_HOME'] = '../pretrained/torchvision-models'

    print(torch.cuda.is_available())

    debug = True
    model = 'LeNet'
    dataset = 'MNIST'
    # model = 'vgg11_bn_small'
    # dataset = 'CIFAR10'
    structure = 'neuron'
    fractions = [0.25]
    strategies = [('LayerInChangeChannel',  {"rwchange": True, "sequential": True, "asymmetric": True, "norm": 2,
                                             "algo": "greedy", "patches": "all", "backward": False})]

    for strategy, prune_kwargs in strategies:
        for reweight in [True]:
                exp = StructuredPruningExperiment(dataset=dataset,
                                              model=model,
                                              strategy=strategy,
                                              fractions=fractions,
                                              reweight=reweight,
                                              bias=False,
                                              structure=structure,
                                              prune_layers=["all"],  # ['features.0', 'features.8', 'features.15', 'features.22'], ["all"]
                                              nbatches=4,
                                              prune_kwargs=prune_kwargs,
                                              train_kwargs={'epochs': 0, 'optim': 'Adam', 'optim_kwargs': {'lr': 1e-3}} if
                                              dataset == 'MNIST' else {'epochs': 0, 'optim': 'Adam', 'scheduler': 'MultiStepLR',
                                                            'optim_kwargs': {'lr': 1e-3, 'weight_decay': 5e-4},
                                                            'scheduler_kwargs': {'gamma': 0.1, 'milestones': [10, 15]}},
                                              dl_kwargs={'num_workers': 0},
                                              # rootdir=f'results/onelayer-pruning/{dataset}-{model}',
                                              # verif=True,
                                              debug=debug,
                                              pretrained=True,
                                              seed=42)
                exp.run()

    structured = True
    cf_key = 'fraction' if structured else 'compression'
    df = df_from_results('results' + ('/tmp' if debug else ''), structured=structured)
    df = df[(df['model'] == model) & (df['dataset'] == dataset)]

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)


