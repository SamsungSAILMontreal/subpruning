import argparse
import json
import os
import itertools
import numpy as np
import pathlib
from shrinkbench.experiment import StructuredPruningExperiment, TrainingExperiment
from shrinkbench.pruning import prunable_modules
from collections import OrderedDict
import subprocess

parser = argparse.ArgumentParser(description='Prune channels in every prunable layer of a NN')

parser.add_argument('--task', dest='task_id', type=int,
                    help='task id if running job array', default=1)
parser.add_argument('--ntasks', dest='ntasks', type=int,
                    help='number of tasks job array', default=1)
parser.add_argument('--job', dest='job_id', type=int,
                    help='job id if running job array', default=1)
parser.add_argument('-r', dest='nruns', type=int,
                    help='number of runs with different seed', default=1)
parser.add_argument('-d', '--dataset', dest='dataset', type=str, help='Dataset to train on', default='MNIST')
parser.add_argument('-m', '--model', dest='model', type=str, help='What NN to use', default='MnistNet')
parser.add_argument('-t', '--structure', dest='structure', type=str, help='type of structure to prune '
                                                                          '(neurons or channels)', default=None)
parser.add_argument('-s', '--strategies', dest='strategies', type=str,
                    help='Pruning strategies to run, separated by comma, or all to run all available layerwise '
                         'strategies', default='all')
parser.add_argument('-l', '--layers', dest='layers', type=str, help='names of layers to prune, separated by comma'
                                        ' or all to prune all layers corresponding to chosen structure', default='all')
parser.add_argument('-a', '--algo', dest='algo', type=str, help='What algorithm to use for SeqInChangeChannel '
                                                                'and InChangeChannel', default=None)
parser.add_argument('-o', '--optim', dest='optim', type=str, help='What optimization algo to use for fine-tuning',
                    default='Adam')
parser.add_argument('-b', '--nbatches', dest='nbatches', type=int, help='# of batches to use in pruning', default=1)
parser.add_argument('-f', '--fractions', dest='fractions', nargs="+", type=float, help='List of fractions of prunable '
                                       'channels to keep', default=[0.01, 0.05, 0.075] + [i*0.05 for i in range(2, 21)])
parser.add_argument('-c', '--compressions', dest='compressions', nargs="+", type=float, help='List of compressions of prunable '
                                       'channels to keep', default=[1/0.01, 1/0.05, 1/0.075] + [1/(i*0.05) for i in range(2, 21)])
parser.add_argument('--reweight', dest='reweight',  choices=('True', 'False'), default=None,
                    help='apply reweighting procedure after pruning')
parser.add_argument('-fw', '--fw', dest='fw',  choices=('True', 'False'), default='False',
                    help='use Frank-Wolfe style greedy in GreedyFSChannel')
parser.add_argument('-fd', '--full_data', dest='full_data',  choices=('True', 'False'), default=None,
                    help='use full training data for pruning in GreedyFSChannel')
parser.add_argument('--patches', dest='patches', type=str, help='use all or disjoint or random patches in LayerInChangeChannel',
                    default='all')
parser.add_argument('--path', dest='path', type=str, help='path to save/read results', default='results')
parser.add_argument('--data_path', dest='data_path', type=str, help='path to dataset', default='../data')


if __name__ == '__main__':

    args = parser.parse_args()

    os.environ['DATAPATH'] = args.data_path
    os.environ['WEIGHTSPATH'] = '../pretrained/shrinkbench-models'
    os.environ['TORCH_HOME'] = '../pretrained/torchvision-models'

    strategies = ['LayerInChangeChannel', 'LayerRandomChannel', 'LayerWeightNormChannel', 'LayerActGradChannel',
                  'LayerGreedyFSChannel'] if args.strategies == "all" else args.strategies.split(',')

    gitcommit_str = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    fixed_params = {'dataset': args.dataset,
                    'model': args.model,
                    'structure': args.structure,
                    'train_kwargs': {'epochs': 0},
                    'dl_kwargs': {'cifar_shape': True} if args.model == 'vgg11_bn_small_mnist' else {},
                    'nbatches': args.nbatches,
                    'fullnet_comp': False,
                    'verif': True,
                    'pretrained': True,
                    'debug': False,
                    'rootdir': f'{args.path}/onelayer-pruning/{args.dataset}-{args.model}-{gitcommit_str}-{args.job_id}'}

    exp = TrainingExperiment(args.dataset, args.model)
    _, channel_prunable, _, _ = prunable_modules(exp.model, args.structure, 'all')

    hyperparam_options = OrderedDict({'seed': [42 + i for i in range(args.nruns)],
                                      'strategy': strategies,
                                      'reweight': [True, False] if args.reweight is None else [args.reweight == "True"],
                                      'prune_layers': [[layer] for layer in (channel_prunable if args.layers == "all"
                                                                             else args.layers.split(','))]})

    prune_kwargs = {"rwchange": [True, False],
                    # 0: no normalization, 1: normalize by # rows of activation matrix, 2: normalize by Squared
                    # Frobenius norm of input of corresponding layer
                    "norm": [0, 1, 2],
                    "out": [True, False],
                    "scale_masks": [True],  # False
                    "fw": [args.fw == "True"],
                    "patches": [args.patches],
                    "epsilon": [0.1, 0.01, 0.001],  # [0, 0.1, 0.01, 0.001]
                    "full_data": [True, False] if args.full_data is None else [args.full_data == "True"],
                    "algo": ['greedy'] if args.algo is None else [args.algo]} #'sieve'

    def get_prune_kwargs_dict(var_keys, fixed_params={}):
        return [dict(zip(var_keys, params), **fixed_params) for params in
                itertools.product(*[prune_kwargs[key] for key in var_keys])]

    prune_kwargs_map = {'LayerInChangeChannel': get_prune_kwargs_dict(["algo", "patches"],
                                                {"sequential": False, "rwchange": True, "norm": 2}),
                        'LayerWeightChangeChannel': get_prune_kwargs_dict(["out", "algo"], {"sequential": False,
                                                                                            "norm": 0}),
                        'LayerRandomChannel': [{}],
                        'LayerWeightNormChannel': [{"ord": 1}],
                        'LayerActGradChannel': [{}],
                        'LayerGreedyFSChannel': get_prune_kwargs_dict(["scale_masks", "full_data", "fw"])
                        }

    # add compressions last to hyperparams dict to have jobs for all compressions of a particular setup consecutive
    compressions = args.compressions
    hyperparam_keys = list(hyperparam_options.keys()) + ["prune_kwargs", "compressions"]
    hyperparams_set = [OrderedDict(zip(hyperparam_keys, params + (prune_kwargs, compression)))
                       for params in itertools.product(*hyperparam_options.values())
                       for prune_kwargs in prune_kwargs_map[params[1]]
                       for compression in compressions]
    nparams_task = int(np.ceil(len(hyperparams_set) / args.ntasks))
    task_hyperparams_set = [hyperparams_set[args.task_id - 1]] if len(hyperparams_set) == args.ntasks else \
        hyperparams_set[nparams_task * (args.task_id - 1):nparams_task * args.task_id]
    parent = pathlib.Path(fixed_params['rootdir'])
    parent.mkdir(parents=True, exist_ok=True)
    config_file = parent / 'config.json'
    if not config_file.is_file():
        config = {**fixed_params, **hyperparam_options, "prune_kwargs": prune_kwargs_map, "compressions": compressions}
        json.dump(config, open(config_file, 'w'), indent=4)

    for hyperparams in task_hyperparams_set:
        params = {**fixed_params, **hyperparams}
        print(f"Running structured pruning experiment with {params}")
        exp = StructuredPruningExperiment(**params)
        exp.run()
