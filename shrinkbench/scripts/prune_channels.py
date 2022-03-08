import argparse
import json
import os
import itertools
import datetime
import numpy as np
import pathlib
from shrinkbench.experiment import StructuredPruningExperiment
from collections import OrderedDict

parser = argparse.ArgumentParser(description='Prune channels of a NN and finetune it')

parser.add_argument('--task', dest='task_id', type=int,
                    help='task id if running job array', default=1)
parser.add_argument('--ntasks', dest='ntasks', type=int,
                    help='number of tasks job array', default=1)
parser.add_argument('--job', dest='job_id', type=int,
                    help='job id if running job array', default=1)
parser.add_argument('-r', dest='nruns', type=int,
                    help='number of runs with different seed', default=1)
parser.add_argument('-s', '--strategies', dest='strategies', type=str,
                    help='Pruning strategies to run, separated by comma, or all to run all available strategies',
                    default='all')
parser.add_argument('-d', '--dataset', dest='dataset', type=str, help='Dataset to train on', default='MNIST')
parser.add_argument('-m', '--model', dest='model', type=str, help='What NN to use', default='MnistNet')
parser.add_argument('-t', '--structure', dest='structure', type=str, help='type of structure to prune '
                                                                          '(neurons or channels)', default='neuron')
parser.add_argument('-l', '--layers', dest='layers', type=str, help='names of layers to prune, separated by comma'
                                        ' or all to prune all layers corresponding to chosen structure', default='all')
parser.add_argument('-a', '--algo', dest='algo', type=str, help='What algorithm to use for SeqInChangeChannel '
                                                                'and InChangeChannel', default=None)
parser.add_argument('-o', '--optim', dest='optim', type=str, help='What optimization algo to use for fine-tuning',
                    default='Adam')
parser.add_argument('-b', '--nbatches', dest='nbatches', type=int, help='# of batches to use in pruning', default=1)
parser.add_argument('-rw', '--rwchange', dest='rwchange',  type=bool, help='use reweighted change if true',
                    default=None)
parser.add_argument('--layjob', dest='layjob_id', type=int, help='job id of one layer pruning job, if not provided use '
                                                                 'equal fractions for all layers', default=None)
parser.add_argument('-f', '--fractions', dest='fractions', nargs="+", type=float, help='List of fractions of prunable '
                                                    'channels to keep', default=[0.01, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0])
parser.add_argument('--select', dest='select', type=str, help='use min or sum in fraction selection',
                    default='min')
parser.add_argument('--patches', dest='patches', type=str, help='use all or disjoint or random patches in LayerInChangeChannel',
                    default='all')
parser.add_argument('--path', dest='path', type=str, help='path to save/read results', default='results')
parser.add_argument('--data_path', dest='data_path', type=str, help='path to dataset', default='../data')


if __name__ == '__main__':

    args = parser.parse_args()

    os.environ['DATAPATH'] = args.data_path
    os.environ['WEIGHTSPATH'] = '../pretrained/shrinkbench-models'
    os.environ['TORCH_HOME'] = '../pretrained/torchvision-models'

    fixed_params = {'dataset': args.dataset,
                    'model': args.model,
                    'structure': args.structure,
                    'prune_layers': args.layers.split(','),
                    'train_kwargs': {'epochs': 10, 'optim': args.optim, 'optim_kwargs': {'lr': 1e-3}} if
                                     args.dataset == 'MNIST' and args.model == 'LeNet' else {'epochs': 10 if
                                     args.dataset == 'MNIST' else 20, 'optim': args.optim, 'scheduler': 'MultiStepLR',
                                     'optim_kwargs': {'lr': 1e-3, 'weight_decay': 5e-4},
                                     'scheduler_kwargs': {'gamma': 0.1, 'milestones': [10, 15]}},
                    'dl_kwargs': {'cifar_shape': True} if args.model == 'vgg11_bn_small_mnist' else {},
                    'nbatches': args.nbatches,
                    'pretrained': True,
                    'debug': False}

    layer_strategies = ['LayerInChangeChannel',  'LayerRandomChannel', 'LayerWeightNormChannel', 'LayerActGradChannel']
    global_strategies = ['RandomChannel', 'WeightNormChannel', 'ActGradChannel']

    strategies = layer_strategies + global_strategies if args.strategies == "all" else args.strategies.split(',')

    hyperparam_options = OrderedDict({'seed': [42 + i for i in range(args.nruns)],
                                      'strategy': strategies,
                                      'reweight': [True, False]})

    prune_kwargs = {"rwchange": [True, False],
                    "asymmetric": [True, False],
                    # 0: no normalization, 1: normalize by # rows of activation matrix, 2: normalize by Squared
                    # Frobenius norm of input of corresponding layer
                    "norm": [0, 1, 2],
                    "out": [True, False],
                    "algo": ['greedy'] if args.algo is None else [args.algo], # 'sieve'
                    "select": [args.select],
                    "patches": [args.patches],
                    "onelayer_results_dir": [f'{args.path}/onelayer-pruning/CIFAR10-vgg11_bn_small-{args.layjob_id}' if args.model=="vgg11_bn_small_mnist" else
                                             f'{args.path}/onelayer-pruning/{args.dataset}-{args.model}-{args.layjob_id}'
                                             if args.layjob_id is not None else None]}

    def get_prune_kwargs_dict(var_keys, fixed_params={}):
        return [dict(zip(var_keys, params), **fixed_params) for params in itertools.product(*[prune_kwargs[key] for key in var_keys])]

    prune_kwargs_map = {'LayerInChangeChannel': get_prune_kwargs_dict(["algo", "asymmetric", "onelayer_results_dir", "select", "patches"],
                                                                      {"sequential": True, "rwchange": True, "norm": 2})
                                              + get_prune_kwargs_dict(["algo", "onelayer_results_dir", "select", "patches"],
                                                                    {"sequential": False, "rwchange": True, "norm": 2}),
                        'InChangeChannel': get_prune_kwargs_dict(["algo"], {"rwchange": True, "norm": 1}),
                        'WeightChangeChannel': get_prune_kwargs_dict(["out", "algo"], {"norm": 2}),
                        'LayerWeightChangeChannel': get_prune_kwargs_dict(["out", "algo", "onelayer_results_dir", "select"],
                                                                          {"sequential": False, "norm": 0}),
                        'RandomChannel': [{}],
                        'LayerRandomChannel': get_prune_kwargs_dict(["onelayer_results_dir", "select"]),
                        'WeightNormChannel': [{"norm": True, "ord": 1}],  # , {"norm": False}],
                        'LayerWeightNormChannel':  get_prune_kwargs_dict(["onelayer_results_dir", "select"],{"ord": 1}),
                        'ActGradChannel': [{"norm": True}], # , {"norm": False}]
                        'LayerActGradChannel':get_prune_kwargs_dict(["onelayer_results_dir", "select"])}

    # add fractions last to hyperparams dict to have jobs for all fractions of a particular setup consecutive
    fractions = args.fractions
    hyperparam_keys = list(hyperparam_options.keys()) + ["prune_kwargs", "fractions"]
    hyperparams_set = [OrderedDict(zip(hyperparam_keys, params + (prune_kwargs, fraction)))
                       for params in itertools.product(*hyperparam_options.values())
                       for prune_kwargs in prune_kwargs_map[params[1]]
                       for fraction in fractions]

    nparams_task = int(np.ceil(len(hyperparams_set) / args.ntasks))
    task_hyperparams_set = [hyperparams_set[args.task_id-1]] if len(hyperparams_set) == args.ntasks else \
        hyperparams_set[nparams_task*(args.task_id-1):nparams_task*args.task_id]

    fixed_params['rootdir'] = f'{args.path}/{args.dataset}-{args.model}-{args.job_id}'
    parent = pathlib.Path(fixed_params['rootdir'])
    parent.mkdir(parents=True, exist_ok=True)
    config_file = parent / 'config.json'
    if not config_file.is_file():
        config = {**fixed_params, **hyperparam_options, "prune_kwargs": prune_kwargs_map, "fractions": fractions}
        json.dump(config, open(config_file, 'w'), indent=4)

    for hyperparams in task_hyperparams_set:
        params = {**fixed_params, **hyperparams}
        print(f"Running structured pruning experiment with {params}")
        exp = StructuredPruningExperiment(**params)
        exp.run()

