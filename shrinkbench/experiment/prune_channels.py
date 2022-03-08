import json
import copy

from .train import TrainingExperiment

from .. import strategies
from ..metrics import model_size, flops
from ..util import printc
import numpy as np
import torch
import time
from shrinkbench.plot import df_from_results, param_label


class StructuredPruningExperiment(TrainingExperiment):
    def __init__(self,
                 dataset,
                 model,
                 strategy,
                 fractions,  # fraction or list of fractions of prunable channels to keep
                 reweight=False,
                 bias=False,
                 structure='neuron',
                 prune_layers=list(),
                 nbatches=1,
                 prune_kwargs=dict(),  # include 'onelayer_results_dir' to select perlayer fractions accordingly
                 seed=42,
                 path=None,
                 rootdir=None,
                 dl_kwargs=dict(),
                 train_kwargs=dict(),
                 verif=False,  # use verification set as validation set if True
                 debug=False,
                 pretrained=True,
                 finetune=False,  # allow fine tuning even if fraction = 1, and use all training set (empty verif set)
                 resume=None,
                 resume_optim=False,
                 save_freq=10):

        super().__init__(dataset, model, seed, path, dl_kwargs, train_kwargs, debug, pretrained, finetune, resume,
                         resume_optim, save_freq)
        size, size_nz = model_size(self.model)
        self.size_nz_orig = size_nz
        print("compression before pruning = ", size / size_nz)

        if np.isscalar(fractions):
            fractions = [fractions]
        self.add_params(strategy=strategy, fractions=fractions, reweight=reweight, bias=bias, structure=structure,
                        prune_layers=prune_layers, nbatches=nbatches, prune_kwargs=prune_kwargs, verif=verif, finetune=finetune)

        self.fix_seed(self.seed)
        self.build_pruning(strategy, fractions, reweight, bias, structure, prune_layers, nbatches, **prune_kwargs)
        self.verif = verif
        self.path = path
        if rootdir is not None:
            self.rootdir = rootdir
        self.save_freq = save_freq

    def build_pruning(self, strategy, fractions, reweight, bias, structure, prune_layers, nbatches, **prune_kwargs):
        constructor = getattr(strategies, strategy)
        # concatenate nbatches together
        for i, (x, y) in zip(range(nbatches), self.train_dl):
            if i == 0:
                xs, ys = x, y
            else:
                xs = torch.cat((xs, x), axis=0)
                ys = torch.cat((ys, y), axis=0)

        if 'onelayer_results_dir' in prune_kwargs:
            if prune_kwargs['onelayer_results_dir'] is not None:
                df = df_from_results(prune_kwargs['onelayer_results_dir'], structured=True)
                strategy_name = strategy + ''.join(sorted([param_label(k, v) if k not in ['sequential', 'asymmetric']
                                                           else '' for k, v in prune_kwargs.items()]))
                onelayer_results_df = df[(df['strategy'] == strategy_name) & (df['reweight'] == reweight) &
                                         (df['structure'] == structure)]
            else:
                onelayer_results_df = None
            del prune_kwargs['onelayer_results_dir']
            prune_kwargs['onelayer_results_df'] = onelayer_results_df

        printc(f"Pruning model with {strategy}", color='GREEN')
        since = time.perf_counter()
        self.pruning = constructor(self.model, xs, ys, fractions=fractions, reweight=reweight, bias=bias,
                                   structure=structure, prune_layers=prune_layers, **prune_kwargs)
        self.pruning_time = (time.perf_counter() - since)/len(fractions)  # assuming all fractions required similar time..

    def run(self):
        for fraction in self.pruning.fractions:
            child = copy.deepcopy(self) if len(self.pruning.fractions) > 1 else self
            del child.params['fractions']
            child.add_params(fraction=fraction)
            since = time.perf_counter()
            child.pruning.apply(fraction)
            child.pruning_time += time.perf_counter() - since
            printc(f"Masked model with fraction={fraction} of prunable channels kept", color='GREEN')

            child.freeze()
            printc(f"Running {repr(child)}", color='YELLOW')
            child.to_device()
            child.build_logging(child.train_metrics, child.path)

            child.save_metrics()
            if fraction < 1 or self.params['finetune']:
                child.run_epochs()

    def save_metrics(self):
        self.metrics = self.pruning_metrics()
        with open(self.path / 'metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=4)
        printc(json.dumps(self.metrics, indent=4), color='GRASS')
        summary = self.pruning.summary(self.params['fraction'])
        summary_path = self.path / 'masks_summary.csv'
        summary.to_csv(summary_path)
        print(summary)

    def pruning_metrics(self):

        metrics = {}
        # Time
        metrics['pruning_time'] = self.pruning_time

        # Model Size
        size, size_nz = model_size(self.model)
        metrics['size'] = size
        metrics['size_nz_orig'] = self.size_nz_orig
        metrics['size_nz'] = size_nz
        metrics['compression_ratio'] = self.size_nz_orig / size_nz

        x, y = next(iter(self.val_dl))
        x, y = x.to(self.device), y.to(self.device)

        # FLOPS
        ops, ops_nz = flops(self.model, x)
        metrics['flops'] = ops
        metrics['flops_nz'] = ops_nz
        metrics['theoretical_speedup'] = ops / ops_nz

        # Training and validation loss and accuracy
        since = time.perf_counter()  # in sec
        for train in [False]:
            prefix = 'train' if train else 'val'
            loss, acc1, acc5 = self.run_epoch(train, opt=False, epoch=-1, verif=self.verif)

            metrics[f'{prefix}_loss'] = loss
            metrics[f'{prefix}_acc1'] = acc1
            metrics[f'{prefix}_acc5'] = acc5
        self.log(timestamp=time.perf_counter() - since)
        self.log_epoch(-1)

        return metrics

