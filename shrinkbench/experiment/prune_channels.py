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
from PySSM import fix_seed_PySSM
import shutil
import os
import torch.nn.utils.prune as prune
from ..pruning.structured_utils import get_module


class StructuredPruningExperiment(TrainingExperiment):
    def __init__(self,
                 dataset,
                 model,
                 strategy,
                 fractions,  # fraction or list of fractions of prunable channels to keep
                 reweight=False,
                 bias=False,
                 structure=None,
                 prune_layers=list(),
                 nbatches=1,
                 prune_kwargs=dict(),  # include 'onelayer_results_dir' to select perlayer fractions accordingly,
                 seed=42,
                 path=None,
                 rootdir=None,
                 dl_kwargs=dict(),
                 train_kwargs=dict(),
                 verif=False,  # use verification set as validation set if True
                 debug=False,
                 pretrained=True,
                 pruned_path=None,  # if not None, load pruned model from pruned_path instead of pruning
                 finetune=False,  # allow fine tuning even if fraction = 1, and use all training set (empty verif set)
                 limited_data=False, # None,  # only use nbatches of data for both pruning and finetuning if True, if None finetune with both limited and full data
                 resume=None,
                 resume_optim=False,
                 save_freq=10):

        self.fix_seed(seed)
        fix_seed_PySSM(seed)

        if limited_data:
            dl_kwargs['nbatches'] = nbatches
        super().__init__(dataset, model, seed, path, dl_kwargs, train_kwargs, debug, pretrained, finetune, resume,
                         resume_optim, save_freq)
        size, size_nz = model_size(self.model)
        self.size_nz_orig = size_nz
        self.to_device()
        x, y = next(iter(self.val_dl))
        x, y = x.to(self.device), y.to(self.device)
        ops, ops_nz = flops(self.model, x)
        self.ops_nz_orig = ops_nz
        print("compression ratio before pruning = ", size / size_nz)
        print("speedup before pruning = ", ops / ops_nz)

        if np.isscalar(fractions):
            fractions = [fractions]
        self.add_params(strategy=strategy, fractions=fractions, reweight=reweight, bias=bias, structure=structure,
                        prune_layers=prune_layers, nbatches=nbatches, prune_kwargs=prune_kwargs, verif=verif,
                        finetune=finetune, pruned_path=pruned_path)

        self.pruned_path = pruned_path
        self.build_pruning(strategy, fractions, reweight, bias, structure, prune_layers, nbatches, **prune_kwargs)
        self.verif = verif
        self.path = path
        if rootdir is not None:
            self.rootdir = rootdir
        self.save_freq = save_freq

    def build_pruning(self, strategy, fractions, reweight, bias, structure, prune_layers, nbatches, **prune_kwargs):
        if self.pruned_path is None:
            constructor = getattr(strategies, strategy)
            for i, (x, y) in zip(range(nbatches), self.train_dl):  # self.prune_dl
                if i == 0:
                    xs, ys = ([x], [y])
                else:
                    xs = xs + [x]
                    ys = ys + [y]

            if 'onelayer_results_dir' in prune_kwargs:
                if prune_kwargs['onelayer_results_dir'] is not None:
                    df = df_from_results(prune_kwargs['onelayer_results_dir'], structured=True)
                    strategy_name = strategy + ''.join(sorted([param_label(k, v) if k not in ['sequential', 'asymmetric', 'epsilon']
                                                               else '' for k, v in prune_kwargs.items()]))
                    onelayer_results_df = df[(df['strategy'] == strategy_name) & (df['reweight'] == reweight)] # (df['structure'] == structure)
                else:
                    onelayer_results_df = None
                del prune_kwargs['onelayer_results_dir']
                prune_kwargs['onelayer_results_df'] = onelayer_results_df

            if 'full_data' in prune_kwargs:
                if prune_kwargs['full_data']:
                    prune_kwargs['train_dl'] = self.train_dl

            printc(f"Pruning model with {strategy}", color='GREEN')
            since = time.perf_counter()
            self.pruning = constructor(self.model, xs, ys, fractions=fractions, reweight=reweight, bias=bias,
                                       structure=structure, prune_layers=prune_layers, **prune_kwargs)
            self.pruning_time = (time.perf_counter() - since)/len(fractions)  # assuming all fractions required similar time..

    def run(self):
        for fraction in self.params['fractions']:
            child = copy.deepcopy(self) if len(self.params['fractions']) > 1 else self
            del child.params['fractions']
            child.add_params(fraction=fraction)
            if self.pruned_path is None:
                since = time.perf_counter()
                child.pruning.apply(fraction)
                child.pruning_time += time.perf_counter() - since
                printc(f"Masked model with fraction={fraction} of prunable channels kept", color='GREEN')
            else:
                model_state = torch.load(f'{self.pruned_path}/checkpoints/checkpoint--1.pt', map_location=self.device)['model_state_dict']
                # generate pruning parametrization
                for key in model_state.keys():
                    if key.endswith("_orig"):
                        mod, pname = key[:-5].rsplit('.', 1)
                        prune.identity(get_module(child.model, mod), pname)
                child.model.load_state_dict(model_state)
                printc(f"Loaded masked model with fraction={fraction} of prunable channels kept", color='GREEN')
            child.freeze()
            printc(f"Running {repr(child)}", color='YELLOW')
            child.to_device()
            child.build_logging(child.train_metrics, child.path)

            if self.pruned_path is None:
                child.save_metrics()
            else:
                assert os.path.isfile(f'{self.pruned_path}/metrics.json'), "missing metrics"
                shutil.copy(f'{self.pruned_path}/metrics.json', f'{self.path}/metrics.json')
                # log validation loss and accuracy before finetuning in logs file
                since = time.perf_counter()
                _ = self.run_epoch(False, opt=False, epoch=-1, verif=self.verif)
                self.log(timestamp=time.perf_counter() - since)
                self.log_epoch(-1)
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
        metrics['flops_nz_orig'] = self.ops_nz_orig
        metrics['flops_nz'] = ops_nz
        metrics['theoretical_speedup'] = self.ops_nz_orig / ops_nz

        # Training and validation loss and accuracy
        since = time.perf_counter()  # in sec
        for train in [False]:
            prefix = 'train' if train else 'val'
            loss, acc1, acc5 = self.run_epoch(train, opt=False, epoch=-1, verif=self.verif)

            metrics[f'{prefix}_loss'] = loss
            metrics[f'{prefix}_acc1'] = acc1
            metrics[f'{prefix}_acc5'] = acc5
        self.log(timestamp=time.perf_counter() - since)
        # checkpoint pruned model before fine-tuning
        self.checkpoint(-1)
        self.log_epoch(-1)

        return metrics

