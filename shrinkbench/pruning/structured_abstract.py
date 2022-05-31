from abc import abstractmethod
from .utils import get_params
from .structured_utils import *
from .mixin import ActivationMixin
from collections import OrderedDict, defaultdict
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import itertools
import time


class StructuredPruning(ActivationMixin):
    # Base class for structured pruning
    # if structure = neuron prunes output neurons of linear layers followed by another linear layer
    # if structure = channel prunes output channels of conv layers followed by another conv layer
    # if structure = None prunes both linear and conv layers
    # we use channel to refer to both neurons and conv channels
    # Computes pruned channels corresponding to all fractions in initialization by default
    # calling apply(fraction) constructs masks and applies pruning for corresponding fraction
    # if reweight=True, we update next layer weights with weights that minimize change of input to next layer
    # if bias=True, we treat the bias as another channel (with no input weights) that can be pruned

    def __init__(self, model, inputs=None, outputs=None, fractions=1, reweight=False, bias=False, structure='neuron',
                 prune_layers=None, **pruning_params):
        inputs = torch.cat(inputs, axis=0)
        outputs = torch.cat(outputs, axis=0)
        self.device = torch.device('cpu')  # 'cuda' if torch.cuda.is_available() else 'cpu'
        super().__init__(model, inputs, outputs, fractions=fractions, reweight=reweight, bias=bias, structure=structure,
                         prune_layers=prune_layers, **pruning_params)
        self.prunable, self.channel_prunable, self.next_module_map, self.bn_map = prunable_modules(self.model, structure,
                                                                                                   prune_layers)
        # if isinstance(self, LayerStructuredPruning):
        #     if len(self.channel_prunable) > 1 and self.onelayer_results_df is not None:
        #         t = time.perf_counter()
        #         self.perlayer_fractions = self.select_perlayer_fractions()
        #         print("perlayer fractions selection time:", time.perf_counter() - t, "secs.")
        #     else:
        #         self.perlayer_fractions = {fraction: {name: fraction for name in self.channel_prunable} for fraction
        #                                    in self.fractions}

        self.preprocessing()
        if fractions == [1]:  # prune nothing
            self.pruned_channels = {1: {name: [] for name in self.channel_prunable}}
            self.pruned_bias = {1: {name: False for name in self.channel_prunable}}
            self.new_params = {1: {name: None for name in self.channel_prunable}}
        else:
            self.pruned_channels, self.pruned_bias, self.new_params = self.model_pruned_channels()
            if self.pruned_bias is None:
                self.pruned_bias = {fraction: {name: False for name in self.channel_prunable} for fraction in fractions}

    def preprocessing(self):
        # do any preprocessing steps needed before calling model_pruned_channels()
        # skipped if fractions==[1]
        pass

    def can_prune(self, module_name):
        # true for any module where output channels can be pruned
        return module_name in self.channel_prunable

    def apply(self, fraction):
        assert fraction in self.fractions, "fraction should be one of the input fractions"
        if fraction != 1:
            self.reweight_params(fraction)
        masks = self.model_masks(fraction)

    def get_module(self, module_name, next=False):
        if next:
            name = self.next_module_map[module_name]
        else:
            name = module_name
        return get_module(self.model, name)

    def module_params(self, module_name, next=False):
        return get_params(self.get_module(module_name, next=next))

    def params(self, only_prunable=True, next=False):
        if only_prunable:
            return {name: self.module_params(name, next) for name in self.channel_prunable}
        else:
            return {name: get_params(module) for name, module in self.model.named_modules()}

    def module_n_channels(self, module_name, bias=False):
        params = self.module_params(module_name)
        return params['weight'].shape[0] + (1 if bias and "bias" in params else 0)

    def n_channels(self, only_prunable=True, bias=False):
        if only_prunable:
            return sum([self.module_n_channels(name, bias) for name in self.channel_prunable])
        else:
            return sum([self.module_n_channels(name, bias) for name, _ in self.model.named_modules()])

    def channels_ind(self):
        return {name: set(range(self.module_n_channels(name, bias=self.bias))) for name in self.channel_prunable}

    @abstractmethod
    def model_pruned_channels(self):
        # returns a dictionary of pruned channels for each fraction, module_name
        # {fraction: {module_name: list of pruned channels in this module}}
        # and optionally a dictionary listing if bias was pruned or not for each fraction, otherwise return none
        # {fraction: True if bias is pruned, False otherwise}
        # and optionally a dictionary of new params values, otherwise return none
        # {fraction: {module_name: {param_name: new param of next module (numpy.ndarray)}}}
        pass

    def layer_masks(self, module_name, fraction, masks=None, scale=1, perm=False):
        if masks is None:
            masks = defaultdict(OrderedDict)

        module = self.get_module(module_name)
        pruned_channels = self.pruned_channels[fraction][module_name]

        bn_name = self.bn_map[module_name]
        for mod in [module] + ([self.get_module(bn_name)] if bn_name is not None else []):
            for name in get_params(mod).keys():  # ['weight', 'bias']:
                indices_structured(mod, name, 0, pruned_channels)
                if perm:  # Multiply params by (non-scaled) masks so metrics can count nonzeros
                    getattr(mod, name + '_orig').data.mul_(getattr(mod, name + '_mask') != 0)
                masks[mod][name] = getattr(mod, name + '_mask')

        next_module = self.get_module(module_name, next=True)
        # scale only next layer weights
        indices_structured(next_module, 'weight', 1, pruned_channels, scale)
        if perm:  # Multiply params by (non-scaled) masks so metrics can count nonzeros
            next_module.weight_orig.data.mul_(next_module.weight_mask != 0)
        masks[next_module]['weight'] = getattr(next_module, 'weight_mask')
        if self.pruned_bias[fraction][module_name] and next_module.bias is not None:
            indices_structured(next_module, 'bias', 0, list(range(0, next_module.bias.shape[0])))
            if perm:
                next_module.bias_orig.data.mul_(next_module.bias_mask != 0)
            masks[next_module]['bias'] = getattr(next_module, 'bias_mask')
        return masks

    def undo_layer_masks(self, module_name, weight_mask):
        undo_pruning(self.get_module(module_name), weight_mask)
        bn_name = self.bn_map[module_name]
        if bn_name is not None:
            undo_pruning(self.get_module(bn_name))
        undo_pruning(self.get_module(module_name, next=True))

    def model_masks(self, fraction):
        masks = defaultdict(OrderedDict)
        for module_name in self.channel_prunable:
            self.layer_masks(module_name, fraction, masks, scale=self.scale if (hasattr(self, 'scale') and not self.reweight) else 1, perm=True)
        return masks

    def reweight_layer_params(self, module_name, fraction):
        next_module = self.get_module(module_name, next=True)
        new_params = self.new_params[fraction][module_name]
        if new_params == {} and self.reweight:
            # update next layer weights with weights that minimize change of its input resulting from pruning output
            # channels of current layer
            params = get_params(next_module)
            acts = self.module_activations(next_module, only_input=True, update=True)
            if isinstance(next_module, nn.Conv2d):
                acts = torch.nn.functional.unfold(torch.from_numpy(acts), next_module.kernel_size, next_module.dilation,
                                                  next_module.padding, next_module.stride).transpose(1, 2).numpy()
                acts = acts.reshape(-1, acts.shape[-1])

            kept_channels = list(set(range(self.module_n_channels(module_name))) -
                                set(self.pruned_channels[fraction][module_name]))

            # if module is linear, kernel_size = 1, if it's conv followed by conv, kernel_size = H x W
            kernel_size = next_module.kernel_size if isinstance(next_module, nn.Conv2d) else \
                int(acts.shape[-1]/self.module_n_channels(module_name))
            map = lambda channels: map_chton(channels, kernel_size)
            neg_input_change = NegInputChange(acts, params, rwchange=True, bias=self.bias, map=map)
            new_params = neg_input_change.get_new_params(kept_channels, cur=False)
            self.new_params[fraction][module_name] = new_params

        for pname, new_param in new_params.items():
            with torch.no_grad():  # applied before pruning, so we're modifying weight/bias not weight_orig/bias_orig
                getattr(next_module, pname).data = torch.from_numpy(new_param).to(self.device)

    def reweight_params(self, fraction):
        if self.new_params is None:
            self.new_params = {fraction: {name: {} for name in self.channel_prunable} for fraction in self.fractions}
        for name in self.channel_prunable:
            self.reweight_layer_params(name, fraction)

    def summary(self, fraction):
        total_n_channels_kept = 0
        rows = []
        for name, module in self.model.named_modules():
            for pname, param in get_params(module).items():
                if hasattr(module, pname+'_mask'):  # isinstance(module, MaskedModule):
                    compression = 1/(getattr(module, pname+'_mask').detach().cpu().numpy() != 0).mean()
                else:
                    compression = 1

                shape = param.shape
                if name in self.channel_prunable:
                    if pname == 'weight':
                        pruned_channels = self.pruned_channels[fraction][name]
                        pruned_fraction = len(pruned_channels)/shape[0]

                        n_channels = self.module_n_channels(name, bias=self.bias)
                        if isinstance(self, LayerStructuredPruning):
                            k = int(self.perlayer_fractions[fraction][name] * n_channels)
                            assert n_channels - len(pruned_channels) - self.pruned_bias[fraction][name] == k, \
                                f"# of kept channels in {name} should be {k}"
                        total_n_channels_kept += n_channels - len(pruned_channels) - self.pruned_bias[fraction][name]
                    else:
                        pruned_channels = [0] if self.pruned_bias[fraction][name] else []
                        pruned_fraction = len(pruned_channels)

                else:
                    pruned_channels = []
                    pruned_fraction = 0

                rows.append([name, pname, compression, np.prod(shape), shape, self.can_prune(name), pruned_fraction, pruned_channels])

        if not isinstance(self, LayerStructuredPruning):
            k = int(fraction * self.n_channels(bias=self.bias))
            assert total_n_channels_kept == k, f"# of total kept channels should be {k}, {total_n_channels_kept} " \
                f"channels were kept"

        columns = ['module', 'param', 'comp', 'size', 'shape', 'channel prunable', 'pruned_fraction', 'pruned_channels']
        return pd.DataFrame(rows, columns=columns)


class LayerStructuredPruning(StructuredPruning):
    # prunes output channels of each prunable layer independently if sequential=False or sequentially starting from
    # first layer to last one. Uses perlayer fraction selection method from Kuzmin, Andrey, et al. if
    # onelayer_results_df is not none, otherwise uses equal fraction for all layers.

    def __init__(self, model, inputs=None, outputs=None, fractions=1, reweight=False, bias=False, structure='neuron',
                 prune_layers=None, sequential=False, onelayer_results_df=None, select='min', **pruning_params):

        self.onelayer_results_df, self.select = onelayer_results_df, select
        self.perlayer_fractions = {}
        super().__init__(model, inputs=inputs, outputs=outputs, fractions=fractions, reweight=reweight, bias=bias,
                         structure=structure, prune_layers=prune_layers, sequential=sequential, **pruning_params)

    def preprocessing(self):
        if self.fractions != [1] and len(self.channel_prunable) > 1 and self.onelayer_results_df is not None:
            t = time.perf_counter()
            self.perlayer_fractions = self.select_perlayer_fractions()
            print("perlayer fractions selection time:", time.perf_counter() - t, "secs.")
        else:
            self.perlayer_fractions = {fraction: {name: fraction for name in self.channel_prunable} for fraction
                                       in self.fractions}

    # TODO: save selected perlayer budgets instead of fractions, since all methods use that
    # TODO: add option to prune each layer until reaching an error threshold
    def select_perlayer_fractions(self, bs=True, eps=1e-5):
        # implements equal accuracy loss method proposed in Kuzmin, Andrey, et al. "Taxonomy and evaluation of
        # structured compression of convolutional neural networks." arXiv preprint arXiv:1912.09802 (2019).
        # but here we select n_channels instead of ranks

        self.onelayer_results_df.loc[:, 'fraction'] = self.onelayer_results_df.fraction.round(3)  # to avoid numerical issues
        fraction_choices = np.sort(self.onelayer_results_df['fraction'].unique())
        perlayer_acc = defaultdict(np.array) if bs else defaultdict(OrderedDict)
        perlayer_nchannels = [self.module_n_channels(name, self.bias) for name in self.channel_prunable]
        nchannels = sum(perlayer_nchannels)
        selected_perlayer_fractions = defaultdict(OrderedDict)

        # get acc for all fraction choices
        for name in self.channel_prunable:
            if bs:
                perlayer_acc[name] = np.zeros(len(fraction_choices))
            for idx, fraction in enumerate(fraction_choices):
                if bs:
                    perlayer_acc[name][idx] = self.onelayer_results_df[(self.onelayer_results_df['prune_layers'] == name)
                                             & (self.onelayer_results_df['fraction'] == fraction)]['pre_acc1'].values[0]
                else:
                    perlayer_acc[name][fraction] = self.onelayer_results_df[(self.onelayer_results_df['prune_layers'] == name)
                                                  & (self.onelayer_results_df['fraction'] == fraction)]['pre_acc1'].values[0]

        L = len(self.channel_prunable)
        if bs:
            # binary search
            assert self.select == 'min', "select should be min if using binary search"
            assert 1 in fraction_choices, "1 should be among fraction choices"  # because we use it for acc_max
            # acc at fraction 1 is not exactly the same due to numerical errors, we take min to guarantee acc_1 is satisfied by all layers
            acc_1 = self.onelayer_results_df[self.onelayer_results_df['fraction']==1]['pre_acc1'].min()
            # enforce monotonicity
            for name in self.channel_prunable:
                perlayer_acc[name] = monotone_envelope(fraction_choices, perlayer_acc[name])

            for fraction in self.fractions:
                nchannels_tokeep = int(fraction * nchannels)
                min_budget = int(nchannels_tokeep >= L)
                min_fraction = list(min_budget / np.array(perlayer_nchannels))
                # find perlayer fractions that lead to largest acc and satisfy nchannels_tokeep
                acc_min, acc_max = 0, acc_1
                while acc_max - acc_min > eps:
                    acc = (acc_min + acc_max)/2
                    # find smallest fraction satisfying this acc for each layer, acc is at least satisfied by fraction=1
                    perlayer_fractions = []
                    for i, name in enumerate(self.channel_prunable):
                        idx = max(0, min(np.searchsorted(perlayer_acc[name], acc, side='left'), len(fraction_choices)-1)) # make sure we don't go out of bounds
                        perlayer_fractions.append(max(fraction_choices[idx], min_fraction[i]))
                    nchannels_kept = sum(np.array(np.multiply(perlayer_fractions, perlayer_nchannels), int))
                    # check if this combination of per layer fractions satisfy this overall fraction of channels to keep
                    if nchannels_kept <= nchannels_tokeep:
                        acc_min = acc
                    else:
                        acc_max = acc

                # check if budget is satisfied at convergence, if not rerun with acc=acc_min
                if nchannels_kept > nchannels_tokeep:
                    perlayer_fractions = []
                    for i, name in enumerate(self.channel_prunable):
                        idx = max(0, min(np.searchsorted(perlayer_acc[name], acc_min, side='left'), len(fraction_choices)-1)) # make sure we don't go out of bounds
                        perlayer_fractions.append(max(fraction_choices[idx], min_fraction[i]))

                # use remaining quota if any by filling up layers gradually from one with largest fraction to smallest
                perlayer_fractions = np.array(perlayer_fractions)
                for idx in np.argsort(-perlayer_fractions):
                    nchannels_kept = sum(np.array(np.multiply(perlayer_fractions, perlayer_nchannels), int))
                    if nchannels_kept < nchannels_tokeep:
                        
                        perlayer_fractions[idx] = np.minimum((int(perlayer_fractions[idx]*perlayer_nchannels[idx]) +
                                                              (nchannels_tokeep - nchannels_kept))/perlayer_nchannels[idx], 1)
                    else:
                        break
                selected_perlayer_fractions[fraction] = OrderedDict(zip(self.channel_prunable, perlayer_fractions))
        else:
            # exhaustive search
            max_acc = {fraction: ((), 0) for fraction in self.fractions}
            for perlayer_fractions in itertools.product(fraction_choices, repeat=L):
                # t = time.perf_counter()
                nchannels_kept = sum(np.array(np.multiply(perlayer_fractions, perlayer_nchannels), int))
                if self.select == 'min':
                    acc = min([perlayer_acc[name][perlayer_fractions[idx]] for idx, name in enumerate(self.channel_prunable)])
                else:
                    acc = sum([perlayer_acc[name][perlayer_fractions[idx]] for idx, name in enumerate(self.channel_prunable)])

                for fraction in self.fractions:
                    # check if this combination of per layer fractions satisfy this overall fraction of channels to keep
                    if nchannels_kept <= int(fraction*nchannels) and acc >= max_acc[fraction][1]:
                        max_acc[fraction] = (perlayer_fractions, acc)
                # print("one iter time:", time.perf_counter() - t, "secs.")

            for fraction in self.fractions:
                perlayer_fractions = np.array(max_acc[fraction][0])
                nchannels_tokeep = int(fraction * nchannels)
                # use remaining quota by filling up layers gradually from one with largest fraction to smallest
                for idx in np.argsort(-perlayer_fractions):
                    nchannels_kept = sum(np.array(np.multiply(perlayer_fractions, perlayer_nchannels), int))
                    if nchannels_kept < nchannels_tokeep:
                        perlayer_fractions[idx] = np.minimum((int(perlayer_fractions[idx]*perlayer_nchannels[idx]) +
                                                            (nchannels_tokeep - nchannels_kept))/perlayer_nchannels[idx], 1)
                    else:
                        break
                selected_perlayer_fractions[fraction] = OrderedDict(zip(self.channel_prunable, perlayer_fractions))

        return selected_perlayer_fractions

    @abstractmethod
    def layer_pruned_channels(self, module_name, fractions):
        # returns a dictionary of pruned channels in corresponding module for each fraction in fractions
        # fractions should be a subset of self.fractions
        # {fraction: {list of pruned channels in this module}}
        # and optionally a dictionary listing if bias was pruned or not for each fraction, otherwise return none
        # {fraction: True if bias is pruned, False otherwise}
        # and optionally a dictionary of new params values, otherwise return none
        # {fraction: {param_name: new param of next module (numpy.ndarray)}}
        pass

    def layer_pruned_channels(self, module_name):
        return self.layer_pruned_channels(module_name, self.fractions)

    def model_pruned_channels(self):
        pruned_channels = defaultdict(OrderedDict)
        pruned_bias = defaultdict(OrderedDict)
        new_params = defaultdict(OrderedDict)
        if not self.sequential:
            for name in self.channel_prunable:
                lay_pruned_channels, lay_pruned_bias, lay_new_params = self.layer_pruned_channels(name, self.fractions)
                for fraction in self.fractions:
                    pruned_channels[fraction][name] = lay_pruned_channels[fraction]
                    pruned_bias[fraction][name] = lay_pruned_bias[fraction] if lay_pruned_bias is not None else False
                    new_params[fraction][name] = lay_new_params[fraction] if lay_new_params is not None else {}
        return pruned_channels, pruned_bias, new_params

    def apply(self, fraction):
        if self.sequential:
            assert fraction in self.fractions, "fraction should be one of the input fractions"
            for name in self.channel_prunable:
                if fraction == 1:
                    self.pruned_channels[fraction][name], self.pruned_bias[fraction][name], \
                        self.new_params[fraction][name] = [], False, None
                else:
                    lay_pruned_channels, lay_pruned_bias, lay_new_params = self.layer_pruned_channels(name, [fraction])
                    self.pruned_channels[fraction][name] = lay_pruned_channels[fraction]
                    self.pruned_bias[fraction][name] = lay_pruned_bias[fraction] if lay_pruned_bias is not None else False
                    self.new_params[fraction][name] = lay_new_params[fraction] if lay_new_params is not None else {}
                    self.reweight_layer_params(name, fraction)

                self.layer_masks(name, fraction, scale=self.scale if (hasattr(self, 'scale') and not self.reweight) else 1, perm=True)

        else:
            super().apply(fraction)
