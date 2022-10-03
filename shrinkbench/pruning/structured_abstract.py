from abc import abstractmethod
from .utils import get_params
from .structured_utils import *
from .mixin import ActivationMixin
from ..metrics import model_size
from collections import OrderedDict, defaultdict
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import itertools
import time
import math
from scipy import optimize

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

    def __init__(self, model, inputs=None, outputs=None, compressions=[1], fullnet_comp=True, reweight=False, bias=False,
                 structure=None, prune_layers=None, **pruning_params):
        inputs = torch.cat(inputs, axis=0)
        outputs = torch.cat(outputs, axis=0)
        self.device = torch.device('cpu')  # 'cuda' if torch.cuda.is_available() else 'cpu'
        super().__init__(model, inputs, outputs, compressions=compressions, fullnet_comp=fullnet_comp, reweight=reweight,
                         bias=bias, structure=structure, prune_layers=prune_layers, **pruning_params)
        self.prunable, self.channel_prunable, self.next_module_map, self.bn_map = prunable_modules(self.model, structure,
                                                                                                 prune_layers)
        self.perlayer_nchannels_dict = {name: self.module_n_channels(name, self.bias) for name in self.channel_prunable}
        self.perlayer_nchannels = np.array(list(self.perlayer_nchannels_dict.values()))
        self.size_orig,  self.size_nz_orig = model_size(model)
        print("compression ratio before pruning = ", self.size_orig / self.size_nz_orig)
        self.prunable_weights_size = {m: m.weight.numel() for m in self.prunable}
        self.prunable_bias_size = {m: m.bias.numel() if m.bias is not None else 0 for m in (self.prunable if self.bias else
                                                             [self.get_module(name) for name in self.channel_prunable])}
        self.prunable_size = sum(self.prunable_bias_size.values()) + sum(self.prunable_weights_size.values())
        self.nonprunable_size = self.size_orig - self.prunable_size

        if fullnet_comp:
            self.budgets = {compression: int(self.size_orig/compression) - self.nonprunable_size for compression in compressions}
        else:
            self.budgets = {compression: int(self.prunable_size/compression)  for compression in compressions}
        for compression in compressions:
            assert 0 <= self.budgets[compression] <= self.prunable_size, \
                f"Cannot compress to {1/compression} model with {self.nonprunable_size/self.size_nz_orig}" + \
                "fraction of unprunable parameters"


        if compressions == [1]:  # prune nothing
            self.pruned_channels = {1: {name: [] for name in self.channel_prunable}}
            self.pruned_bias = {1: {name: False for name in self.channel_prunable}}
            self.new_params = {1: {name: {} for name in self.channel_prunable}}
            if isinstance(self, LayerStructuredPruning):
                self.perlayer_fractions = {1: {name: 1 for name in self.channel_prunable}}
            else:
                self.fractions = {1: 1}
        else:
            self.preprocessing()
            self.pruned_channels, self.pruned_bias, self.new_params = self.model_pruned_channels()
            # if self.pruned_bias is None:
            #     self.pruned_bias = {compression: {name: False for name in self.channel_prunable} for compression in compressions}

    def prunable_pruned_size(self, perlayer_budgets):
        # returns pruned size of the prunable part of the net, takes rounding into account
        # perlayer_budgets should correspond to same order as in perlayer_nchannels
        # TODO: modify to handle self.bias = True case
        pruned_weights_size, pruned_bias_size = self.prunable_weights_size.copy(), self.prunable_bias_size.copy()
        for i, name in enumerate(self.channel_prunable):
            mod, next_mod = self.get_module(name), self.get_module(name, next=True)
            fraction = perlayer_budgets[i] / self.perlayer_nchannels[i]
            pruned_weights_size[mod] *= fraction
            pruned_weights_size[next_mod] *= fraction
            pruned_bias_size[mod] *= fraction
        return sum(pruned_weights_size.values()) + sum(pruned_bias_size.values())

    def can_prune(self, module_name):
        # true for any module where output channels can be pruned
        return module_name in self.channel_prunable

    def apply(self, compression):
        assert compression in self.compressions, "compression should be one of the input compressions"
        if compression != 1:
            self.reweight_params(compression)
        masks = self.model_masks(compression)

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
    def _compress_once(self, fraction):
        # input: fraction of prunable channels to keep
        # output: a dictionary of pruned channels for each module_name
        #  {module_name: list of pruned channels in this module}
        #  and a dictionary listing if bias was pruned or not for each module_name
        #  {module_name: True if bias is pruned, False otherwise}
        #  and a dictionary of new params values for each module_name
        #  {module_name: {param_name: new param of next module (numpy.ndarray)}}
        pass

    def model_pruned_channels(self):
        # search for fraction of prunable channels to keep that yields # of prunable params ~= budget
        # returns a dictionary of pruned channels for each compression & module_name
        # {compression: {module_name: list of pruned channels in this module}}
        # and a dictionary listing if bias was pruned or not for each compression & module_name
        # {compression: {module_name: True if bias is pruned, False otherwise}}
        # and a dictionary of new params values for each compression & module_name
        # {compression: {module_name: {param_name: new param of next module (numpy.ndarray)}}}

        pruned_channels, pruned_bias, new_params, fractions = {}, {}, {}, {}
        for compression in self.compressions:
            def f_opt(fraction):
                # print("try fraction = ", fraction)
                pruned_channels  = self._compress_once(fraction)[0]
                prunable_pruned_size = self.prunable_pruned_size([self.perlayer_nchannels_dict[name] -
                                                        len(pruned_channels[name]) for name in self.channel_prunable])
                # print("current budget difference is: ", self.budgets[compression] - prunable_pruned_size)
                return self.budgets[compression] - prunable_pruned_size
            fractions[compression] = optimize.brentq(f_opt, 0, 1, maxiter=1000, xtol=1e-12, disp=True)
            pruned_channels[compression], pruned_bias[compression], new_params[compression] = self._compress_once(fractions[compression])
            if self.prunable_pruned_size([self.perlayer_nchannels_dict[name] - len(pruned_channels[compression][name])
                                          for name in self.channel_prunable]) > self.budgets[compression]:
                # we don't need to do this more than once since this change of fraction is larger than convergence tol
                fractions[compression] -= 1/sum(self.perlayer_nchannels)
                pruned_channels[compression], pruned_bias[compression], new_params[compression] = self._compress_once(fractions[compression])

            if pruned_bias[compression]=={}:
                pruned_bias[compression] = {name: False for name in self.channel_prunable}
            if new_params[compression]=={}:
                new_params[compression] = {name: {} for name in self.channel_prunable}
        self.fractions = fractions
        return pruned_channels, pruned_bias, new_params

    def layer_masks(self, module_name, compression, masks=None, scale=1, perm=False):
        if masks is None:
            masks = defaultdict(OrderedDict)

        module = self.get_module(module_name)
        pruned_channels = self.pruned_channels[compression][module_name]

        bn_name = self.bn_map[module_name]
        for mod in [module] + ([self.get_module(bn_name)] if bn_name is not None else []):
            for name in get_params(mod).keys():  # ['weight', 'bias']:
                indices_structured(mod, name, 0, pruned_channels)
                if perm:  # Multiply params by (non-scaled) masks so metrics can count nonzeros
                    getattr(mod, name + '_orig').data.mul_(getattr(mod, name + '_mask') != 0)
                masks[mod][name] = getattr(mod, name + '_mask')

        next_module = self.get_module(module_name, next=True)
        # map pruned channels of current layer to next one (needed if mod is conv and next mod is linear)
        pruned_channels_next = map_chton(pruned_channels, int(next_module.weight.shape[1]/module.weight.shape[0]))
        # scale only next layer weights
        indices_structured(next_module, 'weight', 1, pruned_channels_next, scale)
        if perm:  # Multiply params by (non-scaled) masks so metrics can count nonzeros
            next_module.weight_orig.data.mul_(next_module.weight_mask != 0)
        masks[next_module]['weight'] = getattr(next_module, 'weight_mask')
        if self.pruned_bias[compression][module_name] and next_module.bias is not None:
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

    def model_masks(self, compression):
        masks = defaultdict(OrderedDict)
        for module_name in self.channel_prunable:
            self.layer_masks(module_name, compression, masks, scale=self.scale if (hasattr(self, 'scale') and not self.reweight) else 1, perm=True)
        return masks

    def reweight_layer_params(self, module_name, compression):
        next_module = self.get_module(module_name, next=True)
        new_params = self.new_params[compression][module_name]
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
                                set(self.pruned_channels[compression][module_name]))

            # if module is linear, kernel_size = 1, if it's conv followed by linear, kernel_size = H x W
            kernel_size = next_module.kernel_size if isinstance(next_module, nn.Conv2d) else \
                int(acts.shape[-1]/self.module_n_channels(module_name))
            map = lambda channels: map_chton(channels, kernel_size)
            neg_input_change = NegInputChange(acts, params, rwchange=True, bias=self.bias, map=map)
            new_params = neg_input_change.get_new_params(kept_channels, cur=False)
            self.new_params[compression][module_name] = new_params

        for pname, new_param in new_params.items():
            with torch.no_grad():  # applied before pruning, so we're modifying weight/bias not weight_orig/bias_orig
                getattr(next_module, pname).data = torch.from_numpy(new_param).type(getattr(next_module, pname).dtype).to(self.device)

    def reweight_params(self, compression):
        # if self.new_params is None:
        #     self.new_params = {compression: {name: {} for name in self.channel_prunable} for compression in self.compressions}
        for name in self.channel_prunable:
            self.reweight_layer_params(name, compression)

    def summary(self, compression):
        total_n_channels_kept = 0
        rows = []
        for name, module in self.model.named_modules():
            for pname, param in get_params(module).items():
                if hasattr(module, pname+'_mask'):  # isinstance(module, MaskedModule):
                    real_compression = 1/(getattr(module, pname+'_mask').detach().cpu().numpy() != 0).mean()
                else:
                    real_compression = 1

                shape = param.shape
                if name in self.channel_prunable:
                    if pname == 'weight':
                        pruned_channels = self.pruned_channels[compression][name]
                        pruned_fraction = len(pruned_channels)/shape[0]
                        n_channels = self.module_n_channels(name, bias=self.bias)
                        if isinstance(self, LayerStructuredPruning) and self.__class__.__name__ != "LayerSamplingChannel":
                            k = int(self.perlayer_fractions[compression][name] * n_channels)
                            assert n_channels - len(pruned_channels) - self.pruned_bias[compression][name] == k, \
                                f"# of kept channels in {name} should be {k}"
                        total_n_channels_kept += n_channels - len(pruned_channels) - self.pruned_bias[compression][name]
                    else:
                        pruned_channels = [0] if self.pruned_bias[compression][name] else []
                        pruned_fraction = len(pruned_channels)

                else:
                    pruned_channels = []
                    pruned_fraction = 0

                rows.append([name, pname, real_compression, np.prod(shape), shape, self.can_prune(name), pruned_fraction, pruned_channels])

        if not isinstance(self, LayerStructuredPruning):
            k = int(self.fractions[compression] * self.n_channels(bias=self.bias))
            assert total_n_channels_kept == k, f"# of total kept channels should be {k}, {total_n_channels_kept} " \
                f"channels were kept"

        columns = ['module', 'param', 'comp', 'size', 'shape', 'channel prunable', 'pruned_fraction', 'pruned_channels']
        return pd.DataFrame(rows, columns=columns)


class LayerStructuredPruning(StructuredPruning):
    # prunes output channels of each prunable layer independently if sequential=False or sequentially starting from
    # first layer to last one. Uses perlayer fraction selection method from Kuzmin, Andrey, et al. if
    # onelayer_results_df is not none, otherwise uses equal compression for all layers.

    def __init__(self, model, inputs=None, outputs=None, compressions=1, fullnet_comp=True, reweight=False, bias=False,
                 structure='neuron', prune_layers=None, sequential=False, onelayer_results_df=None, select='min',
                 **pruning_params):

        self.onelayer_results_df, self.select = onelayer_results_df, select
        self.perlayer_fractions = {}
        super().__init__(model, inputs=inputs, outputs=outputs, compressions=compressions, fullnet_comp=fullnet_comp,
                         reweight=reweight, bias=bias, structure=structure, prune_layers=prune_layers,
                         sequential=sequential, **pruning_params)

    def preprocessing(self):
        if self.compressions != [1] and len(self.channel_prunable) > 1 and self.onelayer_results_df is not None:
            t = time.perf_counter()
            self.perlayer_fractions = self.select_perlayer_fractions()
            print("perlayer fractions selection time:", time.perf_counter() - t, "secs.")
        else:
            # solve quadratic equation to find fraction needed to have # of prunable params kept = budget
            mod_pruned_twice = {self.get_module(name) for name in set(self.channel_prunable).intersection(set(self.next_module_map.values()))}
            mod_pruned_once = set(self.prunable).difference(mod_pruned_twice)
            a = sum([self.prunable_weights_size[m] for m in mod_pruned_twice])
            b = sum([self.prunable_weights_size[m] for m in mod_pruned_once]) + sum(self.prunable_bias_size.values())
            pos_root = lambda c: -c/b if a == 0 else (-b + math.sqrt(b**2 - 4*a*c))/(2*a)
            self.perlayer_fractions = {compression: {name: 1 if compression == 1 else pos_root(-self.budgets[compression])
                                                     for name in self.channel_prunable} for compression in self.compressions}

    # TODO: save selected perlayer budgets instead of fractions, since all methods use that
    # TODO: add option to prune each layer until reaching an error threshold
    def select_perlayer_fractions(self, eps=1e-9):
        # implements equal accuracy loss method proposed in Kuzmin, Andrey, et al. "Taxonomy and evaluation of
        # structured compression of convolutional neural networks." arXiv preprint arXiv:1912.09802 (2019).
        # but here we select n_channels instead of ranks

        self.onelayer_results_df.loc[:, 'compression'] = self.onelayer_results_df.compression.round(3)  # to avoid numerical issues
        compression_choices = np.sort(self.onelayer_results_df['compression'].unique())[::-1] # sort CR in decreasing order
        perlayer_acc = defaultdict(np.array) # if bs else defaultdict(OrderedDict)
        nchannels = sum(self.perlayer_nchannels)
        selected_perlayer_fractions = defaultdict(OrderedDict)

        # get acc for all compression choices
        for name in self.channel_prunable:
            # if bs:
            perlayer_acc[name] = np.zeros(len(compression_choices))
            for idx, compression in enumerate(compression_choices):
                # if bs:
                perlayer_acc[name][idx] = self.onelayer_results_df[(self.onelayer_results_df['prune_layers'] == name)
                                             & (self.onelayer_results_df['compression'] == compression)]['pre_acc1'].values[0]
                # else:
                #     perlayer_acc[name][fraction] = self.onelayer_results_df[(self.onelayer_results_df['prune_layers'] == name)
                #                                   & (self.onelayer_results_df['fraction'] == fraction)]['pre_acc1'].values[0]

        L = len(self.channel_prunable)

        # binary search
        assert self.select == 'min', "select should be min if using binary search"
        assert 1 in compression_choices, "1 should be among compression choices"  # because we use it for acc_max
        # acc at compression 1 is not exactly the same due to numerical errors, we take min to guarantee acc_1 is satisfied by all layers
        acc_1 = self.onelayer_results_df[self.onelayer_results_df['compression']==1]['pre_acc1'].min()
        # enforce monotonicity
        for name in self.channel_prunable:
            perlayer_acc[name] = monotone_envelope(1/compression_choices, perlayer_acc[name])

        for compression in self.compressions:
            # nchannels_tokeep = int(fraction * nchannels)
            # min_budget = int(nchannels_tokeep >= L)
            min_budgets =  [1]*L #list(1 / np.array(self.perlayer_nchannels))
            if self.prunable_pruned_size(min_budgets) > self.budgets[compression]:
                min_budgets = [0]*L
            # find perlayer fractions that lead to largest acc and satisfy nchannels_tokeep
            acc_min, acc_max = 0, acc_1
            min_fractions = min_budgets/self.perlayer_nchannels #np.array(self.perlayer_nchannels)
            perlayer_fractions = min_fractions.copy()
            while acc_max - acc_min > eps:
                acc = (acc_min + acc_max)/2
                # find smallest fraction satisfying this acc for each layer, acc is at least satisfied by compression=1
                for i, name in enumerate(self.channel_prunable):
                    idx = max(0, min(np.searchsorted(perlayer_acc[name], acc, side='left'), len(compression_choices)-1)) # make sure we don't go out of bounds
                    perlayer_fractions[i] = max(1/compression_choices[idx], min_fractions[i])
                # nchannels_kept = sum(np.array(np.multiply(perlayer_fractions, perlayer_nchannels), int))
                # check if this combination of per layer fractions satisfy this overall fraction of channels to keep
                if self.prunable_pruned_size((perlayer_fractions*self.perlayer_nchannels).astype(int)) <= self.budgets[compression]:
                    acc_min = acc
                else:
                    acc_max = acc

            # check if budget is satisfied at convergence, if not rerun with acc=acc_min
            if self.prunable_pruned_size((perlayer_fractions*self.perlayer_nchannels).astype(int)) > self.budgets[compression]:
                for i, name in enumerate(self.channel_prunable):
                    idx = max(0, min(np.searchsorted(perlayer_acc[name], acc_min, side='left'), len(compression_choices)-1)) # make sure we don't go out of bounds
                    perlayer_fractions[i] = max(1/compression_choices[idx], min_fractions[i])

            # use remaining quota if any by filling up layers gradually from one with largest fraction to smallest
            perlayer_fractions = np.array(perlayer_fractions)
            for idx in np.argsort(-perlayer_fractions):
                # nchannels_kept = sum(np.array(np.multiply(perlayer_fractions, perlayer_nchannels), int))
                prunable_pruned_size = self.prunable_pruned_size((perlayer_fractions*self.perlayer_nchannels).astype(int))
                if  prunable_pruned_size < self.budgets[compression]:
                    size_perchannel = self.prunable_pruned_size((perlayer_fractions*self.perlayer_nchannels).astype(int)
                                                                +(np.arange(L)==idx)) - prunable_pruned_size
                    nchannels_toadd = int((self.budgets[compression] - prunable_pruned_size)/size_perchannel)
                    perlayer_fractions[idx] = np.minimum((int(perlayer_fractions[idx]*self.perlayer_nchannels[idx]) +
                                                          nchannels_toadd)/self.perlayer_nchannels[idx], 1)
                else:
                    break
            selected_perlayer_fractions[compression] = OrderedDict(zip(self.channel_prunable, perlayer_fractions))

        # # exhaustive search
        # max_acc = {fraction: ((), 0) for fraction in self.fractions}
        # for perlayer_fractions in itertools.product(fraction_choices, repeat=L):
        #     # t = time.perf_counter()
        #     nchannels_kept = sum(np.array(np.multiply(perlayer_fractions, perlayer_nchannels), int))
        #     if self.select == 'min':
        #         acc = min([perlayer_acc[name][perlayer_fractions[idx]] for idx, name in enumerate(self.channel_prunable)])
        #     else:
        #         acc = sum([perlayer_acc[name][perlayer_fractions[idx]] for idx, name in enumerate(self.channel_prunable)])
        #
        #     for fraction in self.fractions:
        #         # check if this combination of per layer fractions satisfy this overall fraction of channels to keep
        #         if nchannels_kept <= int(fraction*nchannels) and acc >= max_acc[fraction][1]:
        #             max_acc[fraction] = (perlayer_fractions, acc)
        #     # print("one iter time:", time.perf_counter() - t, "secs.")
        #
        # for fraction in self.fractions:
        #     perlayer_fractions = np.array(max_acc[fraction][0])
        #     nchannels_tokeep = int(fraction * nchannels)
        #     # use remaining quota by filling up layers gradually from one with largest fraction to smallest
        #     for idx in np.argsort(-perlayer_fractions):
        #         nchannels_kept = sum(np.array(np.multiply(perlayer_fractions, perlayer_nchannels), int))
        #         if nchannels_kept < nchannels_tokeep:
        #             perlayer_fractions[idx] = np.minimum((int(perlayer_fractions[idx]*perlayer_nchannels[idx]) +
        #                                                 (nchannels_tokeep - nchannels_kept))/perlayer_nchannels[idx], 1)
        #         else:
        #             break
        #     selected_perlayer_fractions[fraction] = OrderedDict(zip(self.channel_prunable, perlayer_fractions))

        return selected_perlayer_fractions

    def _compress_once(self, fraction):
        pass

    @abstractmethod
    def layer_pruned_channels(self, module_name, compressions):
        # returns a dictionary of pruned channels in corresponding module for each compression in compressions
        # compressions should be a subset of self.compressions
        # {compression: {list of pruned channels in this module}}
        # and optionally a dictionary listing if bias was pruned or not for each compression, otherwise return none
        # {compression: True if bias is pruned, False otherwise}
        # and optionally a dictionary of new params values, otherwise return none
        # {compression: {param_name: new param of next module (numpy.ndarray)}}
        pass

    def model_pruned_channels(self):
        pruned_channels = defaultdict(OrderedDict)
        pruned_bias = defaultdict(OrderedDict)
        new_params = defaultdict(OrderedDict)
        if not self.sequential:
            for name in self.channel_prunable:
                lay_pruned_channels, lay_pruned_bias, lay_new_params = self.layer_pruned_channels(name, self.compressions)
                for compression in self.compressions:
                    pruned_channels[compression][name] = lay_pruned_channels[compression]
                    pruned_bias[compression][name] = lay_pruned_bias[compression] if lay_pruned_bias is not None else False
                    new_params[compression][name] = lay_new_params[compression] if lay_new_params is not None else {}
        return pruned_channels, pruned_bias, new_params

    def apply(self, compression):
        if self.sequential:
            assert compression in self.compressions, "compression should be one of the input compressions"
            for name in self.channel_prunable:
                if compression == 1:
                    self.pruned_channels[compression][name], self.pruned_bias[compression][name], \
                        self.new_params[compression][name] = [], False, {}
                else:
                    lay_pruned_channels, lay_pruned_bias, lay_new_params = self.layer_pruned_channels(name, [compression])
                    self.pruned_channels[compression][name] = lay_pruned_channels[compression]
                    self.pruned_bias[compression][name] = lay_pruned_bias[compression] if lay_pruned_bias is not None else False
                    self.new_params[compression][name] = lay_new_params[compression] if lay_new_params is not None else {}
                    self.reweight_layer_params(name, compression)

                self.layer_masks(name, compression, scale=self.scale if (hasattr(self, 'scale') and not self.reweight) else 1, perm=True)

        else:
            super().apply(compression)
