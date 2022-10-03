"""Structured pruning

Implements structured pruning strategies
"""

from ..pruning import *
from .structured_utils import *
import numpy as np
import torch
import torch.nn as nn
from PySSM import Greedyint, Greedypair, SieveStreamingint, SieveStreamingpair
import sys
import copy
from torch.distributions.multinomial import Multinomial
from collections import OrderedDict, defaultdict
from scipy import optimize
import time


class WeightNormChannel(StructuredPruning):
    # prunes fraction of channels with the lowest input/output weight norm, with ord = order of the norm
    # normalized by # of input/output weights if norm = True
    # when ord = 1, this computes the onorm (if onorm=True) and inorm (if onorm=False) functions proposed in
    # "T. He, Y. Fan, Y. Qian, T. Tan, and K. Yu. Reshaping deep neural network for fast decoding by node-pruning,
    # ICASSP, 2014."

    def __init__(self, model,  inputs=None, outputs=None, compressions=1, fullnet_comp=True, reweight=False, bias=False,
                 structure='neuron', prune_layers=None, onorm=True, ord=1, norm=True):
        super().__init__(model, inputs, outputs, compressions=compressions, fullnet_comp=fullnet_comp, reweight=reweight,
                         bias=bias, structure=structure, prune_layers=prune_layers, onorm=onorm, ord=ord, norm=norm)

    def preprocessing(self):
        weight_norm = lambda w: np.linalg.norm(w, ord=self.ord, axis=1) / (w.shape[1] if self.norm else 1)
        def importance_fn(param, name):
            W = param['weight']
            if self.onorm:
                n_channels = self.module_n_channels(name)
                W = W.reshape(W.shape[0], n_channels, -1).swapaxes(0, 1) # reshape needed if module is conv followed by linear
            W = W.reshape(W.shape[0], -1)  # needed if structure = channel
            return weight_norm(W)
        # axis = 1 if self.onorm else 0
        # weight_norm(np.swapaxes(param['weight'], 0, axis).reshape(param['weight'].shape[axis], -1))
        importances = map_channel_importances(importance_fn, self.params(next=self.onorm))
        V = self.channels_ind()
        self.order = sorted(dict_to_list(V), key=lambda e: importances[e[0]][e[1]])  # e has form (name, index in layer)

    def _compress_once(self, fraction):
        k = int(len(self.order) * fraction)
        pruned_channels = list_to_dict(self.order[:-k] if k != 0 else self.order, self.channel_prunable)
        return pruned_channels, {}, {}


class LayerWeightNormChannel(LayerStructuredPruning):
    # prunes fraction of channels with the lowest output (if onorm=True) or input (if onorm=False) weight norm,
    # in each prunable layer independently, with ord = order of the norm
    # when ord = 1, this computes the onorm (if onorm=True) and inorm (if onorm=False) functions proposed in
    # "T. He, Y. Fan, Y. Qian, T. Tan, and K. Yu. Reshaping deep neural network for fast decoding by node-pruning,
    # ICASSP, 2014."

    def __init__(self, model, inputs=None, outputs=None, fullnet_comp=True, compressions=1, reweight=False, bias=False,
                 structure='neuron', prune_layers=None, onelayer_results_df=None, select="min", onorm=True, ord=1):
        super().__init__(model, inputs, outputs, compressions=compressions, fullnet_comp=fullnet_comp, reweight=reweight,
                         bias=bias, structure=structure, prune_layers=prune_layers,
                         onelayer_results_df=onelayer_results_df, select=select, onorm=onorm, ord=ord)

    def layer_pruned_channels(self, module_name, compressions):
        W = self.module_params(module_name, next=self.onorm)['weight']
        if self.onorm:
            n_channels = self.module_n_channels(module_name)
            W = W.reshape(W.shape[0], n_channels, -1).swapaxes(0, 1) # reshape needed if module is conv followed by linear
        W = W.reshape(W.shape[0], -1)  # needed if structure = channel
        importances = np.linalg.norm(W, ord=self.ord, axis=1)
        order = np.argsort(importances)  # increasing order
        ks = {compression: int(len(order) * self.perlayer_fractions[compression][module_name]) for compression in compressions}
        pruned_channels = {compression: order[:-ks[compression]] if ks[compression] != 0 else order for compression in compressions}
        return pruned_channels, None, None


class ActGradChannel(StructuredPruning, ActivationMixin, GradientMixin):
    # prunes fraction of channels with the lowest abs(activation * gradient), averaged over nbatches of inputs/outputs,
    # normalized by the l2-norm of these importance scores if norm=True

    def __init__(self, model, inputs=None, outputs=None, compressions=1, fullnet_comp=True, reweight=False, bias=False, structure='neuron',
                 prune_layers=None, norm=True):
        super().__init__(model, inputs, outputs, compressions=compressions, fullnet_comp=fullnet_comp, reweight=reweight,
                         bias=bias, structure=structure, prune_layers=prune_layers, norm=norm)

    def preprocessing(self):
        grads = self.gradients(only_input=True, update=True, only_prunable=True)
        acts = self.activations(only_input=True, update=True, only_prunable=True)

        # inner sum needed for structure = channel
        def act_grad_imp(name):
            mod = self.get_module(name, next=True)
            # reshape needed if module is conv followed by linear
            n_channels = self.module_n_channels(name)
            acts[mod] = acts[mod].reshape(acts[mod].shape[0], n_channels, -1)
            grads[mod] = grads[mod].reshape(grads[mod].shape[0], n_channels, -1)
            return np.abs(np.sum(acts[mod] * grads[mod], tuple(range(2, acts[mod].ndim)))).sum(0) / acts[mod].shape[0]

        importances = {name: act_grad_imp(name) for name in self.channel_prunable}
        if self.norm:
            importances = {name: imp/np.linalg.norm(imp) for name, imp in importances.items()}
        V = self.channels_ind()
        self.order = sorted(dict_to_list(V), key=lambda e: importances[e[0]][e[1]])  # e has form (name, index in layer)

    def _compress_once(self, fraction):
        k = int(len(self.order) * fraction)
        pruned_channels = list_to_dict(self.order[:-k] if k != 0 else self.order, self.channel_prunable)
        return pruned_channels, {}, {}


class LayerActGradChannel(LayerStructuredPruning, ActivationMixin, GradientMixin):
    # prunes fraction of channels with the lowest abs(activation * gradient), averaged over nbatches of inputs/outputs,
    # in each prunable layer independently.
    def __init__(self, model, inputs=None, outputs=None, compressions=1, fullnet_comp=True, reweight=False, bias=False,
                 structure='neuron', prune_layers=None, onelayer_results_df=None, select="min"):
        super().__init__(model, inputs, outputs, compressions=compressions, fullnet_comp=fullnet_comp, reweight=reweight,
                         bias=bias, structure=structure, prune_layers=prune_layers,
                         onelayer_results_df=onelayer_results_df, select=select)

    def layer_pruned_channels(self, module_name, compressions):
        # we use input acts & grad of next layer, because output acts of current layer are before the act function
        # input acts & grad of next layer are after act fct, maxpool, dropout, and batchnorm (if any)
        next_module = self.get_module(module_name, next=True)
        # gradients & activations (of all prunable layers) will be computed in first call only
        # no need to update since not sequential
        acts = self.module_activations(next_module, only_input=True, update=False)
        grads = self.module_gradients(next_module, only_input=True, update=False)
        # reshape needed if module is conv followed by linear
        n_channels = self.module_n_channels(module_name)
        acts = acts.reshape(acts.shape[0], n_channels, -1)
        grads = grads.reshape(grads.shape[0], n_channels, -1)
        # inner sum needed for structure = channel
        importances = np.abs(np.sum(acts * grads, tuple(range(2, acts.ndim)))).sum(0) / acts.shape[0]
        order = np.argsort(importances)  # increasing order
        ks = {compression: int(len(order) * self.perlayer_fractions[compression][module_name]) for compression in compressions}
        pruned_channels = {compression: order[:-ks[compression]] if ks[compression] != 0 else order for compression in compressions}
        return pruned_channels, None, None


class RandomChannel(StructuredPruning):
    # randomly prunes fraction of channels in all prunable layers
    def preprocessing(self):
        # assign random importances
        self.importances = map_channel_importances(lambda param, name: np.random.uniform(0, 1, size=param['weight'].shape[0]),
                                              self.params())
        flat_importances = flatten_channel_importances(self.importances)
        self.importances_sorted = np.sort(flat_importances)

    def _compress_once(self, fraction):
        threshold =  self.importances_sorted[-int(len(self.importances_sorted) * fraction) - 1] if fraction < 1 else - 1
        pruned_channels = map_channel_importances(lambda param, name: prune_threshold_channels(param, threshold),
                                                  self.importances)
        return pruned_channels, {}, {}


class LayerRandomChannel(LayerStructuredPruning):
    # randomly prunes fraction of channels in each prunable layer independently
    def layer_pruned_channels(self, module_name, compressions):
        params = self.module_params(module_name)
        pruned_channels = {compression: prune_random_channels(params['weight'], self.perlayer_fractions[compression]
        [module_name]) for compression in compressions}
        return pruned_channels, None, None


# TODO: remove this strategy
class WeightChangeChannel(StructuredPruning):
    # prunes fraction of channels that lead to the smallest sum of change of input/output weight
    # where each term is normalized by the Frobenius norm of input/output weight of corresponding layer if norm=True
    # if reweight=True, we update next layer weights with weights that minimize change of input to next layer
    # if bias = True, we treat the bias as another channel (with no input weights) that can be pruned (only allowed when out=True)

    def __init__(self, model,  inputs=None, outputs=None, compressions=1, out=False, reweight=False, bias=False,
                 structure='neuron', prune_layers=None, norm=0, algo="sieve"):
        if bias:
            assert out is True, "cannot prune bias with out=False"
        super().__init__(model,  inputs, outputs, compressions=compressions, out=out, reweight=reweight, bias=bias,
                         structure=structure, prune_layers=prune_layers, norm=norm, algo=algo)

    def model_pruned_channels(self):
        V = {name: set(range(self.module_n_channels(name, bias=self.bias))) for name in self.channel_prunable}
        V_all = dict_to_list(V)
        params = self.params(next=self.out)
        n_channels = len(V_all)
        maps = None
        for name in self.channel_prunable:
            mod = self.get_module(name, next=self.out)
            maps[name] = lambda channels: map_chton(channels, mod.kernel_size if isinstance(mod, nn.Conv2d) else 1)

        pruned_channels = None
        for compression in self.compressions:
            k = int(fraction * n_channels)
            sum_neg_weight_change = SumNegWeightChange(params, self.out, self.bias, self.norm, maps)
            if self.algo == "sieve":
                opt = SieveStreamingpair(k, sum_neg_weight_change, sum_neg_weight_change.fe_max(V_all), 0.1)
            elif self.algo == "greedy":
                opt = Greedypair(k, sum_neg_weight_change)
            else:
                raise NotImplementedError("Selected algorithm is not supported")
            opt.fit(V_all)
            kept_channels = opt.get_solution()
            if len(kept_channels) < k:
                kept_channels = fillup_sol(kept_channels, opt.get_f(), V_all, k)
            kept_channels = list_to_dict(kept_channels)
            pruned_channels[fraction] = {name: list(V[name] - set(kept_channels[name])) for name in self.channel_prunable}

        return pruned_channels, False, None

# TODO: remove this strategy
class LayerWeightChangeChannel(LayerStructuredPruning):
    # Sequentially prunes fraction of channels in each prunable layer starting from first one,
    # for each layer, prunes the channels that lead to the smallest change of input/output weight
    # if reweight=True, we update next layer weights with weights that minimize change of input to next layer
    # if bias = True, we treat the bias as another channel (with no input weights) that can be pruned (only allowed when out=True)

    def __init__(self, model,  inputs=None, outputs=None, compressions=1, out=False, reweight=False, bias=False,
                 structure='neuron', prune_layers=None, sequential=False, onelayer_results_df=None, select="min",
                 norm=0, algo="sieve"):
        if bias:
            assert out is True, "cannot prune bias with out=False"
        super().__init__(model,  inputs, outputs, compressions=compressions, out=out, reweight=reweight, structure=structure,
                         prune_layers=prune_layers, sequential=sequential, onelayer_results_df=onelayer_results_df,
                         select=select, bias=bias, norm=norm, algo=algo)

    def layer_pruned_channels(self, module_name, compressions):
        next_module = self.get_module(module_name, next=self.out)
        params = get_params(next_module)
        n_channels = self.module_n_channels(module_name, bias=self.bias)
        V = list(range(n_channels))
        map = lambda channels: map_chton(channels, next_module.kernel_size if isinstance(next_module, nn.Conv2d) else 1)
        # if self.algo == "greedy_old":
        #     update_neg_input_change = self.get_neg_input_change_fct(self.get_module(module_name, next=True),
        #                                                             rwchange=self.rwchange, update=True)
        #     kept_channels, new_params = greedy_algo(update_neg_input_change, int(fraction * n_channels), V,
        #                                      save_info_key="W_new" if self.rwchange else None)
        # else:
        neg_weight_change = NegWeightChange(params, self.out, self.bias, self.norm, map)
        pruned_channels, pruned_bias, new_params = {}, {}, {}
        for compression in compressions:
            k = int(self.perlayer_compressions[compression][module_name] * n_channels)
        if self.algo == "sieve":
            opt = SieveStreamingint(k, neg_weight_change, neg_weight_change.fe_max(V), 0.1)
        elif self.algo == "greedy":
            opt = Greedyint(k, neg_weight_change)
        else:
            raise NotImplementedError("Selected algorithm is not supported")
        opt.fit(V)
        kept_channels = opt.get_solution()
        F = opt.get_f()
        if len(kept_channels) < k:
            kept_channels = fillup_sol(kept_channels, F, V, k)
        pruned_channels[compression] = list(set(V) - set(kept_channels))
        return pruned_channels, False, None

# TODO: remove this strategy
class InChangeChannel(StructuredPruning, ActivationMixin):
    # prunes fraction of channels that lead to the smallest sum of change of input in each prunable layer
    # with (rwchange=True) or without (rwchange=False) reweighting of next layer weights,
    # if reweight=True, we update next layer weights with weights that minimize change of input to next layer
    # if bias = True, we treat the bias as another channel (with no input weights) that can be pruned
    # where each term is normalized by the Frobenius norm of input of corresponding layer if norm=2, by # of rows of
    # activation matrix if norm=1, and no normalization if norm=0

    # TODO: obj fct is not monotone nor submodular when rwchange=False, add an algo for this case
    #  (both greedy and sieve will not work well)
    def __init__(self, model, inputs=None, outputs=None, compressions=1, reweight=False, bias=False, structure='neuron',
                 prune_layers=None, norm=0, rwchange=False, algo="sieve", patches="all"):
        super().__init__(model, inputs, outputs, compressions=compressions, reweight=reweight, bias=bias, structure=structure,
                         prune_layers=prune_layers, norm=norm, rwchange=rwchange, algo=algo, patches=patches)

    def model_pruned_channels(self):
        V = self.channels_ind()
        V_all = dict_to_list(V)

        acts_all = self.activations(only_input=True)
        acts, maps = {}, {}
        for name in self.channel_prunable:
            next_module = self.get_module(name, next=True)
            acts[name] = acts_all[next_module]
            if isinstance(next_module, nn.Conv2d):
                # acts shape changes from (nbatches, in_channels, H, W) to (nbatches, npatches, in_channels x kernel_size)
                # after unfold, and transpose, then to (nbatches x npatches, in_channels x kernel_size) after reshape
                acts[name] = torch.nn.functional.unfold(torch.from_numpy(acts[name]), next_module.kernel_size,
                                  next_module.dilation, next_module.padding, next_module.stride).transpose(1, 2).numpy()
                acts[name] = acts[name].reshape(-1, acts[name].shape[-1])

            maps[name] = lambda channels: map_chton(channels, next_module.kernel_size if isinstance(next_module,
                                                                                                    nn.Conv2d) else 1)

        params = self.params(next=True)
        n_channels = len(V_all)

        # if self.algo == "greedy_old":
        #     update_neg_input_change_fcts = {name: self.get_neg_input_change_fct(self.get_module(name, next=True),
        #                                                                         rwchange=self.rwchange)
        #                                     for name in self.channel_prunable}
        #     H_next = lambda i, S, info_S: update_decomposable_score(update_neg_input_change_fcts, i, S, info_S)
        #     ks = np.array(np.array(self.fractions) * len(V_all), int)
        #
        #     kept_channels, new_params = greedy_algo(H_next, ks, V_all, save_info_key="W_new" if self.rwchange else None)
        #
        #     kept_channels = {k: list_to_dict(v) for k, v in kept_channels.items()}
        #     pruned_channels = {fraction: {name: list(V[name] - set(kept_channels[ks[i]][name]))
        #                                      for name in self.channel_prunable} for i, fraction in enumerate(self.fractions)}
        #     if self.rwchange:
        #         new_params = {fraction: new_params[ks[i]] for i, fraction in enumerate(self.fractions)}
        # else:

        pruned_channels, pruned_bias, new_params = {}, {}, {}
        for compression in self.compressions:
            k = int(fraction * n_channels)
            sum_neg_input_change = SumNegInputChange(acts, params, self.rwchange, self.bias, self.norm, maps)
            if self.algo == "sieve":
                opt = SieveStreamingpair(k, sum_neg_input_change, sum_neg_input_change.fe_max(V_all), 0.1)
            elif self.algo == "greedy":
                opt = Greedypair(k, sum_neg_input_change)
            else:
                raise NotImplementedError("Selected algorithm is not supported")
            opt.fit(V_all)
            kept_channels = opt.get_solution()
            F = opt.get_f()
            if len(kept_channels) < k:
                kept_channels = fillup_sol(kept_channels, F, V_all, k, update=True)
            if self.reweight:
                new_params[fraction] = F.get_new_params(kept_channels)
            kept_channels = list_to_dict(kept_channels)
            pruned_channels[fraction] = {name: list(V[name] - set(kept_channels[name])) for name in self.channel_prunable}
            pruned_bias[fraction] = {name: False for name in self.channel_prunable}
            for name in self.channel_prunable:
                i_bias = self.module_n_channels(name)
                if i_bias in pruned_channels[fraction][name]:  # check if "bias channel" is pruned
                    pruned_channels[fraction][name].remove(i_bias)
                    pruned_bias[fraction][name] = True

        return pruned_channels, pruned_bias, new_params


class LayerInChangeChannel(LayerStructuredPruning, ActivationMixin):
    # Prunes fraction of channels in each prunable layer independently if sequential=False, or sequentially
    # starting from first one if sequential=True
    # for each layer, prunes the channels that lead to the smallest change of input to next layer,
    # with (rwchange=True) or without (rwchange=False) reweighting of next layer weights
    # if reweight=True, we update next layer weights with weights that minimize change of input to next layer
    # if bias = True, we treat the bias as another channel (with no input weights) that can be pruned
    # epsilon: randomly sample n/k log(1/epsilon) channels to evaluate at each iteration of Greedy,
    # if epsilon=0 evaluate all remaining channels
    # TODO: obj fct is not monotone nor submodular when rwchange=False, add an algo for this case
    #  (both greedy and sieve will not work well)
    # TODO: remove backward option, it is not implemented
    def __init__(self, model, inputs=None, outputs=None, compressions=1, fullnet_comp=True, reweight=False, asymmetric=False, bias=False,
                 structure='neuron', prune_layers=None, sequential=False, onelayer_results_df=None, select="min", norm=0,
                 rwchange=False, algo="sieve", patches="all", epsilon=0, backward=False, save_kbounds=False):
        if asymmetric:
            sequential = True
        if save_kbounds:
            self.kbounds, self.kbounds_mod = {}, {}
        super().__init__(model, inputs, outputs, compressions=compressions, fullnet_comp=fullnet_comp, reweight=reweight,
                         asymmetric=asymmetric, bias=bias, structure=structure, prune_layers=prune_layers, sequential=sequential,
                         onelayer_results_df=onelayer_results_df, select=select, norm=norm, rwchange=rwchange, algo=algo,
                         patches=patches, epsilon=epsilon, backward=backward, save_kbounds=save_kbounds)
        if asymmetric:
            self.orig_acts_all = self.activations(only_input=True)


    def layer_pruned_channels(self, module_name, compressions):
        next_module = self.get_module(module_name, next=True)
        params = get_params(next_module)
        # no need to update if not sequential, the activations (of all prunable layers) will be computed in first call only
        acts = self.module_activations(next_module, only_input=True, update=self.sequential)
        orig_acts = self.orig_acts_all[next_module] if self.asymmetric else None

        if isinstance(next_module, nn.Conv2d):
            # acts shape changes from (nbatches, in_channels, H, W) --> (nbatches, npatches, in_channels * kernel_size)
            # after unfold, and transpose

            # extract only disjoint patches if patches = disjoint
            stride = next_module.kernel_size if self.patches == "disjoint" else next_module.stride
            acts = torch.nn.functional.unfold(torch.from_numpy(acts), next_module.kernel_size, next_module.dilation,
                                              next_module.padding, stride).transpose(1, 2).numpy()
            if self.patches == "random":
                n_patches = min(np.prod(next_module.kernel_size), acts.shape[1])
                sampled_acts = np.zeros((acts.shape[0], n_patches, acts.shape[-1]))
                sampled_patches = defaultdict(list)
                for idx in range(acts.shape[0]):
                    # from each input sample m patches
                    sampled_patches[idx] = random.sample(range(acts.shape[1]), n_patches)
                    sampled_acts[idx] = acts[idx][sampled_patches[idx], :]
                acts = sampled_acts

            print("n_patches = ", acts.shape[1])
            acts = acts.reshape(-1, acts.shape[-1])
            if self.asymmetric:
                orig_acts = torch.nn.functional.unfold(torch.from_numpy(orig_acts), next_module.kernel_size, next_module.dilation,
                                                  next_module.padding, stride).transpose(1, 2).numpy()
                if self.patches == "random":
                    sampled_orig_acts = np.zeros((orig_acts.shape[0], n_patches, orig_acts.shape[-1]))
                    for idx in range(orig_acts.shape[0]):
                        sampled_orig_acts[idx] = orig_acts[idx][sampled_patches[idx], :]
                    orig_acts = sampled_orig_acts
                orig_acts = orig_acts.reshape(-1, orig_acts.shape[-1])

        n_channels = self.module_n_channels(module_name, bias=self.bias)
        V = list(range(n_channels))
        # if module is linear, kernel_size = 1, if it's conv followed by conv, kernel_size = H x W
        kernel_size = next_module.kernel_size if isinstance(next_module, nn.Conv2d) else int(acts.shape[-1]/n_channels)
        map = lambda channels: map_chton(channels, kernel_size)

        # if self.algo == "greedy_old":
        #     update_neg_input_change = self.get_neg_input_change_fct(self.get_module(module_name, next=True),
        #                                                             rwchange=self.rwchange, update=True)
        #     kept_channels, new_params = greedy_algo(update_neg_input_change, int(fraction * n_channels), V,
        #                                      save_info_key="W_new" if self.rwchange else None)
        # else:
        neg_input_change = NegInputChange(acts, params, self.rwchange, self.asymmetric, orig_acts, self.bias, self.norm,
                                          map, self.backward)
        if self.save_kbounds:
            self.kbounds[module_name], self.kbounds_mod[module_name] = neg_input_change.get_kbound()
            print(f"In {module_name}, upper bound on k for approximate submodularity: {self.kbounds[module_name]}, "
              f"and for approximate modularity {self.kbounds_mod[module_name]}")
        pruned_channels, pruned_bias, new_params = {}, {}, {}
        for compression in compressions:
            k = int(self.perlayer_fractions[compression][module_name] * n_channels)
            card = n_channels - k if self.backward else k
            if self.algo == "sieve":
                opt = SieveStreamingint(k, neg_input_change, neg_input_change.fe_max(V), 0.1)
            elif self.algo == "greedy":
                # TODO: we can obtain solution of all fraction by just running greedy on the largest one and taking
                #  corresponding subsets, but to get corresponding new params we would need to either save them
                #  throughout execution for each k we need, as done in greedy_old or recompute them from scratch
                # ks = [int(self.perlayer_fractions[fraction][module_name] * n_channels) for fraction in fractions]
                opt = Greedyint(card, neg_input_change, int(n_channels*np.log(1/self.epsilon)/k) if self.epsilon > 0 and
                                                                                                 k > 0 else n_channels)
            else:
                raise NotImplementedError("Selected algorithm is not supported")
            opt.fit(V)
            # selected_channels = channels to prune if backward is true, and channels to keep otherwise
            selected_channels = opt.get_solution()
            F = opt.get_f()
            if len(selected_channels) < card:
                selected_channels = fillup_sol(selected_channels, F, V, card, update=True)
            new_params[compression] = F.get_new_params(selected_channels) if self.reweight else {}
            # during debugging: check value of obj, should be equal to 1 - ||B_S W_new - A W ||_F^2 / ||A W ||_F^2
            # W = params['weight']
            # W = W.reshape(W.shape[0], -1).T
            # W_new = new_params[compression]['weight']
            # W_new = W_new.reshape(W_new.shape[0], -1).T
            # A = orig_acts if self.asymmetric else acts
            # B = acts
            # B_S = B.copy()
            # B_S[:, map(list(set(V).difference(set(kept_channels))))] = 0
            # 1 - np.linalg.norm(B_S @ W_new - A @ W)**2 / np.linalg.norm(A @ W)**2
            # opt.get_fval()
            # F(kept_channels)

            pruned_channels[compression] = selected_channels if self.backward else list(set(V) - set(selected_channels))
            pruned_bias[compression] = False
            i_bias = n_channels - self.bias
            if i_bias in pruned_channels[compression]:  # check if "bias channel" is pruned
                pruned_channels[compression].remove(i_bias)
                pruned_bias[compression] = True
        return pruned_channels, pruned_bias, new_params


class LayerGreedyFSChannel(LayerStructuredPruning):
    # implements the layerwise Greedy forward selection algorithm in
    #  Ye, et al. Good Subnetworks Provably Exists: Pruning via Greedy Forward Selection. ICML 2020
    # epsilon: randomly sample n/k log(1/epsilon) channels to evaluate at each iteration of Greedy,
    # if epsilon=0 evaluate all remaining channels
    # If reweight=True, we use our reweighting procedure on their selection of channels to get next layer new weights
    # otherwise, we use their own new weights

    def __init__(self, model, inputs=None, outputs=None, compressions=1, fullnet_comp=True, reweight=False, bias=False,
                 structure='neuron', prune_layers=None, onelayer_results_df=None, select="min",
                 loss_func=nn.CrossEntropyLoss(), epsilon=0, fw=True, scale_masks=True, full_data=False, train_dl=None):
        self.inputs_list, self.outputs_list = inputs, outputs
        if not fw:
            self.scale = 1 #TODO: remove scale_masks param if we go with new code
        super().__init__(model, inputs, outputs, compressions=compressions, fullnet_comp=fullnet_comp, reweight=reweight,
                         bias=bias, structure=structure, prune_layers=prune_layers, sequential=True,
                         onelayer_results_df=onelayer_results_df, select=select, loss_func=loss_func, epsilon=epsilon,
                         fw=fw, scale_masks=scale_masks, full_data=full_data, train_dl=train_dl)

    def layer_pruned_channels(self, module_name, compressions):
        n_channels = self.module_n_channels(module_name)
        # save mask from previous layer's pruning if any
        module = self.get_module(module_name)
        weight_mask = module.weight_mask if hasattr(module, 'weight_mask') else None
        self.model.to(self.device)
        if self.fw: # Frank-Wolfe style greedy (allows repeated elts)
            next_weight_orig = self.get_module(module_name, next=True).weight.data.numpy()
            next_shape = next_weight_orig.shape
            for compression in compressions:
                # TODO: get solution of all compressions from smallest one
                self.pruned_bias[compression][module_name] = False
                self.new_params[compression][module_name] = {}
                self.pruned_channels[compression][module_name] = list(range(n_channels))
                k = int(self.perlayer_fractions[compression][module_name] * n_channels)
                channels_count = np.zeros(n_channels) # keep track of # of time each channel is added
                # nchannels_rem = n_channels
                nchannels_eval = int(n_channels*np.log(1/self.epsilon)/k) if self.epsilon > 0 and k > 0 else n_channels
                nchannels_kept = 0
                niter = 0
                while nchannels_kept<k:
                    niter += 1
                    if self.full_data:
                        x, y = next(iter(self.train_dl))
                    else:
                        # get a mini-batch of data among ones given as inputs
                        idx = random.sample(range(len(self.inputs_list)), 1)[0]
                        x, y = self.inputs_list[idx], self.outputs_list[idx]
                    x, y = x.to(self.device), y.to(self.device)
                    best_loss = float('inf')
                    # np.random.shuffle(self.pruned_channels[compression][module_name])
                    # eval_channels = self.pruned_channels[compression][module_name][:nchannels_eval] #min(nchannels_eval, nchannels_rem)
                    # nchannels_rem -= 1
                    eval_channels = random.sample(range(n_channels),nchannels_eval)
                    for channel in eval_channels:
                        # prune and reweight after adding channel
                        # during debug: save weight=module.weight.data.numpy().copy(), next_weight=self.get_module(module_name, next=True).weight.data.numpy().copy()
                        if channels_count[channel] == 0: # channel not selected yet
                            self.pruned_channels[compression][module_name].remove(channel)
                        channels_count[channel] += 1
                        self.new_params[compression][module_name]['weight'] = (next_weight_orig.reshape(next_shape[0], n_channels, -1) \
                                                                              * np.expand_dims(channels_count*(n_channels/niter), [0,2])).reshape(next_shape)

                        self.reweight_layer_params(module_name, compression)
                        _ = self.layer_masks(module_name, compression)
                        # during debug: check np.linalg.norm(self.get_module(module_name, next=True).weight.data.numpy() - self.new_params[compression][module_name]['weight'])
                        # and check niter== sum(channels_count)
                        # eval loss
                        with torch.no_grad():
                            yhat = self.model(x)
                            loss = self.loss_func(yhat, y)
                        if loss < best_loss:
                            best_loss = loss
                            best_channel = channel
                        # undo evaluation pruning without undoing previous layer pruning if any
                        self.undo_layer_masks(module_name, weight_mask)
                        self.new_params[compression][module_name]['weight'] = next_weight_orig
                        self.reweight_layer_params(module_name, compression)
                        channels_count[channel] -= 1
                        if channels_count[channel] == 0:
                            self.pruned_channels[compression][module_name].append(channel)
                        # during debug: check if (weight==module.weight.data.numpy()).all(), (next_weight==self.get_module(module_name, next=True).weight.data.numpy()).all()
                    # update pruned channels and new params after adding best_channel
                    # no need to actually apply the pruning & rw
                    if channels_count[best_channel] == 0: # adding new channel
                        nchannels_kept += 1
                        self.pruned_channels[compression][module_name].remove(best_channel)
                    channels_count[best_channel] += 1

                print("niter: ", niter, "nchannels_kept", nchannels_kept)
                self.new_params[compression][module_name] = {'weight': (next_weight_orig.reshape(next_shape[0], n_channels, -1)
                * np.expand_dims(channels_count*(n_channels/max(niter, 1)), [0,2])).reshape(next_shape) } if not self.reweight else {}
            pruned_channels = {compression: self.pruned_channels[compression][module_name] for compression in compressions}
            new_params = {compression: self.new_params[compression][module_name] for compression in compressions}
            pruned_bias = {compression: self.pruned_bias[compression][module_name] for compression in compressions}
            return pruned_channels, pruned_bias, new_params
        else:
        # greedy without allowing repeated elts + with scaling |V|/|S| kept during finetuning too!
        # this is the version implemented in their code:  https://github.com/lushleaf/Network-Pruning-Greedy-Forward-Selection
            for compression in compressions:
                # TODO: get solution of all compressions from smallest one
                self.pruned_bias[compression][module_name] = False
                self.new_params[compression][module_name] = None
                self.pruned_channels[compression][module_name] = list(range(n_channels))
                k = int(self.perlayer_fractions[compression][module_name] * n_channels)
                selected_channels = []
                nchannels_rem = n_channels
                nchannels_eval = int(n_channels*np.log(1/self.epsilon)/k) if self.epsilon > 0 and k > 0 else n_channels
                for _ in range(k):
                    if self.full_data:
                        x, y = next(iter(self.train_dl))
                    else:
                        # get a mini-batch of data among ones given as inputs
                        idx = random.sample(range(len(self.inputs_list)), 1)[0]
                        x, y = self.inputs_list[idx], self.outputs_list[idx]
                    x, y = x.to(self.device), y.to(self.device)
                    best_loss = float('inf')
                    np.random.shuffle(self.pruned_channels[compression][module_name])
                    eval_channels = self.pruned_channels[compression][module_name][:min(nchannels_eval, nchannels_rem)]
                    nchannels_rem -= 1
                    self.scale = n_channels/(n_channels - nchannels_rem) if self.scale_masks else 1
                    for channel in eval_channels:
                        self.pruned_channels[compression][module_name].remove(channel)
                        _ = self.layer_masks(module_name, compression, scale=self.scale)
                        with torch.no_grad():
                            yhat = self.model(x)
                            loss = self.loss_func(yhat, y)
                        if loss < best_loss:
                            best_loss = loss
                            best_channel = channel
                        self.pruned_channels[compression][module_name] += [channel]
                        # undo evaluation pruning without undoing previous layer pruning if any
                        self.undo_layer_masks(module_name, weight_mask)
                    selected_channels += [best_channel]
                    self.pruned_channels[compression][module_name].remove(best_channel)
            return {compression: self.pruned_channels[compression][module_name] for compression in compressions}, None, None


class LayerSamplingChannel(LayerStructuredPruning):
    # implement the layerwise sampling-based pruning method proposed in
    # Liebenwein, L., Baykal, C., Lang, H., Feldman, D., and
    # Rus, D. Provable filter pruning for efficient neural networks. ICLR 2020.
    # We adapt their code from https://github.com/lucaslie/torchprune to work with our framework
    # delta is the probability of failure
    # If reweight=True, we use our reweighting procedure on their selection of channels to get next layer new weights
    # otherwise, we use their own new weights.

    def __init__(self, model, inputs=None, outputs=None, compressions=1, fullnet_comp=True, reweight=False, bias=False,
                 structure='neuron', prune_layers=None, delta=1e-12):
        self._probability = {}
        self._probability_div = {}
        self.sensitivity_in = {}  # they define this property for the tracker of each layer, so we need a dictionary
        self.num_patches = {}
        self.perlayer_nchannels_dict = {}
        self._coeffs = {}
        self._sum_sens = {}
        self.perlayer_fractions = {}
        self.c_constant = 3
        # self.delta = 0
        super().__init__(model, inputs, outputs, compressions=compressions, fullnet_comp=fullnet_comp, reweight=reweight,
                         bias=bias, structure=structure, prune_layers=prune_layers, sequential=False, delta=delta)

    def preprocessing(self):
        # # we can infer delta from the size of S we're using, but this leads to too small delta (~1e-64)
        # so decided to use instead same delta values they use even if size of S is different
        # etas = {name: np.prod(self.module_activations(self.get_module(name, next=True), only_input=True, update=False)
        #                       [0].shape) for name in self.channel_prunable}  # n_channels x n_patches
        # eta = sum(etas.values())
        # eta_star = max(etas.values())
        # they have size_s = math.ceil(self.c_constant * math.log(8.0 * eta_star * eta/ self.delta)), so we deduce that
        # self.delta = min(8 * eta_star * eta / math.exp(len(self.inputs)/self.c_constant), 1e-16)
        self.perlayer_nchannels_dict = {name: torch.tensor(val) for name, val in self.perlayer_nchannels_dict.items()}
        for name in self.channel_prunable:
            # self.perlayer_nchannels[name] = torch.tensor(self.module_n_channels(name))
            self._compute_sensitivity(name)
            coeffs, sum_sens, probs, probs_div = self._get_coefficients(name)
            self._probability[name] = probs
            self._probability_div[name] = probs_div
            self._coeffs[name] = coeffs
            self._sum_sens[name] = sum_sens

    def _adapt_activations(self, a_original):
        bigger = a_original >= 0.0
        if torch.all(bigger):
            return a_original

        # change activations matrix to be only positive if necessary
        a_pos = torch.zeros_like(a_original)
        a_neg = torch.zeros_like(a_original)
        a_pos[bigger] = a_original[bigger]
        a_neg[~bigger] = a_original[~bigger]

        return torch.cat((a_pos, -a_neg))

    def _remove_bias(self, module):
        # remove bias since we don't need it for our computations
        if module.bias is not None:
            bias_original = module.bias
            bias_zeros = torch.zeros(bias_original.shape).to(
                bias_original.device
            )
            module.state_dict()["bias"].copy_(bias_zeros)

    def _reshape_z(self, module, z_values):
        # flatten all batch dimensions first if it's linear...
        if isinstance(module, nn.Linear):
            z_values = flatten_all_but_last(z_values)
        z_values = z_values.view(z_values.shape[0], z_values.shape[1], -1)
        return z_values

    def _process_denominator(self, z_values):
        # processing
        eps = torch.Tensor([np.finfo(np.float32).eps]).to(z_values.device)
        mask = torch.le(torch.abs(z_values), eps)
        z_values.masked_fill_(mask, np.Inf)
        return z_values

    # TODO: move this to utilities and use it in our method too?
    def unfold(self, module, x):
        """Unfold depending on type of layer.

        After unfolding we want shape [batch_size, feature_size, patch_size]
        """
        if isinstance(module, nn.Conv2d):
            return nn.functional.unfold(x, kernel_size=module.kernel_size, stride=module.stride, padding=module.padding,
                                        dilation=module.dilation)
        else:
            # flatten all batch dimensions, then unsqueeze last dim
            return flatten_all_but_last(x).unsqueeze(-1)

    def _get_g_sens_f(self, weight_f, activations, z_values_f):
        # compute g
        g_sens_f = weight_f.unsqueeze(0).unsqueeze(-1) * activations
        g_sens_f /= z_values_f.unsqueeze(1)

        return g_sens_f.clone().detach()

    def _reduction(self, g_sens_f, dim):
        return torch.max(g_sens_f, dim=dim)[0]

    def _compute_sensitivity(self, mod_name):
        # they are pruning input channels of mod while we prune output channels, so we apply their method to next mod
        # when mod is a conv followed by linear this does make a difference (pruning channels vs neurons)
        next_module = self.get_module(mod_name, next=True)
        n_channels = self.perlayer_nchannels_dict[mod_name]
        # activations (of all prunable layers) will be computed in first call only
        # no need to update since not sequential
        acts = torch.from_numpy(self.module_activations(next_module, only_input=True, update=False))
        weight = next_module.weight.data
        self.sensitivity_in[mod_name] = torch.zeros(n_channels)

        # remove weights we are not interested in and have two modules ...
        idx_plus = weight > 0.0
        idx_minus = weight < 0.0
        weight_plus = torch.zeros_like(weight)
        weight_minus = torch.zeros_like(weight)
        weight_plus[idx_plus] = weight[idx_plus]
        weight_minus[idx_minus] = weight[idx_minus]

        # save a deepcopy of module with WPlus ...
        module_plus = copy.deepcopy(next_module)
        module_plus.weight.data = weight_plus # .view(next_module.weight.data.shape)
        self._remove_bias(module_plus)

        # save a deepcopy of module with WPlus ...
        module_minus = copy.deepcopy(next_module)
        module_minus.weight.data = weight_minus # .view(next_module.weight.data.shape)
        self._remove_bias(module_minus)

        # Wunfold.shape = (outFeature, filterSize)
        # where filterSize = inFeatures*kappa1*kappa2 for conv2d
        # or filterSize = inNeurons for linear
        w_unfold_plus = weight_plus.view((weight_minus.shape[0], -1))
        w_unfold_minus = weight_minus.view((weight_minus.shape[0], -1))

        for act in acts:
            # from _hook of base_sens_tracker
            a_adapted = self._adapt_activations(act.unsqueeze(0))

            z_plus = module_plus(a_adapted)
            z_minus = module_minus(a_adapted)
            z_plus = self._reshape_z(next_module, z_plus)
            z_minus = self._reshape_z(next_module, z_minus)
            z_plus = self._process_denominator(z_plus)
            z_minus = self._process_denominator(z_minus)

            # shape = (batchSize, filterSize, outExamples) as above
            a_unfolded = self.unfold(next_module, a_adapted)

            self.num_patches[mod_name] = a_unfolded.shape[-1]

            # preallocate g
            batch_size = a_unfolded.shape[0]
            g_sens_in = torch.zeros((batch_size,) + self.sensitivity_in[mod_name] .shape).to(self.device)

            # populate g for this batch
            for idx_f in range(w_unfold_plus.shape[0]):
                # compute g
                g_sens_f = torch.max(self._get_g_sens_f(w_unfold_plus[idx_f], a_unfolded, z_plus[:, idx_f]),
                                     self._get_g_sens_f(w_unfold_minus[idx_f], a_unfolded, z_minus[:, idx_f]))
                # Reduction over outExamples
                g_sens_f = self._reduction(g_sens_f, dim=-1)

                # Reduction over outputChannels, M: isn't this taking max over kernelsize
                g_sens_in_f = self._reduction(g_sens_f.view((g_sens_f.shape[0], n_channels, -1)), dim=-1)

                # store results
                g_sens_in = torch.max(g_sens_in, g_sens_in_f)

            # Max over this batch
            sens_in_batch = torch.max(g_sens_in, dim=0)[0]

            # update the sensitivity with the new g values.
            self.sensitivity_in[mod_name].copy_(torch.max(self.sensitivity_in[mod_name], sens_in_batch))

    def _get_sens_stats(self, mod_name):
        # short-hand
        sens_in = self.sensitivity_in[mod_name]
        sum_sens = sens_in.sum().view(-1)
        probs = sens_in / sum_sens
        # Also save a version with 0 mapped to Inf so division works
        eps = torch.Tensor([np.finfo(np.float32).eps]).to(self.device)
        probs_div = copy.deepcopy(probs)
        probs_div.masked_fill_(probs_div <= eps, np.Inf)
        return sum_sens, probs, probs_div

    def _get_coefficients(self, mod_name):
        """Get the coefficients according to our theorems."""
        weight = self.get_module(mod_name, next=True).weight.data
        num_patches = self.num_patches[mod_name]

        # a few stats from sensitivity
        sum_sens, probs, probs_div = self._get_sens_stats(mod_name)

        # cool stuff
        k_size = weight[0, 0].numel()
        log_numerator = torch.tensor(8.0 * (num_patches + 1) * k_size).to(self.device)
        log_term = self.c_constant * torch.log(log_numerator / self.delta)
        alpha = 2.0 * log_term

        # compute coefficients
        coeffs = copy.deepcopy(sum_sens)
        coeffs *= alpha
        return coeffs, sum_sens, probs, probs_div

    def _get_sample_complexity(self, eps):
        k_constant = 3.0
        m_budget = {name: torch.max((k_constant * (6 + 2 * eps) * coeff / (eps ** 2)).ceil(), torch.tensor(1))
                    for name, coeff in self._coeffs.items()}
        return m_budget

    def _get_unique_samples(self, m_budget):
        for name in self._probability:
            # # skip if inf budget to enforce m_budget = n_channels with eps_max even when some channels have zero prob
            # if not torch.isinf(m_budget[name]):
            #     m_budget[name] = expected_unique(self._probability[name], m_budget[name]).to(dtype=torch.long)
            # else:
            #     m_budget[name] = torch.tensor(len(self._probability[name]))

            # Reverse calibration
            m_budget[name] = expected_unique(self._probability[name], m_budget[name]).to(dtype=torch.long)
        return m_budget

    def _get_proposed_num_features(self, arg, min_budget=0):
        # get budget according to sample complexity
        eps = arg
        m_budget = self._get_sample_complexity(eps)
        m_budget = self._get_unique_samples(m_budget) #.to(dtype=torch.long)
        perlayer_budgets = {name: torch.max(torch.min(m_budget[name], val), torch.tensor(min_budget)) for name, val in
                           self.perlayer_nchannels_dict.items()}
        return perlayer_budgets

    def _get_resulting_size(self, arg):
        """Get resulting size for some arg."""
        perlayer_budgets = self._get_proposed_num_features(arg)
        return torch.tensor(self.prunable_pruned_size([perlayer_budgets[name] for name in self.channel_prunable]) + self.nonprunable_size)

    # def _get_resulting_nchannels(self, arg, min_budget):
    #     """Get resulting size for some arg."""
    #     perlayer_budget = self._get_proposed_num_features(arg, min_budget)
    #     return sum(perlayer_budget.values())

    def select_perlayer_budgets(self, keep_ratio):
        # nchannels = sum(self.perlayer_nchannels.values())
        # nchannels_tokeep = int(fraction * nchannels)
        arg_min = 1e-300
        arg_max = 1e150
        budget = int(self.size_orig*keep_ratio) - self.nonprunable_size if self.fullnet_comp else \
            int(self.prunable_size*keep_ratio)
        # min_budget = int(nchannels_tokeep >= len(self.perlayer_nchannels))
        def f_opt(arg):
            size_resulting = self._get_resulting_size(arg)
            # print("querrying: ", arg, " budget difference: ", budget - size_resulting)
            return budget - size_resulting

        # def f_opt(arg):
        #     nchannels_kept = self._get_resulting_nchannels(arg, min_budget)
        #     return nchannels_tokeep - nchannels_kept

        # solve with bisection method and get resulting feature allocation
        f_value_min = f_opt(arg_min)
        f_value_max = f_opt(arg_max)
        if f_value_min.sign() == f_value_max.sign():
        # this can happen when some probabilities are zero, so even with argmin perlayer_budgets != pelayer_nchannels
        # so we can endup with resulting size < budget for large budgets
            arg_opt, f_value_opt = (
                (arg_min, f_value_min)
                if abs(f_value_min) < abs(f_value_max)
                else (arg_max, f_value_max)
            )
            error_msg = (
                "no bisection possible"
                f"; argMin: {arg_min}, minF: {f_value_min}"
                f"; argMax: {arg_max}, maxF: {f_value_max}"
            )
            print(error_msg)
            # if abs(f_value_opt) / nchannels_tokeep > 0.005:
            #     raise ValueError(error_msg)
        else:
            arg_opt = optimize.brentq(f_opt, arg_min, arg_max, maxiter=1000, xtol=10e-250, disp=False)
        perlayer_budgets = self._get_proposed_num_features(arg_opt)
        prunable_pruned_size = self.prunable_pruned_size([perlayer_budgets[name] for name in self.channel_prunable])
        print("difference in budget = ", budget - prunable_pruned_size - self.nonprunable_size)
        # nchannels_kept = sum(perlayer_budgets.values())
        # print("nchannels_kept", nchannels_kept, "nchannels_tokeep", nchannels_tokeep)
        #
        # # if there's any remaining total budget (this happens for example where there are zero probabilities)
        # # use our heuristic of filling up layers gradually from one with largest fraction to smallest
        # # separate these "extra budgets" from original ones, we won't apply adapt_sample_size to those
        # perlayer_extra_budgets = {name: 0 for name in self.channel_prunable}
        # perlayer_fractions = {name: perlayer_budgets[name]/self.perlayer_nchannels[name] for name in self.channel_prunable}
        # for name in sorted(perlayer_fractions, key=perlayer_fractions.get, reverse=True):
        #     if nchannels_kept < nchannels_tokeep:
        #         perlayer_fractions[name] = np.minimum((int(perlayer_fractions[name]*self.perlayer_nchannels[name]) +
        #                                                 (nchannels_tokeep - nchannels_kept))/self.perlayer_nchannels[name], 1)
        #     else:
        #         break
        #     perlayer_extra_budgets = {name: (perlayer_fractions[name]*self.perlayer_nchannels[name]).long() -
        #                                     perlayer_budgets[name] for name in self.channel_prunable}
        #     nchannels_kept = sum(perlayer_budgets.values()) + sum(perlayer_extra_budgets.values())
        # print("nchannels_kept after filling heuristic", nchannels_kept, "nchannels_tokeep", nchannels_tokeep)
        #
        # TODO: decrease budgets if total nchannels kept is more than allowed total budget (should be unlikely to happen)
        return perlayer_budgets #, perlayer_extra_budgets

    def layer_pruned_channels(self, module_name, compressions):
        pass

    def model_pruned_channels(self):
        """
          Since we might not get the keep_ratio right, we will try to repeat the
          compression until we get it approximately right. This is cheap since
          the expensive part of pruning is the initialization where we compute
          the sensitivities but we can re-use that.
        """
        pruned_channels, new_params, perlayer_budgets = {}, {}, {}
        # n_channels = sum(self.perlayer_nchannels.values())
        # wrapper for root finding and look-up to speed it up.
        f_opt_lookup = {}
        for compression in self.compressions:
            keep_ratio = 1/compression
            # boundaries for binary search over potential keep_ratios
            kr_min = 0.4 * keep_ratio
            kr_max = 0.999
            # adjust fraction
            # fract_target = int(fraction * n_channels)/n_channels

            def _f_opt(kr_compress):
                # check for look-up
                if kr_compress in f_opt_lookup:
                    return f_opt_lookup[kr_compress]
                # print("Try kr = ", kr_compress)
                # compress
                perlayer_budgets, pruned_channels, new_params = self._compress_once(kr_compress)

                # check resulting keep ratio
                prunable_pruned_size = self.prunable_pruned_size([self.perlayer_nchannels_dict[name]
                                                       - len(pruned_channels[name]) for name in self.channel_prunable])
                kr_actual = (prunable_pruned_size + self.nonprunable_size)/self.size_orig if self.fullnet_comp else \
                    prunable_pruned_size/self.prunable_size
                kr_diff = kr_actual - keep_ratio
                print(f"Current diff in keep ratio is: {kr_diff * 100.0:.2f}%")

                # set to zero if we are already close and stop (modifying this to apply only if we're below keep_ratio)
                if abs(kr_diff) < 0.005 * keep_ratio and kr_diff<0:
                    kr_diff = 0.0

                # store look-up
                f_opt_lookup[kr_compress] = (kr_diff, perlayer_budgets, pruned_channels, new_params)

                return f_opt_lookup[kr_compress]

            # some times the keep ratio is pretty accurate
            # so let's try with the correct keep ratio first
            try:
                # we can either run right away or update the boundaries for the
                # binary search to make it faster.
                kr_diff_nominal, perlayer_budgets[compression], pruned_channels[compression], new_params[compression] = \
                    _f_opt(keep_ratio)
                if kr_diff_nominal == 0.0:
                    continue  # go to next compression
                elif kr_diff_nominal > 0.0:
                    kr_max = keep_ratio
                else:
                    kr_min = keep_ratio

            except (ValueError, RuntimeError):
                pass

            # run the root search
            # if it fails we simply pick the best value from the look-up table
            try:
                kr_opt = optimize.brentq(lambda kr: _f_opt(kr)[0], kr_min, kr_max, maxiter=20, xtol=1e-9, disp=True)
                                         # xtol=5e-3, rtol=5e-3
                if _f_opt(kr_opt)[0]>0:
                    raise ValueError # for fair comparison with other methods, enforce kr_opt to be less than keep_ratio
            except (ValueError, RuntimeError):
                kr_diff_opt = float("inf")
                kr_opt = None
                for kr_compress, lookup_val in f_opt_lookup.items():
                    kr_diff = lookup_val[0]
                    if abs(kr_diff) < abs(kr_diff_opt) and kr_diff<0:
                        kr_diff_opt = kr_diff
                        kr_opt = kr_compress
                print(
                    "Cannot approximate keep ratio. "
                    f"Picking best available keep ratio {kr_opt * 100.0:.2f}% "
                    f"with actual diff {kr_diff_opt * 100.0:.2f}%."
                )

            # now run the compression one final time
            _, perlayer_budgets[compression], pruned_channels[compression], new_params[compression] = _f_opt(kr_opt)

        # store perlayer fractions
        self.perlayer_fractions = {compression: {name: perlayer_budgets[compression][name]/self.perlayer_nchannels_dict[name] for
                                              name in self.channel_prunable} for compression in self.compressions}
        pruned_bias = {compression: {name: False for name in self.channel_prunable} for compression in self.compressions}
        return pruned_channels, pruned_bias, new_params

    def _compress_once(self, keep_ratio):
        # if len(self.channel_prunable) > 1:
        t = time.perf_counter()
        perlayer_budgets = self.select_perlayer_budgets(keep_ratio)
        print("perlayer fractions selection time:", time.perf_counter() - t, "secs.")
        # else:
        #     perlayer_budgets = {name: fraction for name in self.channel_prunable}

        pruned_channels = {}
        new_params = {}
        # loop through the layers in reverse to compress (not sure why that makes any difference..)
        for module_name in reversed(self.channel_prunable):
            W = self.get_module(module_name, next=True).weight.data
            k = perlayer_budgets[module_name] #+ perlayer_extra_budgets[module_name]
            # this is from RandFilterPruner.prune
            if k > 0:
                num_to_sample = adapt_sample_size(self._probability[module_name].view(-1), perlayer_budgets[module_name], False)
                num_to_sample = int(num_to_sample)
                # num of times each channel is sampled
                num_samples = Multinomial(num_to_sample, self._probability[module_name]).sample().int()

                # # for fair comparison with other methods I added these lines to ensure exactly k channels are kept
                # # this will make the root search in model_pruned_channels unnecessary, but I'll keep it just in case
                # selected_channels = (num_samples > 0)
                # nchannels_diff = selected_channels.sum() - k
                # print(f"nchannels_diff={nchannels_diff} in {module_name}")
                # if nchannels_diff < 0:
                #     # add k - nchannels_kept unselected channels with highest probability
                #     #probs = (self._probability[module_name] * (~selected_channels)).view(-1).cpu().numpy() # set selected channels probability to zero
                #     probs = copy.deepcopy((self._probability[module_name]).view(-1).cpu().numpy())
                #     probs[selected_channels] = -0.1  # set selected channels probability to -0.1 to avoid being 'added'
                #     idx_top = np.argpartition(probs, nchannels_diff)[nchannels_diff:]
                #     num_samples[idx_top] = 1
                # elif nchannels_diff > 0:
                #     # remove nchannels_kept-k selected channels with lowest probability
                #     probs = copy.deepcopy((self._probability[module_name]).view(-1).cpu().numpy())
                #     probs[~selected_channels] = 1.1  # set non-selected channels probability to 1.1 to avoid being 'removed'
                #     idx_bottom = np.argpartition(probs, nchannels_diff)[:nchannels_diff]
                #     num_samples[idx_bottom] = 0
                # print(f"nchannels_diff={(num_samples > 0).sum() -k} in {module_name} after adjustment")
            else:
                num_samples = torch.zeros(len(self._probability[module_name])).int()  # .to(size_pruned.device)

            pruned_channels[module_name] = (num_samples == 0).nonzero().flatten().tolist()

            if not self.reweight:
                # pre-allocate gammas
                gammas = copy.deepcopy(num_samples).float()

                # this is from FilterSparsifier.sparsify
                if num_samples.sum() > 0:  # self._do_reweighing is True in pfp sparsifier
                    gammas = gammas / num_samples.sum().float() / self._probability_div[module_name]
                else:
                    gammas = (gammas > 0).float()

                # make gammas compatible with Woriginal # self._out_mode is false in pfp
                gammas = gammas.unsqueeze(0).unsqueeze(-1)

                new_params[module_name] = {'weight': (gammas * W.view(W.shape[0], self.perlayer_nchannels_dict[module_name],
                                                                      -1)).view_as(W).numpy()}
            else:
                new_params[module_name] = {}

        # perlayer_budgets = {name: perlayer_budgets[name] + perlayer_extra_budgets[name] for name in self.channel_prunable}
        return perlayer_budgets, pruned_channels, new_params
