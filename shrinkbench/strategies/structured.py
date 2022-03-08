"""Structured pruning

Implements structured pruning strategies
"""

from ..pruning import *
from .structured_utils import *
import numpy as np
import torch
import torch.nn as nn
from PySSM import Greedyint, Greedypair, SieveStreamingint, SieveStreamingpair


class WeightNormChannel(StructuredPruning):
    # prunes fraction of channels with the lowest input/output weight norm, with ord = order of the norm
    # normalized by # of input/output weights if norm = True
    # when ord = 1, this computes the onorm (if onorm=True) and inorm (if onorm=False) functions proposed in
    # "T. He, Y. Fan, Y. Qian, T. Tan, and K. Yu. Reshaping deep neural network for fast decoding by node-pruning,
    # ICASSP, 2014."

    def __init__(self, model,  inputs=None, outputs=None, fractions=1, reweight=False, bias=False, structure='neuron',
                 prune_layers=None, onorm=True, ord=1, norm=True):
        super().__init__(model, inputs, outputs, fractions=fractions, reweight=reweight, bias=bias, structure=structure,
                         prune_layers=prune_layers, onorm=onorm, ord=ord, norm=norm)

    def model_pruned_channels(self):
        axis = 1 if self.onorm else 0
        weight_norm = lambda w: np.linalg.norm(w, ord=self.ord, axis=1) / (w.shape[1] if self.norm else 1)
        importances = map_channel_importances(lambda param: weight_norm(np.swapaxes(param['weight'], 0, axis).
                                             reshape(param['weight'].shape[axis], -1)), self.params(next=self.onorm))
        V = self.channels_ind()
        order = sorted(dict_to_list(V), key=lambda e: importances[e[0]][e[1]])  # e has form (name, index in layer)
        ks = {fraction: int(len(order) * fraction) for fraction in self.fractions}
        pruned_channels = {fraction: list_to_dict(order[:-ks[fraction]] if ks[fraction] != 0 else order)
                           for fraction in self.fractions}
        return pruned_channels, None, None


class LayerWeightNormChannel(LayerStructuredPruning):
    # prunes fraction of channels with the lowest output (if onorm=True) or input (if onorm=False) weight norm,
    # in each prunable layer independently, with ord = order of the norm
    # when ord = 1, this computes the onorm (if onorm=True) and inorm (if onorm=False) functions proposed in
    # "T. He, Y. Fan, Y. Qian, T. Tan, and K. Yu. Reshaping deep neural network for fast decoding by node-pruning,
    # ICASSP, 2014."

    def __init__(self, model, inputs=None, outputs=None, fractions=1, reweight=False, bias=False, structure='neuron',
                 prune_layers=None, onelayer_results_df=None, select="min", onorm=True, ord=1):
        super().__init__(model, inputs, outputs, fractions=fractions, reweight=reweight, bias=bias, structure=structure,
                         prune_layers=prune_layers, onelayer_results_df=onelayer_results_df, select=select, onorm=onorm,
                         ord=ord)

    def layer_pruned_channels(self, module_name, fractions):
        params = self.module_params(module_name, next=self.onorm)
        if self.onorm:
            W = np.swapaxes(params['weight'], 0, 1)
        W = W.reshape(W.shape[0], -1)  # needed if structure = channel
        importances = np.linalg.norm(W, ord=self.ord, axis=1)
        order = np.argsort(importances)  # increasing order
        ks = {fraction: int(len(order) * self.perlayer_fractions[fraction][module_name]) for fraction in fractions}
        pruned_channels = {fraction: order[:-ks[fraction]] if ks[fraction] != 0 else order for fraction in fractions}
        return pruned_channels, None, None


class ActGradChannel(StructuredPruning, ActivationMixin, GradientMixin):
    # prunes fraction of channels with the lowest abs(activation * gradient), averaged over nbatches of inputs/outputs,
    # normalized by the l2-norm of these importance scores if norm=True

    def __init__(self, model, inputs=None, outputs=None, fractions=1, reweight=False, bias=False, structure='neuron',
                 prune_layers=None, norm=True):
        super().__init__(model, inputs, outputs, fractions=fractions, reweight=reweight, bias=bias, structure=structure,
                         prune_layers=prune_layers, norm=norm)

    def model_pruned_channels(self):
        grads = self.gradients(only_input=True, update=True, only_prunable=True)
        acts = self.activations(only_input=True, update=True, only_prunable=True)

        # inner sum needed for structure = channel
        act_grad_imp = lambda mod: np.abs(np.sum(acts[mod] * grads[mod], tuple(range(2, acts[mod].ndim)))).sum(0) / \
                                   acts[mod].shape[0]
        importances = {name: act_grad_imp(self.get_module(name, next=True)) for name in self.channel_prunable}
        if self.norm:
            importances = {name: imp/np.linalg.norm(imp) for name, imp in importances.items()}
        V = self.channels_ind()
        order = sorted(dict_to_list(V), key=lambda e: importances[e[0]][e[1]])  # e has form (name, index in layer)
        ks = {fraction: int(len(order) * fraction) for fraction in self.fractions}
        pruned_channels = {fraction: list_to_dict(order[:-ks[fraction]] if ks[fraction] != 0 else order)
                           for fraction in self.fractions}
        return pruned_channels, None, None


class LayerActGradChannel(LayerStructuredPruning, ActivationMixin, GradientMixin):
    # prunes fraction of channels with the lowest abs(activation * gradient), averaged over nbatches of inputs/outputs,
    # in each prunable layer independently
    def __init__(self, model, inputs=None, outputs=None, fractions=1, reweight=False, bias=False, structure='neuron',
                 prune_layers=None, onelayer_results_df=None, select="min"):
        super().__init__(model, inputs, outputs, fractions=fractions, reweight=reweight, bias=bias, structure=structure,
                         prune_layers=prune_layers, onelayer_results_df=onelayer_results_df, select=select)

    def layer_pruned_channels(self, module_name, fractions):
        next_module = self.get_module(module_name, next=True)
        grads = self.module_gradients(next_module, only_input=True, update=True)
        acts = self.module_activations(next_module, only_input=True, update=True)
        # inner sum needed for structure = channel
        importances = np.abs(np.sum(acts * grads, tuple(range(2, acts.ndim)))).sum(0) / acts.shape[0]
        order = np.argsort(importances)  # increasing order
        ks = {fraction: int(len(order) * self.perlayer_fractions[fraction][module_name]) for fraction in fractions}
        pruned_channels = {fraction: order[:-ks[fraction]] if ks[fraction] != 0 else order for fraction in fractions}
        return pruned_channels, None, None


class RandomChannel(StructuredPruning):
    # randomly prunes fraction of channels in all prunable layers
    def model_pruned_channels(self):
        # assign random importances
        importances = map_channel_importances(lambda param: np.random.uniform(0, 1, size=param['weight'].shape[0]),
                                              self.params())
        flat_importances = flatten_channel_importances(importances)
        importances_sorted = np.sort(flat_importances)
        thresholds = [importances_sorted[-int(len(importances_sorted) * fraction) - 1] for fraction in self.fractions]
        pruned_channels = {fraction: map_channel_importances(lambda param: prune_threshold_channels(param, thresholds[i]),
                                          importances) for i, fraction in enumerate(self.fractions)}
        return pruned_channels, None, None


class LayerRandomChannel(LayerStructuredPruning):
    # randomly prunes fraction of channels in each prunable layer independently
    def layer_pruned_channels(self, module_name, fractions):
        params = self.module_params(module_name)
        pruned_channels = {fraction: prune_random_channels(params['weight'], self.perlayer_fractions[fraction]
        [module_name]) for fraction in fractions}
        return pruned_channels, None, None


class WeightChangeChannel(StructuredPruning):
    # prunes fraction of channels that lead to the smallest sum of change of input/output weight
    # where each term is normalized by the Frobenius norm of input/output weight of corresponding layer if norm=True
    # if reweight=True, we update next layer weights with weights that minimize change of input to next layer
    # if bias = True, we treat the bias as another channel (with no input weights) that can be pruned (only allowed when out=True)

    def __init__(self, model,  inputs=None, outputs=None, fractions=1, out=False, reweight=False, bias=False,
                 structure='neuron', prune_layers=None, norm=0, algo="sieve"):
        if bias:
            assert out is True, "cannot prune bias with out=False"
        super().__init__(model,  inputs, outputs, fractions=fractions, out=out, reweight=reweight, bias=bias,
                         structure=structure, prune_layers=prune_layers, norm=norm, algo=algo)

    def model_pruned_channels(self):
        V = {name: set(range(self.module_n_channels(name, bias=self.bias))) for name in self.channel_prunable}
        V_all = dict_to_list(V)
        params = self.params(next=self.out)
        n_channels = len(V_all)
        maps = {}
        for name in self.channel_prunable:
            mod = self.get_module(name, next=self.out)
            maps[name] = lambda channels: map_chton(channels, mod.kernel_size if isinstance(mod, nn.Conv2d) else 1)

        pruned_channels = {}
        for fraction in self.fractions:
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

        return pruned_channels, None, None


class LayerWeightChangeChannel(LayerStructuredPruning):
    # Sequentially prunes fraction of channels in each prunable layer starting from first one,
    # for each layer, prunes the channels that lead to the smallest change of input/output weight
    # if reweight=True, we update next layer weights with weights that minimize change of input to next layer
    # if bias = True, we treat the bias as another channel (with no input weights) that can be pruned (only allowed when out=True)

    def __init__(self, model,  inputs=None, outputs=None, fractions=1, out=False, reweight=False, bias=False,
                 structure='neuron', prune_layers=None, sequential=False, onelayer_results_df=None, select="min",
                 norm=0, algo="sieve"):
        if bias:
            assert out is True, "cannot prune bias with out=False"
        super().__init__(model,  inputs, outputs, fractions=fractions, out=out, reweight=reweight, structure=structure,
                         prune_layers=prune_layers, sequential=sequential, onelayer_results_df=onelayer_results_df,
                         select=select, bias=bias, norm=norm, algo=algo)

    def layer_pruned_channels(self, module_name, fractions):
        next_module = self.get_module(module_name, next=self.out)
        params = get_params(next_module)
        n_channels = self.module_n_channels(module_name, bias=self.bias)
        V = list(range(n_channels))
        map = lambda channels: map_chton(channels, next_module.kernel_size if isinstance(next_module, nn.Conv2d) else 1)
        neg_weight_change = NegWeightChange(params, self.out, self.bias, self.norm, map)
        pruned_channels, pruned_bias, new_params = {}, {}, {}
        for fraction in fractions:
            k = int(self.perlayer_fractions[fraction][module_name] * n_channels)
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
        pruned_channels[fraction] = list(set(V) - set(kept_channels))
        return pruned_channels, False, None


class InChangeChannel(StructuredPruning, ActivationMixin):
    # prunes fraction of channels that lead to the smallest sum of change of input in each prunable layer
    # with (rwchange=True) or without (rwchange=False) reweighting of next layer weights,
    # if reweight=True, we update next layer weights with weights that minimize change of input to next layer
    # if bias = True, we treat the bias as another channel (with no input weights) that can be pruned
    # where each term is normalized by the Frobenius norm of input of corresponding layer if norm=2, by # of rows of
    # activation matrix if norm=1, and no normalization if norm=0

    # TODO: obj fct is not monotone nor submodular when rwchange=False, add an algo for this case
    #  (both greedy and sieve will not work well)
    def __init__(self, model, inputs=None, outputs=None, fractions=1, reweight=False, bias=False, structure='neuron',
                 prune_layers=None, norm=0, rwchange=False, algo="sieve", patches="all"):
        super().__init__(model, inputs, outputs, fractions=fractions, reweight=reweight, bias=bias, structure=structure,
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
        pruned_channels, pruned_bias, new_params = {}, {}, {}
        for fraction in self.fractions:
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

    # TODO: obj fct is not monotone nor submodular when rwchange=False, add an algo for this case
    #  (both greedy and sieve will not work well)
    def __init__(self, model, inputs=None, outputs=None, fractions=1, reweight=False, asymmetric=False, bias=False,
                 structure='neuron', prune_layers=None, sequential=False, onelayer_results_df=None, select="min", norm=0,
                 rwchange=False, algo="sieve", patches="all", backward=False):
        super().__init__(model, inputs, outputs, fractions=fractions, reweight=reweight, asymmetric=asymmetric,
                         bias=bias, structure=structure, prune_layers=prune_layers, sequential=sequential,
                         onelayer_results_df=onelayer_results_df, select=select, norm=norm, rwchange=rwchange, algo=algo,
                         patches=patches, backward=backward)
        if asymmetric:
            self.orig_acts_all = self.activations(only_input=True)

    def layer_pruned_channels(self, module_name, fractions):
        next_module = self.get_module(module_name, next=True)
        params = get_params(next_module)
        acts = self.module_activations(next_module, only_input=True, update=True)
        orig_acts = self.orig_acts_all[next_module] if self.asymmetric else None
        if isinstance(next_module, nn.Conv2d):
            # acts shape changes from (nbatches, in_channels, H, W) --> (nbatches, npatches, in_channels * kernel_size)
            # after unfold, and transpose

            # extract only disjoint patches if patches = disjoint
            stride = next_module.kernel_size if self.patches == "disjoint" else next_module.stride
            acts = torch.nn.functional.unfold(torch.from_numpy(acts), next_module.kernel_size, next_module.dilation,
                                              next_module.padding, stride).transpose(1, 2).numpy()
            # print("n_patches = ", acts.shape[1])
            if self.patches == "random":
                n_patches = np.prod(next_module.kernel_size)
                sampled_acts = np.zeros((acts.shape[0], n_patches, acts.shape[-1]))
                sampled_patches = defaultdict(list)
                for idx in range(acts.shape[0]):
                    # from each input sample m patches, with m = kernel_size
                    sampled_patches[idx] = random.sample(range(acts.shape[1]), n_patches)
                    sampled_acts[idx] = acts[idx][sampled_patches[idx], :]
                acts = sampled_acts

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

        map = lambda channels: map_chton(channels, next_module.kernel_size if isinstance(next_module, nn.Conv2d) else 1)
        n_channels = self.module_n_channels(module_name, bias=self.bias)
        V = list(range(n_channels))
        neg_input_change = NegInputChange(acts, params, self.rwchange, self.asymmetric, orig_acts, self.bias, self.norm,
                                          map, self.backward)
        pruned_channels, pruned_bias, new_params = {}, {}, {}
        for fraction in fractions:
            k = int(self.perlayer_fractions[fraction][module_name] * n_channels)
            card = n_channels - k if self.backward else k
            if self.algo == "sieve":
                opt = SieveStreamingint(k, neg_input_change, neg_input_change.fe_max(V), 0.1)
            elif self.algo == "greedy":
                # TODO: we can obtain solution of all fraction by just running greedy on the largest one and taking
                #  corresponding subsets, but to get corresponding new params we would need to either save them
                #  throughout execution for each k we need, as done in greedy_old or recompute them from scratch
                # ks = [int(self.perlayer_fractions[fraction][module_name] * n_channels) for fraction in fractions]
                opt = Greedyint(card, neg_input_change)
            else:
                raise NotImplementedError("Selected algorithm is not supported")
            opt.fit(V)
            # selected_channels = channels to prune if backward is true, and channels to keep otherwise
            selected_channels = opt.get_solution()
            F = opt.get_f()
            if len(selected_channels) < card:
                selected_channels = fillup_sol(selected_channels, F, V, card, update=True)
            new_params[fraction] = F.get_new_params(selected_channels)
            # during debugging: check value of obj, should be equal to 1 - ||B_S W_new - A W ||_F^2 / ||A W ||_F^2
            # W = params['weight']
            # W = W.reshape(W.shape[0], -1).T
            # W_new = new_params[fraction]['weight']
            # W_new = W_new.reshape(W_new.shape[0], -1).T
            # A = orig_acts if self.asymmetric else acts
            # B = acts
            # B_S = B.copy()
            # B_S[:, map(list(set(V).difference(set(kept_channels))))] = 0
            # 1 - np.linalg.norm(B_S @ W_new - A @ W)**2 / np.linalg.norm(A @ W)**2
            # opt.get_fval()
            # F(kept_channels)

            pruned_channels[fraction] = selected_channels if self.backward else list(set(V) - set(selected_channels))
            pruned_bias[fraction] = False
            i_bias = n_channels - self.bias
            if i_bias in pruned_channels[fraction]:  # check if "bias channel" is pruned
                pruned_channels[fraction].remove(i_bias)
                pruned_bias[fraction] = True
        return pruned_channels, pruned_bias, new_params
