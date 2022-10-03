"""functions for choosing channels
"""
import numpy as np
import random
import math
import torch
from collections import defaultdict
from scipy import optimize


def prune_random_channels(tensor, fraction):
    # assumes channels is the first axis
    n_channels = tensor.shape[0]
    pruned_channels = random.sample(range(n_channels), n_channels - int(fraction*n_channels))
    return pruned_channels


def map_channel_importances(fn, importances):
    return {module_name: fn(params, module_name)
            for module_name, params in importances.items()}


def flatten_channel_importances(importances):
    return np.concatenate([
        importance.flatten()
        for importance in importances.values()
    ])


def prune_threshold_channels(importances, threshold):
    # compute mask for tensor masking weights of channels with abs(importance) <= threshold, and corresponding bias if any
    pruned_channels = np.where(np.abs(importances) <= threshold)[0]
    return pruned_channels


# utilities for LayerSamplingChannel adapted from https://github.com/lucaslie/torchprune

def expected_unique(probabilities, sample_size):
    """Get expected number of unique samples.

    This computes the expected number of unique samples for a multinomial
    distribution from which we sample a fixed number of times.

    """
    vals = 1 - (1 - probabilities) ** sample_size
    expectation = torch.sum(torch.as_tensor(vals), dim=-1)
    return torch.ceil(expectation)


# TODO: if we always have uniform=False remove this function
def _adapt_uniform_sample_size(probabilities, sample_size_target):
    """Do _adapt_sample_size for a uniform distribution (faster)."""
    idx = probabilities.nonzero()
    sample_size_target = int(sample_size_target)
    n_size = idx.shape[0]

    sample_size_target = min(sample_size_target, n_size - 1)
    # If uniform sampling, we have a closed form solution.
    ret = math.log(n_size / (n_size - sample_size_target)) / math.log(
        n_size / (n_size - 1)
    )
    return int(math.ceil(ret))


def adapt_sample_size(probabilities, size_pruned, uniform=False):
    """Get the number of samples to obtain some desired unique number.

    Args:
        probabilities (array): probabilities of multinomial distribution.
        size_pruned (int): desired number of unique samples
        uniform (bool, optional): indicate uniform dist. Defaults to False.

    Returns:
        int: number of times to sample to obtain desired unique samples.
    """
    # check for case of only one probability
    if probabilities.numel() == 1 and probabilities.item() > 0:
        return size_pruned
    if size_pruned in (0, 1):
        return size_pruned

    probabilities = probabilities.flatten()
    # check for uniform
    if uniform:
        # Adapting the sample size for uniform does not require solving an
        # optimization problem and has a closed form solution, so we make a
        # different call for efficiency.
        return _adapt_uniform_sample_size(probabilities, size_pruned)

    idx = probabilities.nonzero()
    size_pruned = int(size_pruned)
    n_size = idx.shape[0]

    if n_size == 0 or n_size == 1:
        return n_size

    size_pruned = min(size_pruned, n_size - 1)
    # p = np.array(probabilities[idx])
    probs = probabilities[idx].flatten()

    def f_diff(x_arg):
        return expected_unique(probs, int(x_arg)) - size_pruned

    # A lower bound on the number of required samples has a closed form
    # solution using Jensen's inequality and the concavity of 1 - (1-x)^m.
    arg_min = math.log(n_size / (n_size - size_pruned)) / math.log(
        1 / (1 - probs.mean())
    )
    arg_min = int(math.floor(arg_min))

    arg_max = max(1, arg_min)
    while f_diff(arg_max) < 0:
        arg_max *= 2

    sample_size = int(optimize.bisect(f_diff, arg_min, arg_max))

    return sample_size


def flatten_all_but_last(tensor):
    """Flatten all dimensions except last one and return tensor."""
    return tensor.view(tensor[..., 0].numel(), tensor.shape[-1])