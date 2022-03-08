"""functions for choosing channels
"""
import numpy as np
import random
from collections import defaultdict


def prune_random_channels(tensor, fraction):
    # assumes channels is the first axis
    n_channels = tensor.shape[0]
    pruned_channels = random.sample(range(n_channels), n_channels - int(fraction*n_channels))
    return pruned_channels


def map_channel_importances(fn, importances):
    return {module_name: fn(params)
            for module_name, params in importances.items()}


def flatten_channel_importances(importances):
    return np.concatenate([
        importance.flatten()
        for importance in importances.values()
    ])


def prune_threshold_channels(importances, threshold):
    # compute mask for tensor masking weights of channels with abs(importance) <= threshold, and corresponding bias if any
    pruned_channels = np.where(np.logical_and(importances <= threshold, importances >= -threshold))[0]
    return pruned_channels


