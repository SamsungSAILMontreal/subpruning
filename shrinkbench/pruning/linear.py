import torch.nn as nn
from .abstract import Pruning
from .utils import fraction_to_keep
from .modules import *


class LinearPruning(Pruning):
    # prune any linear layer which is not a classifier layer
    def __init__(self, model, inputs=None, outputs=None, compression=1):
        super().__init__(model, inputs, outputs, compression=compression)
        self.prunable = self.prunable_modules()
        self.fraction = fraction_to_keep(self.compression, self.model, self.prunable)

    def can_prune(self, module):
        if hasattr(module, 'is_classifier'):
            return not module.is_classifier
        if isinstance(module, (LinearMasked, nn.Linear)):
            return True
        return False