""" Module with examples of common pruning patterns
"""
from .abstract import Pruning
from .utils import get_activations, get_param_gradients, get_gradients


class ActivationMixin(Pruning):

    def update_activations(self, only_prunable=True):
        assert self.inputs is not None, \
            "Inputs must be provided for activations"
        self._activations = get_activations(self.model, self.inputs, self.prunable if only_prunable else None)
        self._input_activations = {mod: act[0] for mod, act in self._activations.items()}
        self._output_activations = {mod: act[1] for mod, act in self._activations.items()}

    def activations(self, only_prunable=True, only_input=False, only_output=False, update=False):
        if not hasattr(self, '_activations') or update:
            self.update_activations()
        if only_input:
            activations = self._input_activations
        elif only_output:
            activations = self._output_activations
        else:
            activations = self._activations
        if only_prunable:
            return {module: activations[module] for module in self.prunable}
        else:
            return activations

    def module_activations(self, module, only_input=False, only_output=False, update=False):
        if not hasattr(self, '_activations') or update:
            self.update_activations()
        if only_input:
            return self._input_activations[module]
        elif only_output:
            return self._output_activations[module]
        else:
            return self._activations[module]


class GradientMixin(Pruning):

    def update_gradients(self, param=True, only_prunable=True):
        assert self.inputs is not None and self.outputs is not None, \
            "Inputs and Outputs must be provided for gradients"
        if param:
            self._param_gradients = get_param_gradients(self.model, self.inputs, self.outputs)
        else:
            self._gradients = get_gradients(self.model, self.inputs, self.outputs,
                                            self.prunable if only_prunable else None)
            self._input_gradients = {mod: grad[0] for mod, grad in self._gradients.items()}
            self._output_gradients = {mod: grad[1] for mod, grad in self._gradients.items()}

    def param_gradients(self, only_prunable=True):
        if not hasattr(self, "_param_gradients"):
            self.update_gradients()
        if only_prunable:
            return {module: self._param_gradients[module] for module in self.prunable}
        else:
            return self._param_gradients

    def module_param_gradients(self, module):
        if not hasattr(self, "_param_gradients"):
            self.update_gradients()
        return self._param_gradients[module]

    def gradients(self, only_prunable=True, only_input=False, only_output=False, update=False):
        if not hasattr(self, '_gradients') or update:
            self.update_gradients(param=False, only_prunable=only_prunable)
        if only_input:
            gradients = self._input_gradients
        elif only_output:
            gradients = self._output_activations
        else:
            gradients = self._gradients
        if only_prunable:
            return {module: gradients[module] for module in self.prunable}
        else:
            return gradients

    def module_gradients(self, module, only_input=False, only_output=False, update=False):
        if not hasattr(self, '_gradients') or update:
            self.update_gradients(param=False)
        if only_input:
            return self._input_gradients[module]
        elif only_output:
            return self._output_gradients[module]
        else:
            return self._gradients[module]

    # def input_gradients(self):
    #     raise NotImplementedError("Support coming soon")
    #
    # def output_gradients(self):
    #     raise NotImplementedError("Support coming soon")
