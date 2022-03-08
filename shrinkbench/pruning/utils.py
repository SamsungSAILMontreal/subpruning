"""Auxiliary utils for implementing pruning strategies
"""

from collections import OrderedDict, defaultdict
from .modules import Conv2dMasked, LinearMasked

import torch
from torch import nn


def hook_applyfn(hook, model, forward=False, backward=False):
    """

    [description]

    Arguments:
        hook {[type]} -- [description]
        model {[type]} -- [description]

    Keyword Arguments:
        forward {bool} -- [description] (default: {False})
        backward {bool} -- [description] (default: {False})

    Returns:
        [type] -- [description]
    """
    assert forward ^ backward, \
        "Either forward or backward must be True"
    hooks = []

    def register_hook(module):
        if (
            not isinstance(module, nn.Sequential)
            and
            not isinstance(module, nn.ModuleList)
            and
            not isinstance(module, nn.ModuleDict)
            and
            not (module == model)
        ):
            if forward:
                hooks.append(module.register_forward_hook(hook))
            if backward:
                hooks.append(module.register_backward_hook(hook))

    return register_hook, hooks


def get_params(model, recurse=False):
    """Returns dictionary of paramters

    Arguments:
        model {torch.nn.Module} -- Network to extract the parameters from

    Keyword Arguments:
        recurse {bool} -- Whether to recurse through children modules

    Returns:
        Dict(str:numpy.ndarray) -- Dictionary of named parameters their
                                   associated parameter arrays
    """
    params = {k: v.detach().cpu().numpy().copy()
              for k, v in model.named_parameters(recurse=recurse)}
    # torch.nn.utils.prune renames weight/bias param weight_orig/bias_orig
    # and creates weight/bias attribute instead for pruned weights
    pnames = list(params.keys())
    for pname in pnames:
        if pname.endswith("_orig"):
            params.pop(pname)
            params[pname[:-5]] = getattr(model, pname[:-5]).detach().cpu().numpy().copy()

    return params


def get_activations(model, input, modules=None):

    activations = OrderedDict()

    def store_activations(module, input, output):
        if isinstance(module, nn.ReLU):
            # TODO ResNet18 implementation reuses a
            # single ReLU layer?
            return
        assert module not in activations, \
            f"{module} already in activations"
        # TODO [0] means first input, not all models have a single input
        activations[module] = (input[0].detach().cpu().numpy().copy(),
                               output.detach().cpu().numpy().copy(),)


    fn, hooks = hook_applyfn(store_activations, model, forward=True)
    if modules is None:
        model.apply(fn)
    else:  # only store activations of modules
        for module in modules:
            module.apply(fn)

    with torch.no_grad():
        model(input)

    for h in hooks:
        h.remove()

    return activations


# def get_gradients(model, inputs, outputs):
#     # TODO implement using model.register_backward_hook()
#     # So it is harder than it seems, the grad_input contains also the gradients
#     # with respect to the weights and so far order seems to be (bias, input, weight)
#     # which is confusing
#     # Moreover, a lot of the time the output activation we are looking for is the
#     # one after the ReLU and F.ReLU (or any functional call) will not be called by
#     # the forward or backward hook
#     # Discussion here
#     # https://discuss.pytorch.org/t/how-to-register-hook-function-for-functional-form/25775
#     # Best way seems to be monkey patching F.ReLU & other functional ops
#     # That'll also help figuring out how to compute a module graph
#     pass

def get_gradients(model, inputs, outputs, modules=None, loss_func=None):
    # TODO: As explained above this could have problems in some cases, so we need to test if it works properly for every
    #  model we want to use. So far tested for LeNet and vgg-11 models
    if loss_func is None:
        loss_func = nn.CrossEntropyLoss()

    gradients = OrderedDict()

    def store_gradients(module, grad_input, grad_output):
        assert module not in gradients, \
            f"{module} already in gradients"
        if isinstance(module, (LinearMasked, nn.Linear)):
            # grad_input of linear is in this order (bias, input, weight)
            ind = 1
        elif isinstance(module, (Conv2dMasked, nn.Conv2d)):
            # grad_input of conv2d is in this order (bias, input, weight)
            ind = 0
        else:
            raise NotImplementedError("TODO: check index of input/output gradient of module")
        gradients[module] = (grad_input[ind].detach().cpu().numpy().copy() if grad_input[ind] is not None else None,
                               grad_output[0].detach().cpu().numpy().copy())

    fn, hooks = hook_applyfn(store_gradients, model, backward=True)
    if modules is None:
        model.apply(fn)
    else:  # only store gradients of modules
        for module in modules:
            module.apply(fn)

    training = model.training
    model.train()
    pred = model(inputs)
    loss = loss_func(pred, outputs)
    loss.backward()
    model.zero_grad()
    model.train(training)

    for h in hooks:
        h.remove()

    return gradients


def get_param_gradients(model, inputs, outputs, loss_func=None, by_module=True):

    if loss_func is None:
        loss_func = nn.CrossEntropyLoss()

    training = model.training
    model.train()
    pred = model(inputs)
    loss = loss_func(pred, outputs)
    loss.backward()

    if by_module:
        gradients = defaultdict(OrderedDict)
        for module in model.modules():
            assert module not in gradients
            for name, param in module.named_parameters(recurse=False):
                if param.requires_grad and param.grad is not None:
                    gradients[module][name] = param.grad.detach().cpu().numpy().copy()

    else:
        gradients = OrderedDict()
        for name, param in model.named_parameters():
            assert name not in gradients
            if param.requires_grad and param.grad is not None:
                gradients[name] = param.grad.detach().cpu().numpy().copy()

    model.zero_grad()
    model.train(training)

    return gradients


def fraction_to_keep(compression, model, prunable_modules):
    """ Return fraction of params to keep to achieve desired compression ratio

    Compression = total / ( fraction * prunable + (total-prunable))
    Using algrebra fraction is equal to
    fraction = total/prunable * (1/compression - 1) + 1

    Arguments:
        compression {float} -- Desired overall compression
        model {torch.nn.Module} -- Full model for which to compute the fraction
        prunable_modules {List(torch.nn.Module)} -- Modules that can be pruned in the model.

    Returns:
        {float} -- Fraction of prunable parameters to keep to achieve desired compression
    """
    from ..metrics import model_size
    total_size, _ = model_size(model)
    prunable_size = sum([model_size(m)[0] for m in prunable_modules])
    nonprunable_size = total_size - prunable_size
    fraction = 1 / prunable_size * (total_size/compression - nonprunable_size)
    assert 0 < fraction <= 1, \
        f"Cannot compress to {1/compression} model with {nonprunable_size/total_size}" + \
        "fraction of unprunable parameters"
    return fraction
