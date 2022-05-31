import numpy as np
from PySSM import SubmodularFunctionint, SubmodularFunctionpair
from collections import defaultdict
import heapq
import torch.nn.utils.prune as prune
import torch.nn as nn
from scipy.sparse.linalg import svds
from scipy.interpolate import interp1d


def get_module(model, module_name):
    module = model
    if module_name == '':
        return module
    for s in module_name.split('.'):
        if s.isdigit():
            module = module[int(s)]
        else:
            module = getattr(module, s)
    return module


def prunable_modules(model, structure, prune_layers=None):
    # returns channel_prunable = list of module names where output channels can be pruned
    # next_module_map = map from each module name in channel_prunable to name of its next module
    # prunable = (union of channel_prunable + their next modules)
    # prunable_classes = (Conv2dMasked, nn.Conv2d) if structure == "channel" else (LinearMasked, nn.Linear)

    prunable_classes = nn.Conv2d if structure == "channel" else nn.Linear if structure == "neuron" \
        else (nn.Conv2d, nn.Linear)
    named_modules_list = list(model.named_modules())  # does not include functionals
    prunable = set()
    channel_prunable = []
    next_module_map, bn_map = {}, {}
    for i, (name, module) in enumerate(named_modules_list):
        if ('all' in prune_layers) or (name in prune_layers):
            # get next module with parameters which is not a BatchNorm module
            next_name, bn_name = None, None
            for j in range(i + 1, len(named_modules_list)):
                mod_name, mod = named_modules_list[j]
                # skip modules without parameters like ReLU, MaxPool, Dropout, but not empty sequential
                # (used for shortcuts)
                if len(list(mod.parameters())) > 0 or isinstance(mod, nn.Sequential):
                    if isinstance(mod, nn.BatchNorm2d):
                        bn_name, bn_module = mod_name, mod
                    else:
                        next_name, next_module = mod_name, mod
                        break

            if next_name is not None:
                if isinstance(module, prunable_classes) and isinstance(next_module, prunable_classes):
                    prunable = prunable.union({module, next_module})
                    channel_prunable.append(name)
                    next_module_map[name] = next_name
                    bn_map[name] = bn_name
    prunable = list(prunable)
    return prunable, channel_prunable, next_module_map, bn_map


def map_chton(channels, kernel_size):
    channel_size = np.prod(kernel_size)
    return [i + channel_size * channel for channel in channels for i in range(channel_size)]


class IndicesStructured(prune.BasePruningMethod):
    """Prune all chosen indices along chosen dim in a tensor
       function extending torch.nn.utils.prune (see https://pytorch.org/tutorials/intermediate/pruning_tutorial.html)
       pruning 'weight' removes it from the mod params and a new param called 'weight_orig' replaces it. This new param
       stores the unpruned version of the tensor and is updated during training. A new attribute 'weight' is added and
       is updated in each forward pass with weight = weight_orig * weight_mask.
    """
    PRUNING_TYPE = 'global'

    def __init__(self, dim, indices, scale):
        super(IndicesStructured, self).__init__()
        self.dim = dim
        self.indices = indices
        self.scale = scale

    def compute_mask(self, t, default_mask):
        mask = self.scale * default_mask.clone()
        # e.g.: slc = [None, None, None], if len(t.shape) = 3
        slc = [slice(None)] * len(t.shape)
        # e.g.: slc = [None, None, [0, 2, 3]] if dim=2 & indices=[0,2,3]
        slc[self.dim] = self.indices
        mask[slc] = 0
        return mask


def indices_structured(module, name, dim, indices, scale=1):
    IndicesStructured.apply(module, name, dim, indices, scale)
    return module


def undo_pruning(module, weight_mask=None):
    # reset module to its state before pruning done after the one corresponding to weight mask
    for pname in ['weight', 'bias']:
        if hasattr(module, pname + '_orig'):
            porig = getattr(module, pname + '_orig').data
            prune.remove(module, pname)
            getattr(module, pname).data = porig
    if weight_mask is not None:  # reapply weight mask if any
        prune.custom_from_mask(module, 'weight', weight_mask)


def dict_to_list(S):
    # convert dict of lists to list of tuples of (key, val)
    return [(key, ind) for key in S for ind in S[key]]


def list_to_dict(S):
    # convert list of tuples of (key, val) to dict of lists
    S_dict = defaultdict(list)
    for e in S:
        S_dict[e[0]].append(e[1])
    return S_dict


def fillup_sol(S, F, V, k, update=False):
    # adds k - |S| elts from V \ S to S, selects elts that has largest marginal gains F(i | S)
    # assumes that current sol of F is S
    VmS = set(V).difference(S)
    marg_vals = {}
    for e in VmS:
        marg_vals[e] = F.peek(S, e, len(S))
    added_elts = heapq.nlargest(k - len(S), marg_vals, key=marg_vals.__getitem__)
    if update:
        F.update(S, added_elts, len(S))
    return S + added_elts


def monotone_envelope(x, y):
    # input: list of inputs and outputs of a 1d function
    # returns monotone non-decreasing envelope of function
    y = np.minimum(y, y[-1])  # enforce max value to be the last one
    x_monotone, y_monotone = [x[0]], [y[0]]
    last_monotone_y = y[0]
    for i in range(1, len(x)):
        if y[i] >= last_monotone_y:  # y[i-1]:
            x_monotone.append(x[i])
            y_monotone.append(y[i])
            last_monotone_y = y[i]
    f = interp1d(x_monotone, y_monotone)
    return f(x)


class ScaledFctWrapper(SubmodularFunctionint):
    """
    Wrapper class which given a SubmodularFunction F
    implements SubmodularFunction interface of c . F(map(S)), where c is a scalar, and map is an optional map of indices
    """
    def __init__(self, F, c):
        super().__init__()
        self.F = F
        self.c = c

    def peek(self, S, i, pos):
        return self.c * self.F.peek(S, i, pos)

    def update(self, S, i, pos):
        self.F.update(S, i, pos)

    def __call__(self, S):
        return self.c * self.F(S)

    def get_fval(self):
        return self.c * self.F.get_fval()

    def clone(self):
        return ScaledFctWrapper(self.F.clone(), self.c)


class NegInputChange(ScaledFctWrapper):
    def __init__(self, acts, params, rwchange, asymmetric=False, orig_acts=None, bias=False, norm=0, map=lambda S: S,
                 backward=False):
        self.acts, self.params, self.rwchange, self.asymmetric, self.orig_acts, self.bias, self.norm, self.map, \
        self.backward = acts, params, rwchange, asymmetric, orig_acts, bias, norm, map, backward

        W = params['weight']
        # reshape needed for structure = channel, transpose needed because update_colsubset expects W of shape
        # (in_features, out_features)
        W = W.reshape(W.shape[0], -1).T
        A = orig_acts if asymmetric else acts
        B = acts
        if self.bias:
            W = np.concatenate((W, np.expand_dims(params["bias"], axis=0)), axis=0)  # append row of bias weights
            B = np.concatenate((B, np.ones((B.shape[0], 1))), axis=1)  # add col of ones for bias
            A = np.concatenate((A, np.ones((A.shape[0], 1))), axis=1)  # add col of ones for bias

        # B should have all 2k cols lin. indep. for function to be weakly submodular
        # print("rank:", np.linalg.matrix_rank(B))
        c = 1 / np.linalg.norm(A @ W) ** 2 if norm == 2 else 1 / A.shape[0] if norm == 1 else 1
        super().__init__(ColSubsetSel(B, W, rwchange, map, A, backward), c if backward else -c)

    def get_new_params(self, S, cur=True):
        if self.rwchange:
            W_new = self.F.get_W_new(S, cur)
            # reshape needed for structure = channel
            new_params = {'weight': W_new[:W_new.shape[0] - self.bias, :].T.reshape(self.params['weight'].shape)}
            if self.bias:
                new_params['bias'] = W_new[-1, :]
        else:
            new_params = None
        return new_params

    def fe_max(self, V):
        # get max singleton value
        fe_max = 0
        for e in V:
            fe_max = max(fe_max, self([e]))
        return fe_max

    def clone(self):
        return NegInputChange(self.acts, self.params, self.rwchange, self.asymmetric, self.orig_acts, self.bias,
                              self.norm, self.map)


class NegWeightChange(ScaledFctWrapper):
    def __init__(self, params, out=True, bias=False, norm=0, map=lambda S: S):
        self.params, self.out, self.bias, self.norm, self.map = params, out, bias, norm, map
        axis = 0 if self.out else 1
        # reshape needed for structure = channel
        W = np.swapaxes(params['weight'], 0, axis).reshape(params['weight'].shape[axis], -1)
        I = np.identity(W.shape[1])
        if self.bias:
            W = np.concatenate((W, np.expand_dims(params["bias"], axis=1)), axis=1)  # append col of bias weights
        # W should have all k cols lin. indep. or all n_l - k rows lin. indep. for function to be weakly DR-submodular
        # print("rank:", np.linalg.matrix_rank(W))
        c = -1/np.linalg.norm(W)**2 if norm == 2 else -1 / W.shape[0] if norm == 1 else -1
        super().__init__(ColSubsetSel(W, I, True, map), c)

    def fe_max(self, V):
        # get max singleton value
        fe_max = 0
        for e in V:
            fe_max = max(fe_max, self([e]))
        return fe_max

    def clone(self):
        return NegWeightChange(self.params, self.out, self.bias, self.norm, self.map)


class DecomposableFct(SubmodularFunctionpair):
    """
    Implements SubmodularFunction interface of decomposable function F(S) = sum_{l} F_l(S_l)
    given dictionary of SubmodularFunctions F_l
    assumes that all F_l's are normalized F_l(emptyset) = 0
    """
    def __init__(self, Fs):
        super().__init__()
        self.Fs = Fs
        self.fvals = {key: 0 for key in Fs.keys()}
        self.fval = 0
        # keep track of dictionary form of current solution to avoid reconstructing it each time an element is added
        self.S_dict = defaultdict(list)

    def peek(self, S, I, pos):
        # i: tuple (key, val) with key one of the keys of Fs
        # S: list of tuples (key, val) with key one of the keys of Fs
        if I.__class__ != list:
            I = [I]
        I_dict = list_to_dict(I)
        fval = self.fval
        for key in I_dict:
            fval_Ikey = self.Fs[key].peek(self.S_dict[key], I_dict[key], pos)
            fval = fval - self.fvals[key] + fval_Ikey
        return fval

    def update(self, S, I, pos):
        if I.__class__ != list:
            I = [I]
        I_dict = list_to_dict(I)
        for key in I_dict:
            self.Fs[key].update(self.S_dict[key], I_dict[key], pos)
            self.fvals[key] = self.Fs[key].get_fval()  # assumes that Fs have such function (not part of the C++ class)
            self.S_dict[key] += I_dict[key]
        self.fval = sum(self.fvals.values())

    def __call__(self, S):
        S_dict = list_to_dict(S)
        return sum(self.Fs[key](S_dict[key]) for key in S_dict)

    def get_fval(self):
        # return function value of current solution
        return self.fval

    def clone(self):
        return DecomposableFct({key: Fl.clone() for key, Fl in self.Fs.items()})


class SumNegInputChange(DecomposableFct):
    def __init__(self, acts, params, rwchange, bias=False, norm=0, maps=None):
        self.acts, self.params, self.rwchange, self.bias, self.norm, self.maps = acts, params, rwchange, \
                                                                                       bias, norm, maps
        sum_neg_input_change_fcts = {name: NegInputChange(acts[name], params[name], rwchange, bias=bias, norm=norm,
                                                          map=(lambda S: S) if maps is None else maps[name])
                                     for name in self.params.keys()}
        super().__init__(sum_neg_input_change_fcts)

    def get_new_params(self, S, cur=True):
        if self.rwchange:
            S_dict = self.S_dict if cur else list_to_dict(S)
            return {name: self.Fs[name].get_new_params(S_dict[name], cur) for name in self.Fs.keys()}
        else:
            return {name: None for name in self.Fs.keys()}

    def fe_max(self, V):
        # get max singleton value
        fe_max = 0
        for e in V:
            fe_max = max(fe_max, self([e]))
        return fe_max

    def clone(self):
        return SumNegInputChange(self.acts, self.params, self.rwchange, self.bias, self.norm, self.maps)


class SumNegWeightChange(DecomposableFct):
    def __init__(self, params, out=True, bias=False, norm=0, maps=None):
        self.params, self.out, self.bias, self.norm, self.maps = params, out, bias, norm, maps
        sum_neg_weight_change_fcts = {name: NegWeightChange(params[name], out, bias, norm,
                                                            (lambda S: S) if maps is None else maps[name])
                                      for name in self.params.keys()}
        super().__init__(sum_neg_weight_change_fcts)

    def fe_max(self, V):
        # get max singleton value
        fe_max = 0
        for e in V:
            fe_max = max(fe_max, self([e]))
        return fe_max

    def clone(self):
        return SumNegWeightChange(self.params, self.out, self.bias, self.norm, self.maps)


class ColSubsetSel(SubmodularFunctionint):
    """
    Computes (reweighted) column subset selection residual F, normalized to be zero at emptyset
    F(S) = min_W_new ||B_S W_new - A W||_F^2 if reweighted is true
    F(S) = ||B_S W - A W||_F^2 otherwise
    :param B: numpy 2D array
    :param A: numpy 2D array, if None A=B
    :param W: numpy 2D array of weights
    :param map: function to map input indices to corresponding indices of cols of A
                (useful to prune attention heads, convolution filters, etc)
    This is not really a submodular function but it will have the same interface defined in SubmodularFunction class
    in PySSM to be able to use the algorithms defined there
    """
    def __init__(self, B, W, reweighted, map, A=None, backward=False):
        super().__init__()
        if A is None or np.array_equal(A, B):
            asymmetric, A = False, B
        else:
            asymmetric = True
        self.A, self.B, self.W, self.reweighted, self.map, self.asymmetric, self.backward = A, B, W, reweighted, map, \
                                                                                            asymmetric, backward
        self.V = set(range(B.shape[-1]))
        self.added = 0

        if not self.reweighted:
            self.ATA = A.T @ A
            #print("smallest (w_im a_i)^T(w_jm a_j)is: ",
            #      np.min([np.min(np.diag(W[:, m]) @ self.ATA @ np.diag(W[:, m])) for m in range(W.shape[-1])]))
            self.WWT = W @ W.T
            # ATA should have non-negative entries for function to be submodular
            self.ATAWWT = [self.ATA[i, :] @ self.WWT[:, i] for i in self.V]
        self.reset_state()

    def reset_state(self):
        self.fval = 0  # normalized obj is 0 at emptyset
        self.Sc = list(self.V)  # can't index numpy arrays with sets
        if self.reweighted:
            self.projS_B = np.zeros_like(self.B)
            self.xS_B = np.zeros((self.B.shape[-1], self.B.shape[-1]))
            self.xS_A = self.xS_B
            # self.W_new = np.zeros_like(W)
            self.SIc, self.projSI_B, self.xSI_B, self.xSI_A = self.Sc, self.projS_B, self.xS_B, self.xS_A

    def peek(self, S, I, pos, update=False):
        # if pos >= # of elements in current sol, returns function value of adding I to current sol, otherwise computes
        # function value of swapping I with element at position pos.
        # :param S: list of indices in current solution
        # :param i: index of new element to potentially add to solution
        # pos: position at which to potentially add element

        if I.__class__ != list:
            I = [I]
        I = self.map(I)
        if pos >= self.added:
            self.SIc = list(set(self.Sc) - set(I))
            if self.reweighted:
                marginal, self.projSI_B, self.xSI_B, self.xSI_A = rw_colsubset_marg(self.B, self.W, I, self.SIc,
                                                    self.projS_B, self.xS_B, self.asymmetric, self.A, self.xS_A, update)
                return marginal + self.fval
            else:
                return colsubset_marg(self.ATA, self.WWT, self.ATAWWT, self.map(S), I, self.SIc) + self.fval

        else:
            raise NotImplementedError()

    def update(self, S, I, pos):
        # if pos >= # of elements in current sol, updates function value resulting from adding I to current sol,
        # otherwise updates function value resulting from swapping I with element at position pos.
        # S: current solution
        # i: element to add to solution
        # pos: position at which to add element

        if I.__class__ != list:
            I = [I]
        self.fval = self.peek(S, I, pos, update=True)
        self.Sc = self.SIc
        if self.reweighted:
            self.projS_B, self.xS_B, self.xS_A = self.projSI_B, self.xSI_B, self.xSI_A

        self.added += len(I)

    def get_W_new(self, S, cur=True):
        # compute new weights of current sol if cur is True else compute from scratch
        # xS_A is 0 on rows outside S, and if A = B, it is equal to identity at[S,S],
        # so xSI_A W = W[S, :] + xS_A[S, Sc] W[Sc,:]
        S = self.map(S)
        W_new = np.zeros_like(self.W)
        if cur:
            xS_A, Sc = self.xS_A, self.Sc
        else:
            Sc = list(self.V - set(S))
            _, _, _, xS_A = rw_colsubset_marg(self.B, self.W, S, Sc, np.zeros_like(self.B), np.zeros_like(self.xS_B),
                                              self.asymmetric, self.A, np.zeros_like(self.xS_A), update=True)
        if self.asymmetric:
            # during debug check if xS_A is correct by comparing with np.linalg.lstsq(self.B[:, S], self.A, rcond=None)[0]
            W_new[S, :] = xS_A[S, :] @ self.W
        else:
            W_new[S, :] = self.W[S, :] + xS_A[S, :][:, Sc] @ self.W[Sc, :]
        return W_new

    def __call__(self, S):
        # compute function value from scratch without updating state of function
        S = self.map(S)
        Sc = list(self.V - set(S))
        if self.reweighted:
            fval, _, _, _ = rw_colsubset_marg(self.B, self.W, S, Sc, np.zeros_like(self.B), np.zeros_like(self.xS_B),
                                              self.asymmetric, self.A, np.zeros_like(self.xS_A), update=False)
            return fval
        else:
            return colsubset_marg(self.ATA, self.WWT, self.ATAWWT, [], S, Sc)

    def get_fval(self):
        # return function value of current solution
        return self.fval

    def clone(self):
        return ColSubsetSel(self.B, self.W, self.reweighted, self.map, self.A)


def colsubset_marg(ATA, WWT, ATAWWT, S, I, SIc):
    """
    Compute marginal F(S U I) - F(S) with F(S) = ||A_V\S W||_F^2
    """
    Ic = SIc + S
    if len(I) <= len(Ic):
        ATAWWT_I = sum([ATA[i, I] @ WWT[I, i] for i in I])
    else:
        ATAWWT_I = sum([ATAWWT[i] - ATA[i, Ic] @ WWT[Ic, i] for i in I])

    if len(S) <= len(SIc):  # choose the more efficient method to compute marginal
        marginal = ATAWWT_I + sum([- 2 * ATAWWT[i] + 2 * ATA[i, S] @ WWT[S, i] for i in I])
    else:
        marginal = -ATAWWT_I + sum([- 2 * ATA[i, SIc] @ WWT[SIc, i] for i in I])
    return marginal


def rw_colsubset_marg(B, W, I, SIc, projS_B, xS_B, asymmetric=False, A=None, xS_A=None, update=False, backward=False):
    """
    Compute marginal F(S U I) - F(S) with F(S) = min_W_new ||B_S W_new - A W||_F^2 if backward is False
    else compute marginal F(V \ (S U I)) - F(V \ S)
    :param B: numpy 2D array
    :param W: numpy 2D array of weights
    :param I: index or list of indices of new elements
    :param SIc: list of indices corresponding to complement of S U I
    :param projS_B: numpy 2D array containing proj_S(b_j) for all columns b_j of B, i.e., projS_B = B xS_B
    :param xS_B: numpy 2D array of least squares solutions s.t. xS(b_j) = argmin_{supp(x) <= S} ||b_j - B x||_2^2
    :param A: numpy 2D array, if None A=B
    :param xS_A: numpy 2D array of least squares solutions s.t. xS(a_j) = argmin_{supp(x) <= S} ||a_j - B x||_2^2,
                 if None xS_A=xS_B
    :param update: if false don't update projSI_B, xSI_B, xSI_A
    :return:
    marginal: F(S U I) - F(S)
    projSI_B: numpy 2D array containing proj_SI(b_j) for all columns b_j of B
    xSI_B: numpy 2D array of least squares solutions s.t.  xSI(b_j) = argmin_{supp(x) <= S + I} ||b_j - B x||_2^2
    xSI_A: numpy 2D array of least squares solutions s.t.  xSI(a_j) = argmin_{supp(x) <= S + I} ||a_j - B x||_2^2
    We only need projS(b_j) and xS(b_j) for j not in S, but for simplicity in indexing projS_B and xS_B contain
    projS(b_j) and xS(b_j) for all j in V
    """
    # TODO: implement backward variant
    if I.__class__ != list:
        I = [I]
    if not asymmetric:
        A, xS_A = B, xS_B

    if len(I) == 0 or I is None:
        return 0, projS_B, xS_B, xS_A

    B_I = B[:, I]
    B_SIc = B[:, SIc]

    # compute residual
    projS_BI = projS_B[:, I]
    R_BI = B_I - projS_BI

    # compute F(S U I) - F(S) = -||R(B_I) gamma(A) W||_F^2
    # where gamma(a_j) = argmin_gamma ||a_j - R_S(B_I) gamma||_2^2,
    # if A=B, F(S U I) - F(S) = -||R(A_I) gamma(A_Sc) W(Sc,:)||_F^2
    #                         = -||R(A_I) (W(I,:) + gamma(A_SIc) W(SIc,:))||_F^2
    # since gamma(a_i) = 1, and gamma(a_j) = 0 for all j in S
    s_max = svds(R_BI, k=1, return_singular_vectors=False)[0] if len(I) > 1 else np.linalg.norm(R_BI)
    # if R_BI is a vector, largest singular value = ||R_BI||_2, but svds require 1<=k<min(R_BI.shape)
    if s_max < np.sqrt(np.prod(R_BI.shape)) * np.finfo(R_BI.dtype).eps:
        return 0, projS_B, xS_B, xS_A

    if len(I) == 1:
       # compute proj of a_j on R_ai
       R_BI_sqnorm = np.linalg.norm(R_BI)**2
       if update or not asymmetric:
           gamma_BSIc = (R_BI.T @ B_SIc) / R_BI_sqnorm
       if asymmetric:
           gamma_A = (R_BI.T @ A) / R_BI_sqnorm
           marginal = -R_BI_sqnorm * np.linalg.norm(gamma_A @ W) ** 2
       else:
           marginal = -R_BI_sqnorm * np.linalg.norm(W[I, :] + gamma_BSIc @ W[SIc, :])**2
    else:
        if update or not asymmetric:
            # set to zero singular values below machine precision of R_BI.dtype
            gamma_BSIc = np.linalg.lstsq(R_BI, B_SIc, rcond=np.finfo(R_BI.dtype).eps/s_max)[0]  # np.finfo(R_BI.dtype).eps/s_max
        if asymmetric:
            # set to zero singular values below machine precision of R_BI.dtype
            gamma_A = np.linalg.lstsq(R_BI, A, rcond=np.finfo(R_BI.dtype).eps/s_max)[0] #np.finfo(R_BI.dtype).eps/s_max
            # order of matrix multiplication chosen to reduce computation cost
            marginal = -np.linalg.norm(R_BI @ (gamma_A @ W)) ** 2
        else:
            # order of matrix multiplication chosen to reduce computation cost
            marginal = -np.linalg.norm(R_BI @ (W[I, :] + gamma_BSIc @ W[SIc, :]))**2

    if update:
        # proj_{S U i}(b_j) = proj_S(b_j) + proj_{R_bi}(b_j)
        proj_RBI_BSIc = R_BI @ gamma_BSIc
        projSI_B = projS_B.copy()
        projSI_B[:, SIc] += proj_RBI_BSIc
        projSI_B[:, I] = B_I

        # xSI(a_j) = xS(a_j) + (1_i - xS(b_i)) gamma(a_j)
        # xSI(b_j) = xS(b_j) + (1_i - xS(b_i)) gamma(b_j)
        xSI_B = xS_B.copy()
        u = - xS_B[:, I]
        u[I, :] = np.identity(len(I))
        xSI_B[:, SIc] += u @ gamma_BSIc
        xSI_B[:, I] = 0
        xSI_B[I, I] = 1
        if asymmetric:
            xSI_A = xS_A + u @ gamma_A
        else:
            xSI_A = xSI_B
    else:
        projSI_B, xSI_B, xSI_A = None, None, None

    # during debug check if marginal, xS_A & xS_B are correct using:
    # S = list(set(range(B.shape[1])).symmetric_difference(set(SIc+I)))
    # B_S = B.copy()
    # B_S[:, SIc + I] = 0
    # B_SI = B.copy()
    # B_SI[:, SIc] = 0
    # xSI_A_correct = np.linalg.lstsq(B_SI, A, rcond=None)[0]
    # xS_A_correct = np.linalg.lstsq(B_S, A, rcond=None)[0]

    # np.linalg.norm(B_SI @ xSI_A - A)
    # np.linalg.norm(B_SI @ xSI_A_correct - A)
    # np.linalg.norm((B_SI @ xSI_A_correct - A) @ W) ** 2 - np.linalg.norm((B_S @ xS_A_correct - A) @ W) ** 2
    # marginal
    return marginal, projSI_B, xSI_B, xSI_A

