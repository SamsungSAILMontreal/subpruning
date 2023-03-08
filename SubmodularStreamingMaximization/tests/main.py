#!/usr/bin/env python3

import numpy as np
from numpy.linalg import slogdet

from PySSM import SubmodularFunctionlist
from PySSM import IVM, FastIVM
from PySSM import RBFKernel
from PySSM import fix_seed_PySSM
from PySSM import Greedylist
from PySSM import Randomlist
from PySSM import SieveStreaminglist
from PySSM import SieveStreamingPPlist
from PySSM import ThreeSieveslist


def logdet(X):
    X = np.array(X)
    K = X.shape[0]
    kmat = np.zeros((K,K))

    for i, xi in enumerate(X):
        for j, xj in enumerate(X):
            kval = 1.0*np.exp(-np.sum((xi-xj)**2) / 1.0)
            if i == j:
                kmat[i][i] = 1.0 + kval / 1.0**2
            else:
                kmat[i][j] = kval / 1.0**2
                kmat[j][i] = kval / 1.0**2
    return slogdet(kmat)[1]


class FastLogdet(SubmodularFunctionlist):
    def __init__(self, K):
        super().__init__()
        self.added = 0
        self.K = K
        self.kmat = np.zeros((K,K))

    def peek(self, X, x, pos):
        # if self.added == 0:
        #     return 0

        if pos >= self.added:
            #X = np.array(X)
            x = np.array(x)

            row = []
            for xi in X:
                kval = 1.0*np.exp(-np.sum((xi-x)**2) / 1.0)
                row.append(kval)
            kval = 1.0*np.exp(-np.sum((x-x)**2) / 1.0)
            row.append(1.0 + kval / 1.0**2)

            self.kmat[:self.added, self.added] = row[:-1]
            self.kmat[self.added, :self.added + 1] = row
            return slogdet(self.kmat[:self.added + 1,:self.added + 1])[1]
        else:
            print("pos < solution size")
            return 0

    def update(self, X, x, pos):
        #X = np.array(X)
        if pos >= self.added:
            fval = self.peek(X, x, pos)
            self.added += 1
            return fval
        else:
            return 0

    def clone(self):
        return FastLogdet(self.K)

        # print("CLONE")
        # cloned = FastLogdet.__new__(FastLogdet)
        # print(cloned)
        # # clone C++ state
        # #SubmodularFunction.__init__(self, cloned)
        # FastLogdet.__init__(self, self.K)
        # # clone Python state
        # cloned.__dict__.update(self.__dict__)
        # print("CLONE DONE")
        # print(cloned.__call__)
        # print(self.__call__)
        # return cloned

    def __call__(self, X):
        return logdet(X)



# # optimizers = [SieveStreaming] #Greedy, Random
# # for clazz in optimizers:
# # kernel = RBFKernel(sigma=1,scale=1)
# # slowIVM = IVM(kernel = kernel, sigma = 1.0)
#
# # opt = clazz(K, slowIVM)
# #opt = clazz(K, logdet)
# # fastLogDet = FastLogdet(K)
# # opt = SieveStreaming(K, fastLogDet, 2.0, 0.1)
# # opt = SieveStreamingPP(K, fastLogDet, 2.0, 0.1)
#
# fastLogDet = FastLogdet(K)
# opt = ThreeSieves(K, fastLogDet, 2.0, 0.1, "sieve", T = 100)

# X = list(range(10))
fix_seed_PySSM(0)
X = [
    [0, 0],
    [1, 1],
    [0.5, 1.0],
    [1.0, 0.5],
    [0, 0.5],
    [0.5, 1],
    [0.0, 1.0],
    [1.0, 0.]
]

K = 3
kernel = RBFKernel(sigma=1, scale=1)
ivm = IVM(kernel=kernel, sigma=1.0)
fastivm = FastIVM(K=K, kernel=kernel, sigma=1.0)
#fastLogDet = FastLogdet(K)
# optimizers = [SieveStreaminglist(K, FastLogdet(K), 2.0, 0.1), SieveStreaminglist(K, FastLogdet(K), 2.0, 0.1), Greedylist(K, FastLogdet(K)), ThreeSieveslist(K, FastLogdet(K), 2.0, 0.1, "sieve", T=10)]
optimizers = [Greedylist(K, FastLogdet(K), 3)]
for opt in optimizers:
    opt.fit(X)

    # Alternativley, you can use the streaming interface.
    #for x in X:
    #    opt.next(x)

    fval = opt.get_fval()
    solution = np.array(opt.get_solution())
    f = opt.get_f()
    print("Found a solution with fval = {}".format(fval))
    print("kmat saved in f = {}".format(f.kmat))
    print("solution = ", solution)
