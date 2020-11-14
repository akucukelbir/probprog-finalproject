# This file contains utility functions used elswhere in the code
import numpy as np
import pyro.distributions as dist
from pyro.distributions import TorchDistribution
from torch.distributions.utils import broadcast_all
from pyro.distributions.torch import MultivariateNormal 
import torch
from numbers import Number

def make_grid_adjacency_matrix(rows, cols):
    """Function taken from:
    https://stackoverflow.com/questions/16329403/
    how-can-you-make-an-adjacency-matrix-which-would-emulate-a-2d-grid
    
    Returns the adjacency matrix representing a grid with rows and cols in args
    Notes: The matrix is unfolded among rows"""
    n = rows*cols
    M = np.zeros((n,n))
    for r in range(rows):
        for c in range(cols):
            i = r*cols + c
            # Two inner diagonals
            if c > 0: M[i-1,i] = M[i,i-1] = 1.0
            # Two outer diagonals
            if r > 0: M[i-cols,i] = M[i,i-cols] = 1.0
    return M

def make_sparse_representation_matrix(rows, cols):
    """
    Returns sparse representation of the adjacency matrix of a grid.
    In particular, returns two arrays indicated the nonzero indices
    """
    
    D = make_grid_adjacency_matrix(rows,cols)
    A1, A2 = np.nonzero(D)

    #Removes duplicates
    subset = A1 < A2
    A1 = np.expand_dims(A1[subset], 0)
    A2 = np.expand_dims(A2[subset], 0)
    return np.concatenate((A1,A2),0)


class FastCar(TorchDistribution):
    r""" Fast implementation of the the distribution for a CAR model. 
    Reference for full implementation can be found in 
    https://mc-stan.org/users/documentation/case-studies/mbjoseph-CARStan.html
    Alpha,tau,n should be integers
    eigne, Dpsarse and should be tensors with dtype float"""

    support = MultivariateNormal.support

    def __init__(self, alpha, tau, n, eigen, D_sparse, W_sparse):
        self.alpha = alpha
        self.tau = tau
        self.n = n
        self.eigen = eigen
        self.D_sparse = D_sparse
        self.W_sparse = W_sparse
        self.W_n = len(self.W_sparse[0])
        self.partial_sum = self.n * torch.log(self.tau) + torch.sum(torch.log(1- self.alpha *self.eigen))
        self._normal = dist.MultivariateNormal(torch.zeros(n), torch.eye(n))
        
        
        batch_shape = torch.Size() if isinstance(alpha, Number) else self.alpha.size()
        event_shape = torch.Size([n])
        super().__init__(batch_shape, event_shape)

        
    def sample(self, sample_shape = ()):
        return self._normal.sample()
    

    def log_prob(self, phi):
        phit_D = phi * self.D_sparse
        phit_W = torch.zeros(self.n,dtype = phit_D.dtype)
        phit_W[self.W_sparse[0]] = phit_W[self.W_sparse[0]] + phi[self.W_sparse[1]]
        phit_W[self.W_sparse[1]] = phit_W[self.W_sparse[1]] + phi[self.W_sparse[0]]
        result = (1/2) * (self.partial_sum + self.tau * (torch.sum(phit_D * phi) - self.alpha * torch.sum(phit_W * phi)))
        return result






                       
# alpha = torch.tensor(0.1)
# tau = torch.tensor(2)
# rows = 20 
# cols = 20 
# W = torch.Tensor(make_grid_adjacency_matrix(rows, cols))
# W_sparse = make_sparse_representation_matrix(rows,cols)
# D_sparse = torch.sum(W,1)
# D_invsq = torch.diag(1/torch.sqrt(D_sparse))
# D = torch.diag(torch.sum(W, 1))
# B = D_invsq @ W @ D_invsq
# eigenvalues, _ = torch.eig(B)
# eigen = eigenvalues[:,0]
# d = FastCar(alpha, tau, 20 * 20, eigen, D_sparse, W_sparse)
# a = dist.MultivariateNormal(torch.zeros(400), torch.eye(400)).sample()





