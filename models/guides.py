import pyro
from pyro import plate
import pyro.distributions as dist

import torch
from torch.distributions.constraints import positive



def negative_binomial_guide(num_sites, num_days, num_predictors, predictors, data=None):
    #Parameters for p
    alpha_0 = pyro.param('alpha_0', 5*torch.ones(num_sites), constraint =positive)
    alpha_1 = pyro.param('alpha_1', 5*torch.ones(num_sites), constraint =positive)
    
    means_epsilon = pyro.param('means_epsilon', torch.zeros(num_sites))
    means_betas = pyro.param('means_betas', torch.zeros(num_predictors))
    variance_epsilon = pyro.param('variance_epsilon', torch.ones(num_sites), constraint=positive)
    variance_betas = pyro.param('variance_beta', torch.ones(num_predictors), constraint=positive)
    
    with plate("beta_plates", num_predictors):
        pyro.sample('betas', dist.Normal(means_betas, variance_betas))
    
    with plate('sites',size=num_sites): 
        pyro.sample('epsilon',dist.Normal(means_epsilon, variance_epsilon))
        pyro.sample('p', dist.Beta(alpha_0,alpha_1))
