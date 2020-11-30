import torch 
import pyro
import pyro.optim as optim 
import pyro.distributions as dist
from pyro import plate
from pyro.infer import config_enumerate, TraceEnum_ELBO, Trace_ELBO, SVI
import numpy as np
import predictors
import pyro.contrib.autoguide as autoguide

def base_model(num_sites, num_days, data=None):
    with plate('sites', size=num_sites, dim=-2):
        epsilon = pyro.sample('epsilon', dist.Normal(-5, 15))
        with plate('days', size=num_days, dim=-1):
            accidents = pyro.sample('accidents', dist.Poisson(torch.exp(epsilon)), obs=data)
            
    return accidents


def log_linear_model(num_sites, num_days, num_predictors,predictors, data=None):
    betas = pyro.sample('betas', dist.Normal(0 * torch.ones(num_predictors), 10* torch.ones(num_predictors)))
    thetas = predictors @ betas
    with plate('sites', size=num_sites, dim=-2):
        epsilon = pyro.sample('epsilon', dist.Normal(0, 2)).expand(num_sites, num_days)
        with plate('days', size=num_days, dim=-1):
            thetas = thetas + epsilon
            accidents = pyro.sample('accidents', dist.Poisson(torch.exp(thetas)), obs=data) 

    return accidents


@config_enumerate
def log_linear_mm_model(num_sites, num_days, num_predictors, num_clusters, predictors, data=None):
    weight = pyro.sample('weights', dist.Dirichlet(torch.ones(num_clusters)))
    with plate('beta_components', num_predictors):
        with plate('beta_clusters', num_clusters):
            betas = pyro.sample('betas', dist.Normal(0 * torch.ones(num_clusters, num_predictors), 2 ))
       
    log_thetas = predictors @ betas.T
    with plate('epsilons', num_sites): 
        epsilon = pyro.sample('epsilon', dist.Normal(0,5))

    epsilon = (epsilon.unsqueeze(-1) * torch.ones(num_sites, num_days)).unsqueeze(-1)
    thetas = torch.exp(log_thetas + epsilon)
    with plate('sites', num_sites): 
        assignments = pyro.sample('assignments', dist.Categorical(weight))
        #selection = (assignments * torch.ones(num_days,1)).T.long()
        a = torch.arange(num_sites).expand(num_days,num_sites).permute(1,0).long()
        b = torch.arange(num_days).expand(num_sites, num_days).long()
        select = Vindex(thetas)[a,b,assignments.unsqueeze(-1)]
        accidents = pyro.sample('accidents', dist.Poisson(select).to_event(1), obs=data)
        
    return accidents



def negative_binomial_log_linear_model(num_sites, num_days, num_predictors,predictors, data=None):
    with plate("betas_plates", num_predictors): 
        betas = pyro.sample('betas', dist.Normal(torch.zeros(num_predictors), 10 * torch.ones(num_predictors)))
        
    with plate('sites_params', num_sites): 
        epsilon = pyro.sample('epsilon', dist.Normal(torch.zeros(num_sites), 5))
        epsilon = epsilon.unsqueeze(-1).expand(num_sites, num_days)
        theta = torch.exp(predictors @ betas + epsilon)
        p = pyro.sample('p', dist.Beta(1000,10))
        p = p.unsqueeze(-1).expand(num_sites, num_days)
        r = ((1-p)/p)*theta
        
        
    with plate('sites', size=num_sites, dim=-2):
        with plate('days', size=num_days, dim=-1):
            accidents = pyro.sample('accidents', dist.NegativeBinomial(r,p), obs=data) 
            
    return accidents


def train(model, guide, model_args, kappa, t_0, threshold=1, max_iters = 2000,  loss=Trace_ELBO, lr=1):
    """
    Trains model with guide. Kappa t_0 are parameters discussd in class
    Returns list with losses 
    """
    optimizer = torch.optim.Adam
    
    l1 = lambda x: 1/((t_0 + x)** kappa)
    optim_args = {'lr': lr}
    optim_params = {
        'optimizer': optimizer,
        'optim_args': optim_args,
        'lr_lambda':l1,
        }
    
    scheduler = optim.LambdaLR(optim_params)
    svi = SVI(model, guide, scheduler,loss=Trace_ELBO())

    pyro.clear_param_store()
    losses = [np.inf]
    for i in range(max_iters):
        scheduler.step()
        elbo = svi.step(**model_args)
        if np.abs(elbo - losses[-1]) < threshold:
            break
        
        losses.append(elbo)
        if i % 50 == 0: 
            print("In step {} the Elbo is {}".format(i,elbo))

    return losses[1:]

def train_log_linear(accidents, preds, predictor_labels, kappa, t_0, max_iters=2000):
    guide = autoguide.AutoDiagonalNormal(log_linear_model)
    preds = predictors.get_some_predictors(preds, predictor_labels)
    preds = torch.Tensor(preds)
    accidents = torch.Tensor(accidents)
    model_args = {
            'num_sites': accidents.shape[0],
            'num_days': accidents.shape[1],
            'num_predictors': preds.shape[2],
            'predictors': preds,
            'data': torch.tensor(accidents)
    }

    return train(log_linear_model, guide, kappa=kappa, t_0=t_0, model_args=model_args, max_iters=max_iters), guide 
