"""
Class to get functions from the posterior. 
Was copied from tutorials. 
We used this becuase pyro's version seemed to have a bug
"""

from collections import defaultdict
from pyro import poutine
from pyro.poutine.util import prune_subsample_sites
import warnings
import torch 


class Predict(torch.nn.Module):
    def __init__(self, model, guide, num_samples=800):
        super().__init__()
        self.model = model
        self.guide = guide
        self.num_samples = num_samples

    def forward(self, *args, **kwargs):
        total_samples = {}
        for i in range(self.num_samples): 
            if i % 50 == 0: 
                print("done with {}".format(i))
            guide_trace = poutine.trace(self.guide).get_trace(*args, **kwargs)
            model_trace = poutine.trace(poutine.replay(self.model, guide_trace)).get_trace(*args, **kwargs)
            for site in prune_subsample_sites(model_trace).stochastic_nodes:
                if site not in total_samples: 
                    total_samples[site] = []
                total_samples[site].append(model_trace.nodes[site]['value'])
        for key in total_samples.keys():
            total_samples[key] = torch.stack(total_samples[key])

        return total_samples
