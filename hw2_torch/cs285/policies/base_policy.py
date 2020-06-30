import numpy as np
import torch
import torch.nn as nn

class BasePolicy(nn.Module):

    def __init__(self, **kwargs):
       super(BasePolicy, self).__init__(**kwargs)

    def build_graph(self):
        raise NotImplementedError

    def get_action(self, obs):
        raise NotImplementedError

    def update(self, obs, acs):
        raise NotImplementedError

    def save(self, filepath):
    	raise NotImplementedError

    def restore(self, filepath):
    	raise NotImplementedError