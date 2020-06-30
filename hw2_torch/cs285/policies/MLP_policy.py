import numpy as np

#from cs285.infrastructure.tf_utils import build_mlp
#import tensorflow_probability as tfp
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions import MultivariateNormal
from torch.autograd import Variable

class MLPPolicyPG(object):

    def __init__(self,
        ac_dim,
        ob_dim,
        n_layers,
        size,
        learning_rate=1e-4,
        training=True,
        discrete=False, # unused for now
        nn_baseline=False, # unused for now
        **kwargs):
        super().__init__(**kwargs)

        # init vars
        
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline


        self.pgpolicy=self.PGpolicy(self.ac_dim,self.ob_dim,self.n_layers,self.size)
        self.nnpolicy=self.PGpolicy(1,self.ob_dim,self.n_layers,self.size)
        self.optimizer= optim.Adam(pgpolicy.parameters,lr= self.learning_rate)
    class PGPolicy(nn.Module):

        def __init__(self,in_fea, out_fea,n_layer,hidden_size,act=nn.ReLU):
            super(Net,self).__init__()


            self.act=act()

            self.fci = nn.Linear(in_fea,hidden_size)
            self.fcs = nn.ModuleLists([nn.Linear(hidden_size,hidden_size) for i in range(n_layer)])
            self.fco = nn.Linear(hidden_size,out_fea)

        def  forward(self,x):
            x =self.act(self.fci(x))
            for l in self.fcs:  
                x=F.relu(l(x))

            x=self.fco(x)
            return x




    def policy_forward_pass(self):
        if self.discrete:
            logits_na=self.pgpolicy(self.obs_batch)
            self.parameters = logits_na
        else:
            mean = self.pgpolicy(self.obs_batch)
            std = torch.ones(self.ac_dim,dtype=torch.float)
            self.parameters = (mean,logstd)

    def action_sampling(self):
        if self.discrete:
            logits_na = self.parameters
            self.sample_ac = torch.multinomial(logits_na,1)

        else:
            mean, std =self.parameters
            self.sample_ac = torch.normal(mean,std)

    def define_log_prob(self):
        self.policy_forward_pass()
        if self.discrete:
            logits_na = self.parameters
            m = Categorical(logits_na)
            action=m.sample()
            self.logprob_n = m.log_prob(action)

        else:
            mean,std = self.parameters
            mvn = MultivariateNormal(loc,scale_tril = torch.diag(std))
            action = mvn.sample()
            self.logprob_n = mvn.log_prob(action)

    def baseline_forward_pass(self):
        self.baseline_prediction = self.nnpolicy(self.obs_batch)

    def get_action(self,obs):
        self.policy_forward_pass()
        self.action_sampling()

        return self.sample_ac


    def train_op(self)

        self.define_log_prob()


        self.loss= - self.logprob_n*self.adv_n
        optimizer.zero_grad() 
        self.loss.backward()
        self.optimizer.step()

        if self.nn_baseline:
            optimizer.zero_grad() 
            self.baseline_forward_pass()
            criterion= nn.MSELoss()
            self.baseline_loss = criterion(self.targets_nn,self.baseline_prediction)

            self.baseline_loss.backward()
            self.optimizer.step()

    def update(self,obs,acs,adv_n=None,qvals=None):
        self.obs_batch=obs
        self.action=acs
        self.adv_n=adv_n
        self.qvals=qvals
        if self.nn_baseline:
            self.targets_nn = (qvals - np.mean(qvals)/(np.std(qvals)+1e-8))


        self.train_op()
        return self.loss


            
        


