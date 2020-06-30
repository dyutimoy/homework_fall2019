import os
import time

import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical

import sys
sys.path.append('/content/homework_fall2019/hw2')
from cs285.infrastructure.rl_trainer import RL_Trainer
from cs285.agents.pg_agent import PGAgent




class RL_Trainer(object):

    def __init__(self,params):

        ##INIT


        self.params = params
        self.logger = Logger(self.params['logdir'])  #TODO LOGGER 

        seed = self.params['seeds']
        np.random.seed(seed)


        ##ENV

        self.env = gym.make(self.params['env_name'])
        self.env.seed(seed)


        #max length of episodes
        self.params['ep_len'] = self.params['ep_len'] or self.env.spec.max_episode_steps
        MAX_VIDEO_LEN = self.params['ep_len']

        #Check discrete or continuous
        discrete= isinstance(self.env.action_space, gym.spaces.Discrete)
        self.params['agent_params']['discrete'] = discrete


        ob_dim = self.env.observation_space.shape[0]
        ac_dim = self.env.action_space.n if discrete else self.env.action_space.shape[0]
        self.params['agent_params']['ac_dim']= ac_dim
        self.params['agent_params']['ob_dim']= ob_dim

        #video save
        if 'model' in dir(self.env):
            self.fps = 1/self.env.model.opt.timestep  #what is model
        #else mostly I guess
        else:
            self.fps = self.env.env.metada['video.frames_per_second']    

        ##AGENT

        agent_class = self.params['agent_class']
        self.agent = agent_class(self.env,self.params['agent_params'])

    def run_training_loop(self,n_iter, collect_policy,eval_policy):

        self.total_envsteps = 0
        self.start_time =time.time()

        for itr in range(n_iter):
            print("\n\n********** Iteration %i ************"%itr)

            if ite % self.params['video_log_freq'] == 0 and self.params['video_log_freq'] != -1:
                self.log_video = True
            else:
                self.log_video = False

            # decide if metrics should be logged
            if itr % self.params['scalar_log_freq'] == 0:
                self.log_metrics = True
            else:
                self.log_metrics = False


            training_returns = self.collect_training_trajectories(itr,collect_policy,self.params['batch_size'])
            paths, envsteps_this_batch, train_video_paths = training_paths
            self.total_envsteps +=envsteps_this_batch

            self.agent.add_to_replay_buffer(paths)

            self.train_agent()

    def collect_training_trajectories(self,itr,collect_policy, batch_size);
        print("\nCollecting data to be used for training...")
        paths, envsteps_this_batch = sample_trajectories(self.env,collect_policy,batch_size*self.params['ep_len'],self.params['ep_len'])

        train_video_paths = None
        if self.log_video:
            print('\nCollecting train rollouts to be used for saving videos...')
            ## TODO look in utils and implement sample_n_trajectories
            train_video_paths = sample_n_trajectories(self.env, collect_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)

        return paths, envsteps_this_batch, train_video_steps
        

class PG_Trainer(object):

    def __init__(self, params):
    
        #####################
        ## SET AGENT PARAMS
        #####################

        computation_graph_args = {
            'n_layers': params['n_layers'],
            'size': params['size'],
            'learning_rate': params['learning_rate'],
            }

        estimate_advantage_args = {
            'gamma': params['discount'],
            'standardize_advantages': not(params['dont_standardize_advantages']),
            'reward_to_go': params['reward_to_go'],
            'nn_baseline': params['nn_baseline'],
        }

        train_args = {
            'num_agent_train_steps_per_iter': params['num_agent_train_steps_per_iter'],
        }

        agent_params = {**computation_graph_args, **estimate_advantage_args, **train_args}

        self.params = params
        self.params['agent_class'] = PGAgent
        self.params['agent_params'] = agent_params
        self.params['batch_size_initial'] = self.params['batch_size']

        ################
        ## RL TRAINER
        ################

        self.rl_trainer = RL_Trainer(self.params)

    def run_training_loop(self):

        self.rl_trainer.run_training_loop(
            self.params['n_iter'], 
            collect_policy = self.rl_trainer.agent.actor,
            eval_policy = self.rl_trainer.agent.actor,
            )


def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='todo')
    parser.add_argument('--n_iter', '-n', type=int, default=200)
        
    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--nn_baseline', action='store_true')
    parser.add_argument('--dont_standardize_advantages', '-dsa', action='store_true')
    parser.add_argument('--batch_size', '-b', type=int, default=1000) #steps collected per train iteration
    parser.add_argument('--eval_batch_size', '-eb', type=int, default=400) #steps collected per eval iteration

    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1)
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=64)

    parser.add_argument('--ep_len', type=int) #students shouldn't change this away from env's default
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--use_gpu', '-gpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)
    parser.add_argument('--video_log_freq', type=int, default=-1)   # video log disabled
    parser.add_argument('--scalar_log_freq', type=int, default=1)

    parser.add_argument('--save_params', action='store_true')

    args = parser.parse_args()

    # convert to dictionary
    params = vars(args)
    
    # for this assignment, we train on everything we recently collected
    # so making train_batch_size=batch_size 
    params['train_batch_size']=params['batch_size']

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    logdir_prefix = 'pg_'

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    logdir = logdir_prefix + args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    ###################
    ### RUN TRAINING
    ###################

    trainer = PG_Trainer(params)
    trainer.run_training_loop()


if __name__ == "__main__":
    main()
