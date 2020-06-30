import time

from collections import OrderedDict
import pickle
import numpy as np
import gym
import os
#import pybullet,pybullet_envs
import torch
from cs285.infrastructure.utils import *
from cs285.infrastructure.logger import Logger


MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40

class RL_Trainer(object):

    def __init__(self,params):

        ##INIT
        self.params = params
        self.logger = Logger(self.params['logdir'])  #TODO LOGGER 

        seed = self.params['seed']
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
            self.fps = self.env.env.metadata['video.frames_per_second']    

        ##AGENT

        agent_class = self.params['agent_class']
        self.agent = agent_class(self.env,self.params['agent_params'])

    def run_training_loop(self,n_iter, collect_policy,eval_policy):

        self.total_envsteps = 0
        self.start_time =time.time()

        for itr in range(n_iter):
            print("\n\n********** Iteration %i ************"%itr)

            if itr % self.params['video_log_freq'] == 0 and self.params['video_log_freq'] != -1:
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

    def collect_training_trajectories(self,itr,collect_policy, batch_size):
        print("\nCollecting data to be used for training...")
        paths, envsteps_this_batch = sample_trajectories(self.env,collect_policy,batch_size*self.params['ep_len'],self.params['ep_len'])

        train_video_paths = None
        if self.log_video:
            print('\nCollecting train rollouts to be used for saving videos...')
            ## TODO look in utils and implement sample_n_trajectories
            train_video_paths = sample_n_trajectories(self.env, collect_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)

        return paths, envsteps_this_batch, train_video_steps
    
    def train_agent(self):

        print('\nTraining agent using sampled data from replay buffer...')
        for train_step in range(self.params['num_agent_train_steps_per_iter']):

            ob_batch,ac_batch, re_batch, next_ob_batch, terminal_batch = self.agent.sample(self.params['train_batch_size'])
            print("obs shape:{0}".format(ob_batch.shape))
            print("action shape:{0}".format(ac_batch.shape))

            self.agent.train(ob_batch,ac_batch,re_batch, next_ob_batch, terminal_batch) 

            if train_step % 100 == 0:
                print('\n Print loss for train steps:{0} is {1}'.format(train_step,self.agent.actor.loss_val)) 

    def perform_logging(self, itr, paths, eval_policy, train_video_paths):

        # collect eval trajectories, for logging
        print("\nCollecting data for eval...")
        eval_paths, eval_envsteps_this_batch = sample_trajectories(self.env, eval_policy, self.params['eval_batch_size'], self.params['ep_len'])

        # save eval rollouts as videos in tensorboard event file
        if self.log_video and train_video_paths != None:
            print('\nCollecting video rollouts eval')
            eval_video_paths = sample_n_trajectories(self.env, eval_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)

            #save train/eval videos
            print('\nSaving train rollouts as videos...')
            self.logger.log_paths_as_videos(train_video_paths, itr, fps=self.fps, max_videos_to_save=MAX_NVIDEO,
                                            video_title='train_rollouts')
            self.logger.log_paths_as_videos(eval_video_paths, itr, fps=self.fps,max_videos_to_save=MAX_NVIDEO,
                                             video_title='eval_rollouts')

        # save eval metrics
        if self.log_metrics:
            # returns, for logging
            train_returns = [path["reward"].sum() for path in paths]
            eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]

            # episode lengths, for logging
            train_ep_lens = [len(path["reward"]) for path in paths]
            eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

            # decide what to log
            logs = OrderedDict()
            logs["Eval_AverageReturn"] = np.mean(eval_returns)
            logs["Eval_StdReturn"] = np.std(eval_returns)
            logs["Eval_MaxReturn"] = np.max(eval_returns)
            logs["Eval_MinReturn"] = np.min(eval_returns)
            logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

            logs["Train_AverageReturn"] = np.mean(train_returns)
            logs["Train_StdReturn"] = np.std(train_returns)
            logs["Train_MaxReturn"] = np.max(train_returns)
            logs["Train_MinReturn"] = np.min(train_returns)
            logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

            logs["Train_EnvstepsSoFar"] = self.total_envsteps
            logs["TimeSinceStart"] = time.time() - self.start_time


            if itr == 0:
                self.initial_return = np.mean(train_returns)
            logs["Initial_DataCollection_AverageReturn"] = self.initial_return

            # perform the logging
            for key, value in logs.items():
                print('{} : {}'.format(key, value))
                self.logger.log_scalar(value, key, itr)
            print('Done logging...\n\n')

            self.logger.flush()   


