import os



import numpy as np
import math
import random

class QL(object):
    def __init__(self, env, args):


        self.args = args


        num_inputs = env.observation_space.n
        action_space = env.action_space
        print(action_space)
        print(num_inputs)
        self.obs_dim = num_inputs + 1
        self.state_size = self.obs_dim ** args['frame_stacking_len']
        print(self.state_size)
        self.frame_stacking_len = args['frame_stacking_len']

        self.num_actions = action_space.n
        self.gamma = args['gamma']
        self.alpha = args['alpha']
        self.epsilon = args['epsilon']
        self.action_space = action_space



        self.Q = np.zeros((self.state_size  , self.num_actions))
        self.state_counts = np.zeros((self.state_size,1))
        self.list_of_last_k_obs = []
        self.restart()
        # print(self.list_of_last_k_obs)


    def restart(self):
        self.list_of_last_k_obs = [0] * self.frame_stacking_len

    def get_state(self):
        state = 0

        for i in range(self.frame_stacking_len):
            state += self.list_of_last_k_obs[i] * (self.obs_dim ** i)

        return state

    def update_list_of_obs(self,obs):
        # print(self.list_of_last_k_obs)
        self.list_of_last_k_obs = [obs] + self.list_of_last_k_obs[:-1]
        # print(self.list_of_last_k_obs)


    def get_list_of_obs(self):
        return self.list_of_last_k_obs
    def load_list_of_obs(self,ls):
        self.list_of_last_k_obs = ls


    def select_action(self , state):



        if random.random() < self.epsilon:
            action = np.random.randint(0, self.num_actions)
        else:
            action = np.argmax(self.Q[state,:])

        return action

    def update_Q(self,state , next_state , action , reward):

        self.Q[state,action] += self.alpha * (reward + self.gamma * np.max(self.Q[next_state,:]) - self.Q[state,action])






    # def get_obs_dim(self):
    #     return self.obs_dim





