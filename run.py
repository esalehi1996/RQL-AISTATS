import itertools

import numpy as np
import os
from env.Tiger import TigerEnv
from env.RockSampling import RockSamplingEnv
from env.DroneSurveillance import DroneSurveillanceEnv
from env.cheesemaze import CheeseMazeEnv
from env.voicemail import VoicemailEnv
from agent import QL

import gym
import gym_minigrid
from gym_minigrid.wrappers import *
import pickle
# env = gym.make("Battleship-v0")
# env = TigerEnv()
def run_exp(args):
    list_of_test_rewards_allseeds = []
    list_of_discount_test_rewards_allseeds = []
    if args['env_name'] == 'Tiger':
        env = TigerEnv()
        test_env = TigerEnv()
        max_env_steps = args['max_env_steps']
        if args['max_env_steps'] == -1:
            max_env_steps = 100
    elif args['env_name'] == 'RockSampling':
        env = RockSamplingEnv()
        test_env = RockSamplingEnv()
        max_env_steps = args['max_env_steps']
        if args['max_env_steps'] == -1:
            max_env_steps = 200
    elif args['env_name'] == 'Cheesemaze':
        env = CheeseMazeEnv()
        test_env = CheeseMazeEnv()
        max_env_steps = args['max_env_steps']
        if args['max_env_steps'] == -1:
            max_env_steps = 100
    elif args['env_name'] == 'Voicemail':
        env = VoicemailEnv()
        test_env = VoicemailEnv()
        max_env_steps = args['max_env_steps']
        if args['max_env_steps'] == -1:
            max_env_steps = 100
    elif args['env_name'] == 'DroneSurveillance':
        env = DroneSurveillanceEnv()
        test_env = DroneSurveillanceEnv()
        max_env_steps = args['max_env_steps']
        if args['max_env_steps'] == -1:
            max_env_steps = 200
    args['max_env_steps'] = max_env_steps
    for seed in range(args['num_seeds']):
        np.random.seed(seed)
        env.seed(seed)
        test_env.seed(seed)
        list_of_test_rewards = []
        list_of_discount_test_rewards = []
        print('-------------------------------------')
        print('seed number '+str(seed)+' running')
        print('-------------------------------------')
        total_numsteps = 0
        k_steps = 0
        agent = QL(env, args)

        ls_running_rewards = []
        avg_reward = 0
        avg_episode_steps = 0
        k_episode = 0
        for i_episode in itertools.count(1):
            agent.restart()
            episode_reward = 0
            episode_steps = 0
            done = False
            obs = env.reset()
            agent.update_list_of_obs(obs)
            while not done:


                state = agent.get_state()
                # print(obs,state , agent.get_list_of_obs())


                action = agent.select_action(state)


                next_obs, reward, done, _ = env.step(action)

                agent.update_list_of_obs(next_obs)
                obs = next_obs

                next_state = agent.get_state()

                agent.update_Q(state, next_state , action , reward)





                episode_steps += 1
                total_numsteps += 1
                k_steps += 1
                episode_reward = reward + episode_reward

                state = next_state
                if total_numsteps % args['logging_freq'] == args['logging_freq']-1:
                    # print(episode_steps)
                    # evalll = True
                    saved_list_of_obs = agent.get_list_of_obs()
                    avg_reward , avg_discount_adj_reward = log_test_and_save(test_env, agent , args, avg_reward, k_episode, i_episode, total_numsteps, avg_episode_steps , seed )
                    list_of_test_rewards.append(avg_reward)
                    list_of_discount_test_rewards.append(avg_discount_adj_reward)


                    avg_reward = 0
                    avg_episode_steps = 0
                    k_episode = 0

                    agent.load_list_of_obs(saved_list_of_obs)
                if episode_steps >= max_env_steps:
                    break

            k_episode += 1
            # print(ls_states,ls_actions,ls_rewards)
            ls_running_rewards.append(episode_reward)
            avg_reward = avg_reward + episode_reward
            avg_episode_steps = episode_steps + avg_episode_steps
            # assert False

            if total_numsteps > args['num_steps']:
                break
        list_of_test_rewards_allseeds.append(list_of_test_rewards)
        list_of_discount_test_rewards_allseeds.append(list_of_discount_test_rewards)



    env.close()
    arr_r = np.zeros([args['num_seeds'], args['num_steps']//args['logging_freq']], dtype=np.float32)
    arr_d_r = np.zeros([args['num_seeds'], args['num_steps']//args['logging_freq']], dtype=np.float32)
    for i in range(args['num_seeds']):
        arr_r[i,:] = np.array(list_of_test_rewards_allseeds[i])
        arr_d_r[i,:] = np.array(list_of_discount_test_rewards_allseeds[i])

    np.save(args['logdir']+'/'+args['exp_name']+'_arr_r',arr_r)
    np.save(args['logdir'] + '/'+args['exp_name']+'_arr_d_r', arr_d_r)




def log_test_and_save(env , agent  , args , avg_reward  , k_episode , i_episode , total_numsteps , avg_episode_steps  , seed ):
    avg_running_reward = avg_reward / k_episode
    avg_reward = 0.
    avg_discount_adj_reward = 0.
    episodes = 10
    for _ in range(episodes):
        agent.restart()
        obs = env.reset()
        episode_reward = 0
        episode_rewards = []
        done = False
        steps = 0
        while not done:
            steps += 1
            agent.update_list_of_obs(obs)
            state = agent.get_state()


            action = agent.select_action(state)
            # action = env.action_space.sample()
            obs, reward, done, _ = env.step(action)
            episode_rewards.append(reward)
            episode_reward += reward

            if steps >= args['max_env_steps']:
                # print('max steps reached!!!')
                break
        avg_reward += episode_reward
        rets = []
        R = 0
        for i, r in enumerate(episode_rewards[::-1]):
            R = r + args['gamma'] * R
            rets.insert(0, R)
        avg_discount_adj_reward += rets[0]
    avg_reward /= episodes
    avg_discount_adj_reward /= episodes


    # writer.add_scalar('avg_reward/test', avg_reward, i_episode)

    print("----------------------------------------")
    # if args['model_alg'] == 'AIS':
    print("Seed: {}, Episode: {}, Total_num_steps: {},  episode steps: {}, avg_train_reward: {}, avg_test_reward: {}, avg_test_discount_adjusted_reward: {}".format(
                seed,i_episode, total_numsteps, avg_episode_steps / k_episode, avg_running_reward, avg_reward,
                avg_discount_adj_reward))
    print("----------------------------------------")


    return avg_reward , avg_discount_adj_reward
