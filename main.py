import os
import time

import argparse
import json
from run import *




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_seeds' ,type=int, default=1)
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--env_name', type=str)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--frame_stacking_len', type=int, default=4)
    parser.add_argument('--alpha', type=float, default=0.8)
    parser.add_argument('--num_steps', type=int, default=4000000)
    parser.add_argument('--max_env_steps', type=int, default=-1)
    parser.add_argument('--logging_freq' , type=int , default=5000)
    parser.add_argument('--epsilon', type=float, default=0.1)
    args = parser.parse_args()

    # convert to dictionary
    params = vars(args)

    print(args)
    print(params)

    print(args.env_name)

    # HARDCODE EPISODE LENGTHS FOR THE ENVS USED IN THIS MB ASSIGNMENT
    # if params['env_name']=='reacher-ift6163-v0':
    #     params['ep_len']=200
    # if params['env_name']=='cheetah-ift6163-v0':
    #     params['ep_len']=500
    # if params['env_name']=='obstacles-ift6163-v0':
    #     params['ep_len']=100
    #
    # ##################################
    # ### CREATE DIRECTORY FOR LOGGING
    # ##################################
    #
    # logdir_prefix = 'hw4_'  # keep for autograder
    #
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
    print(data_path)
    #
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    #
    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)
    #
    print("\n\n\nLOGGING TO: ", logdir, "\n\n\n")

    print(params)
    config_path = os.path.join(logdir,'config.json')
    print(config_path)
    with open(config_path, 'w') as fp:
        json.dump(params, fp , indent=4)
    #
    # ###################
    # ### RUN TRAINING
    # ###################
    #
    # trainer = MB_Trainer(params)
    # trainer.run_training_loop()
    run_exp(params)


if __name__ == "__main__":
    main()
