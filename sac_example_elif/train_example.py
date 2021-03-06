import numpy as np
import math
import torch
import wandb
import simstar
import sys
import os
from tensorboardX import SummaryWriter
from collections import namedtuple
from simstarEnv import SimstarEnv
from model import Model


# training mode: 1      evaluation mode: 0
TRAIN = 0

# useful tool for visualizations
USE_WANDB  = False

# ./trained_models/EVALUATION_NAME_{EVALUATION_REWARD};      will be used only if TRAIN = 0
EVALUATION_REWARD = 507754

# "best" or "checkpoint";      will be used only if TRAIN = 0
EVALUATION_NAME = "best"

# "StraightRoad", "CircularRoad", "DutchGrandPrix", "HungaryGrandPrix"
TRACK_NAME = simstar.Environments.HungaryGrandPrix

# port number has to be the same with the SimStar.sh -nullrhi -api-port=PORT
PORT = 8080
HOST = "127.0.0.1"

# bot vehicles will be added; the configuration and speed of other vehicles could be changed from simstarEnv.py
WITH_OPPONENT = True

# when the process is required to be speeded up, the synchronized mode will have to be turned on
SYNC_MODE = True

# times speeding up the training process [1-6] 
SPEED_UP = 6


# port number can be updated from console argument
if len(sys.argv) > 1:
    PORT = int(sys.argv[1])
    print("Port is Overwritten: ", PORT)


# create model saving folders
if not os.path.exists('saved_models'):
    os.mkdir('saved_models')
if not os.path.exists('trained_models'):
    os.mkdir('trained_models')

# NOTE: users should create their own username (entity) and project in https://wandb.ai/
# initialize data logging configurations
if USE_WANDB:
    wandb.init(project='final', entity='blg638e', config={
        "TRACK_NAME": TRACK_NAME,
        "PORT": PORT,
        'WITH_OPPONENT': WITH_OPPONENT,
        'SPEED_UP': SPEED_UP,
    })
    wandb_config = wandb.config


# main training function
def train():
    env = SimstarEnv(track=TRACK_NAME,
     add_opponents=WITH_OPPONENT, synronized_mode=SYNC_MODE,
     speed_up=SPEED_UP, host=HOST, port=PORT)
    
    insize = 4 + 3 + 2
    outsize = env.action_space.shape[0]

    # default hyper-parameters, has to be modified if required
    hyperparams = {
        "lrvalue": 0.0005,
        "lrpolicy": 0.0001,
        "gamma": 0.97,
        "episodes": 15000,
        "buffersize": 250000,
        "tau": 0.001,
        "batchsize": 64,
        "alpha": 0.2,
        "maxlength": 10000,
        "hidden": 256
    }
    HyperParams = namedtuple("HyperParams", hyperparams.keys())
    hyprm = HyperParams(**hyperparams)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = Model(env, hyprm, insize, outsize).to(device)
    
    if TRAIN:
        writer = SummaryWriter(comment="_model")
    else:
        load_model(agent=agent, reward=EVALUATION_REWARD, name=EVALUATION_NAME)

    best_reward = 0.0
    average_reward = 0.0
    total_average_reward = 0.0
    total_reward = []
    total_steps = 0

    for eps in range(hyprm.episodes):
        obs = env.reset()
        state = np.hstack((obs.angle, obs.speedX, obs.curvature, obs.min_opponents, obs.min_second_opponents, obs.track7,obs.track9, obs.track11, obs.trackPos))

        episode_reward = 0.0

        for step in range(hyprm.maxlength):
            
            action = np.array(agent.select_action(state=state))
            obs, reward, done, summary = env.step(action)

            next_state = np.hstack((obs.angle, obs.speedX, obs.curvature, obs.min_opponents, obs.min_second_opponents, obs.track7,obs.track9, obs.track11, obs.trackPos))

            if (math.isnan(reward)):
                print("\nBad Reward Found\n")
                break
            
            episode_reward += reward

            if TRAIN:
                agent.memory.push(state, action, reward, next_state, done)
                if len(agent.memory.memories) > hyprm.batchsize:
                    agent.update(agent.memory.sample(hyprm.batchsize))

            if done:
                break

            state = next_state

            if np.mod(step, 250) == 0:
                print("Episode:", eps+1, " Step:", step, " Action:", action, " Reward:", reward)

        process = ((eps+1) / hyprm.episodes) * 100

        total_average_reward = average_calculation(total_average_reward, eps+1, episode_reward)
        
        total_reward.append(episode_reward)
        average_reward = torch.mean(torch.tensor(total_reward[-20:])).item()

        total_steps = total_steps + step
        lap_progress = env.progress_on_road

        if TRAIN:
            if (eps + 1) % 1000 == 0:
                print("Checkpoint is Saved !")
                save_model(agent=agent, reward=episode_reward, name="checkpoint")

            if episode_reward > best_reward:
                print("Model is Saved, Best Reward is Achieved !")
                best_reward = episode_reward
                save_model(agent=agent, reward=best_reward, name="best")
        
            tensorboard_writer(writer, eps+1, step, total_average_reward, average_reward, episode_reward, best_reward, total_steps, lap_progress*100)

        print("\nSummary: ", summary)
        print("Process: {:2.1f}%, Total Steps: {:d},  Episode Reward: {:2.3f},  Best Reward: {:2.2f},  Total Average Reward: {:2.2f}, Lap Progress (%): {:2.1f}\n".format(process, total_steps, episode_reward, best_reward, total_average_reward, lap_progress*100), flush=True)
    print("")


def average_calculation(prev_avg, num_episodes, new_val):
    total = prev_avg * (num_episodes - 1)
    total = total + new_val
    return np.float(total / num_episodes)


def tensorboard_writer(writer, eps, step_number, total_average_reward, average_reward, episode_reward, best_reward, total_steps, lap_progress):
    writer.add_scalar("step number - episode" , step_number, eps)
    writer.add_scalar("episode reward", episode_reward, eps)
    writer.add_scalar("average reward - episode", average_reward, eps)
    writer.add_scalar("average reward - total steps", average_reward, total_steps)
    writer.add_scalar("total average reward - episode", total_average_reward, eps)
    writer.add_scalar("total average reward - total steps", total_average_reward, total_steps)
    writer.add_scalar("best reward - episode", best_reward, eps)
    writer.add_scalar("best reward - total steps", best_reward, total_steps)
    writer.add_scalar("lap progress - episode", lap_progress, eps)
    writer.add_scalar("lap progress - total steps", lap_progress, total_steps)

    if USE_WANDB:
        wandb.log({
            "step number": step_number,
            "episode": eps,
            "episode_reward": episode_reward,
            "average_reward": average_reward,
            "total_average_reward": total_average_reward, 
            "best_reward": best_reward,
            "total_steps": total_steps,
            "lap_progress": lap_progress
            })


def save_model(agent, reward, name):
    path = "saved_models/" + name + "_" + str(int(reward)) + ".dat"

    torch.save({
        'actor_state_dict': agent.actor.state_dict(),
        'critic_1_state_dict': agent.critic_1.state_dict(),
        'critic_2_state_dict': agent.critic_2.state_dict(),
        'episode_reward': reward
        }, path)


def load_model(agent, reward, name):
    try:
        path = "trained_models/" + name + "_" + str(int(reward)) + ".dat"
        checkpoint = torch.load(path)

        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.critic_1.load_state_dict(checkpoint['critic_1_state_dict'])
        agent.critic_2.load_state_dict(checkpoint['critic_2_state_dict'])

    except FileNotFoundError:
        print("Checkpoint Not Found")
        return


if __name__ == "__main__":
    
    train()
