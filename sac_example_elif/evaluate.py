"""
General Introduction:
    * the objective of this script is that students could see evaluation process of their models against other RL models
    * students could evaluate their best trained models with this script
    * the evaluation is done againts uploaded default best trained models with sac with default parameters
    * students should give path to their best models in LOAD_CUSTOM_MODEL
    * opponent vehicle number could be changed in OPPONENT_NUM and their initial racing positions in AGENT_LOCATIONS
    * by changing CONTROL_OTHER_AGENTS boolean students could evaluate their models against default RL trained model or IDM (autopilot) vehicles
    * there are already designed bult-in tracks which could be changed from the list: [HungaryGrandPrix, DutchGrandPrix, CircularRoad, StraightRoad]
    * its important to modify load_checkpoint() function if your model's network structure is not default Soft-Actor-Critic Network
    * evaluation in each step is done until NUM_EVAL_STEPS iteration number is reached, however this could be changed
    * in the competition, student will race their models against each others RL models
"""

import torch
import simstar
import numpy as np
from simstarEnv import SimstarEnv
from collections import namedtuple
from model import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print()

# user's own best model could be loaded from saved_models folder
# TODO: right now default model is loaded, however users should evaluate their own models 
#LOAD_CUSTOM_MODEL = 'saved_models/checkpoint_27214.dat'
LOAD_CUSTOM_MODEL = 'default_model/best_253953.dat'

# default model is trained with given sac agent code and training is breaked at 320K steps
LOAD_DEFAULT_MODEL = 'default_model/best_253953.dat'
#LOAD_DEFAULT_MODEL='saved_models/checkpoint_27214.dat'
NUM_EVAL_EPISODE = 5
NUM_EVAL_STEPS = 4000

ADD_OPPONENTS = True
OPPONENT_NUM = 5

# True: controls opponent vehicles with loaded default model weights
# False: opponent vehicles will be controled with IDM (Intelligent Driver Model)
CONTROL_OTHER_AGENTS = False

# initial locations of the opponents could be defined in meters with respect to the main agent
AGENT_LOCATIONS = [25, 50, 75, 100, 125]
if CONTROL_OTHER_AGENTS:
    AGENT_INIT_SPEEDS = [0, 0, 0, 0, 0]
else:
    AGENT_INIT_SPEEDS = [45, 80, 55, 100, 40]


def evaluate(port=8080):
    env = SimstarEnv(track=simstar.Environments.HungaryGrandPrix,  port=port, add_opponents=ADD_OPPONENTS, num_opponents=OPPONENT_NUM, speed_up=1, synronized_mode=True)
    
    # update agent init configs
    env.agent_locations = AGENT_LOCATIONS
    env.agent_speeds = AGENT_INIT_SPEEDS

    # total length of chosen observation states
    insize = 4 + env.track_sensor_size + env.opponent_sensor_size
    outsize = env.action_space.shape[0]

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

    # load actor network from checkpoint
    agent = Model(env=env, params=hyprm, n_insize=insize, n_outsize=outsize).to(device)
    load_checkpoint(agent)
    
    if CONTROL_OTHER_AGENTS:
        opponent_agent = Model(env=env, params=hyprm, n_insize=insize, n_outsize=outsize).to(device)
        load_checkpoint(opponent_agent)

    total_reward = 0

    for eps in range(NUM_EVAL_EPISODE):
        obs = env.reset()
        state = np.hstack((obs.angle, obs.track, obs.trackPos, obs.speedX, obs.speedY, obs.opponents))

        agent_observations = env.get_agent_observations()
        if CONTROL_OTHER_AGENTS:
            env.change_opponent_control_to_api()

        agent_actions = []

        epsisode_reward = 0

        for i in range(NUM_EVAL_STEPS):
            action = np.array(agent.select_action(state=state))

            if CONTROL_OTHER_AGENTS:
                # set other agent actions
                env.set_agent_actions(agent_actions)

            obs, reward, done, summary = env.step(action)
            next_state = np.hstack((obs.angle, obs.speedX, obs.speedY, obs.opponents, obs.track, obs.trackPos))

            if CONTROL_OTHER_AGENTS:
                agent_actions = []
                for agent_obs in agent_observations:
                    agent_state = np.hstack((agent_obs.angle, agent_obs.speedX, agent_obs.speedY, agent_obs.opponents, agent_obs.track, agent_obs.trackPos))
                    agent_action = np.array(opponent_agent.select_action(state=agent_state))
                    agent_actions.append(agent_action)
                
                # get other agent observation
                agent_observations = env.get_agent_observations()

            epsisode_reward += reward

            if done:
                # do not restart 
                if "accident" != summary['end_reason']:
                    break
                
            state = next_state

        total_reward += epsisode_reward
        print("Episode: %d, Reward: %.1f"%(i, epsisode_reward))
    
    print("Average reward over %d episodes: %.1f"%(NUM_EVAL_EPISODE, total_reward/NUM_EVAL_EPISODE))


def load_checkpoint(agent): 
    try:
        checkpoint = torch.load(LOAD_CUSTOM_MODEL)
        print("keys are: ",checkpoint.keys())

        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.critic_1.load_state_dict(checkpoint['critic_1_state_dict'])
        agent.critic_2.load_state_dict(checkpoint['critic_2_state_dict'])

        if 'epsisode_reward' in checkpoint: reward = float(checkpoint['epsisode_reward']) 
    except FileNotFoundError:
        raise FileNotFoundError("custom model weights are not found")


if __name__ == "__main__":
    evaluate()