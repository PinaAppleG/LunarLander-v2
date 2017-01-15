
from agents import DQNAgent
from experiments import Experiment
from evaluator import evaluate
import random
import numpy as np

from keras.optimizers import Adam

import gym

# Seed to be used for initializing the environment and agent for
# repeatability
seed = 16

# Create LunarLander-v2 gym environment
env = gym.make('LunarLander-v2')
# Set seed for PRN generator of numpy, random module and gym env.
np.random.seed(seed)
random.seed(seed)
env.seed(seed)

## Instantiate a basic DQN agent
ragent = DQNAgent(name='BasicDQNAgent-1', state_dim=env.observation_space.shape[0], action_dim=env.action_space.n, epsdecay=0.975,
                  buffersize=500000, samplesize=32, minsamples=1000, gamma=0.99,
                  nnparams = {  # Basic DQN setting
                      'hidden_layers'  : [ (40, 'relu'), (40, 'relu') ],
                      'loss'           : 'mse',
                      'optimizer'      : Adam(lr=0.00025),
                      'target_network' : False })

# Create an experiment with the LunarLander env and basic DQN agent for 500 train/test episodes
exp    = Experiment(env, ragent, logdir="../log", verbose=True, num_episodes=500)

# Training trials
exp.run(testmode=False)

# Test trials
exp.run(testmode=True)

# Evaluate the agent in test trials
evaluate('../log/LunarLander_BasicDQNAgent-1_test_data.csv', '../log/LunarLander_BasicDQNAgent-1_test_evaluation.png', 'Basic DQN')

#plt.show()
