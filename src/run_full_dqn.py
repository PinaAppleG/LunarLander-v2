from agents import DQNAgent
from experiments import Experiment
from evaluator import evaluate
import random
import numpy as np

from keras.optimizers import Adam
from keras import backend as K

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

# Create a keras Huber loss function
def hubert_loss(y_true, y_pred):
    err = y_pred - y_true
    return K.mean( K.sqrt(1+K.square(err))-1, axis=-1 )

## Instantiate the improved DQN agent
ragent = DQNAgent(name='FullDQNAgent-1', state_dim=env.observation_space.shape[0], action_dim=env.action_space.n, epsdecay=0.975,
                  buffersize=500000, samplesize=32, minsamples=1000, gamma=0.99, update_target_freq=600,
                  nnparams = {  # Improved DQN setting
                      'hidden_layers'  : [ (50, 'relu'), (40, 'relu') ],
                      'loss'           : hubert_loss,
                      'optimizer'      : Adam(lr=0.0005),
                      'target_network' : True })

# Create an experiment with the LunarLander env and improved DQN agent for 500 train/test episodes
exp    = Experiment(env, ragent, logdir="../log", verbose=True, num_episodes=500)

# Training trials
exp.run(testmode=False)

# Test trials
exp.run(testmode=True)

# Evaluate the agent in test trials
evaluate('../log/LunarLander_FullDQNAgent-1_test_data.csv', '../log/LunarLander_FullDQNAgent-1_test_evaluation.png', 'Experiment')

#plt.show()
