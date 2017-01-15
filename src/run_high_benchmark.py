#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import Callback
import random

from datalogger import ExperimentLogger
from evaluator import evaluate

ENV_NAME = 'LunarLander-v2'

# Create LunarLander-v2 gym environment
env = gym.make(ENV_NAME)
# Seed to be used for initializing the environment and agent for
# repeatability
sd = 16
# Set seed for PRN generator of numpy, random module and gym env.
np.random.seed(sd)
random.seed(sd)
env.seed(sd)
nb_actions = env.action_space.n

# Start gym monitor
env.monitor.start('../monitor/LunarLander_HighBenchMark-1')

# Create a NN with 2 hidden layer each with 40 nodes and ReLU activation function
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(40))
model.add(Activation('relu'))
model.add(Dense(40))
model.add(Activation('relu'))
# Create ouput layer with linear activation function
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())


# Create keras-rl Memory buffer for experience replay
memory = SequentialMemory(limit=500000, window_length=1)
# Create epsilon greedy policy
policy = EpsGreedyQPolicy(eps=1.0)

# Create a keras-rl callback class for decaying the epsilon in epsilon greedy policy
class EpsDecayCallback(Callback):
    def __init__(self, eps_poilcy, decay_rate=0.95):
        self.eps_poilcy = eps_poilcy
        self.decay_rate = decay_rate
    def on_episode_begin(self, episode, logs={}):
        self.eps_poilcy.eps *= self.decay_rate
        print 'eps = %s' % self.eps_poilcy.eps

# Create a keras-rl callback class for live plotting of key agent statistics
class LivePlotCallback(Callback):
    def __init__(self, nb_episodes=500, prefix='highbenchmark'):
        self.nb_episodes = nb_episodes
        self.el = ExperimentLogger('../log', prefix, nb_episodes)
        
    def on_episode_end(self, episode, logs):
        rw = logs['episode_reward']
        steps = logs['nb_episode_steps']
        self.el.log_episode(rw, 0.0, steps)

# Instantiate keras-rl DQN high benchmark agent
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy, enable_double_dqn=False)
# Compile the network with Adam optimizer and mse loss metric
dqn.compile(Adam(lr=0.002, decay=2.25e-05), metrics=['mse'])

# Create a list of keras-rl callback objects
cbs = [EpsDecayCallback(eps_poilcy=policy, decay_rate=0.975)]
cbs += [LivePlotCallback(nb_episodes=500, prefix='highbenchmark_train')]
# Perform around 500 training episodes
dqn.fit(env, nb_steps=200596, visualize=False, verbose=2, callbacks=cbs)

#dqn.save_weights('../monitor/LunarLander_HighBenchMark-1/dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Close the gym monitor
env.monitor.close()

# evaluate the algorithm for 500 test episodes.
cbs = [LivePlotCallback(nb_episodes=500, prefix='highbenchmark_test')]
dqn.test(env, nb_episodes=500, nb_max_episode_steps=1000, visualize=False, callbacks=cbs)
# Evaluate the agent performance using test log files
evaluate('../log/highbenchmark_test_data.csv', '../log/highbenchmark_test_evaluation.png', 'High Benchmark')

