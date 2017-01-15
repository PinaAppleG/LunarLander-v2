
from agents import RandomAgent
from experiments import Experiment
from evaluator import evaluate

import matplotlib.pyplot as plt
import pickle

import gym

# Seed to be used for initializing the environment and agent for
# repeatability
seed = 16

# Create LunarLander-v2 gym environment
env = gym.make('LunarLander-v2')
# Set seed for PRN generator of gym env.
env.seed(seed)

## Instantiate a random Agent - low benchmark
ragent = RandomAgent(name='RandomAgent-1', state_dim=env.observation_space.shape[0], action_dim=env.action_space.n, seed=seed)

# Create an experiment with the LunarLander env and random agent for 500 test episodes
exp    = Experiment(env, ragent, logdir="../log", verbose=True, num_episodes=500)

# Test trials
exp.run()

# Get mean and standard deviations of the state space observations made by the agent
mean, std = ragent.describe_state_variables()

# Plot a bar chart of mean and std of state space observations made by the agent
fig = plt.figure()
plt.subplot(211)
plt.bar(list(xrange(0,8)), mean)
plt.xlabel('State space components')
plt.ylabel('Mean')
plt.title('Mean of state space components')

plt.subplot(212)
plt.bar(list(xrange(0,8)), std)
plt.xlabel('State space components')
plt.ylabel('Standard deviation')
plt.title('Standard deviation of state space components')

fig.tight_layout()

print 'Mean of observed states ='
print mean

print 'Standard deviation of observed states ='
print std

# Store the mean and std of state space observations made by the agent to pickle file
pickle.dump( (mean, std), open( "../params/state-stats.pkl", "wb" ) )

# Save the bar chart of mean and std of state space to appropriate file
fig.savefig('../log/state-space.png')

# Evaluate the agent in test trials
evaluate('../log/LunarLander_RandomAgent-1_train_data.csv', '../log/LunarLander_RandomAgent-1_evaluation.png', 'Low Benckmark')

plt.show()
