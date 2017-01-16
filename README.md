# Capstone project - LunarLander-v2

## Problem statement
[LunarLander-v2](https://gym.openai.com/envs/LunarLander-v2) is an environment in OpenAI's gym package. The aim is to learn an agent to guide a space vehicle from starting point to the landing pad without crashing. The state space is **R**<sup>8</sup> and there are four possible actions { do nothing, fire left orientation engine, fire main engine, fire right orientation engine }.


## Software requirements

* python 2.7
* numpy
* OpenAI's gym [package](https://github.com/openai/gym)
* Keras with tensorflow or theano

### Soft dependency - no need to install in the standard way
* Keras-rl [package](https://github.com/matthiasplappert/keras-rl) - required for running the high benchmark agent.

## Setting up the project :

```bash
$ git clone https://github.com/matthiasplappert/keras-rl.git

$ git clone https://github.com/dennisfrancis/LunarLander-v2.git

$ cd keras-rl

$ git am < ../LunarLander-v2/keras-rl-testrun-fix.patch   # Apply a patch for keras-rl to work with the project

$ cd ../LunarLander-v2/src

$ ln -s ../../keras-rl/rl rl   # create a symlink to rl dir of keras-rl to src dir
```

## Running the agents

```bash
$ cd src                          # Go to source code directory

$ rm -rf ../monitor               # Required only if you have run some of the agents before

$ python run_random_agent.py      # To run random agent or low benchmark

$ python run_high_benchmark.py    # To run high benchmark agent

$ python run_basic_dqn.py         # To run basic DQN agent

$ python run_full_dqn.py          # To run improved DQN agent
```

## Logs and plots
Logs and plots will go into `log` subdir.

