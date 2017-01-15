import os.path
import pickle
import random
import numpy as np

from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras import backend as K

from memory import Memory


class RandomAgent(object):
    """ This agent selects a random action independent of the current state """

    def __init__(self, name, state_dim, action_dim, seed=16):
        """ Accepts a unique agent name, number of variables in the state,
            number of actions and initialize the agent """
        # Initialize the required class members
        self.name       = name
        self.state_dim  = state_dim
        self.action_dim = action_dim
        # Set the rand seed for repeatability
        random.seed(seed)
        ## Only for determining the mean and standard deviation of state variables.
        self.state_sum    = np.zeros(self.state_dim)
        self.statesqr_sum = np.zeros(self.state_dim)
        # Keeps count of observations
        self.observation_count = 0
        

    def decide(self, curstate, testmode=False):
        """ Accepts current state as input and returns action to take """
        # Returns a random action
        return random.randint(0, self.action_dim-1)

    def observe(self, prevstate, action, reward, curstate, done):
        """ Accepts an observation (s,a,r,s',done) as input, uses it to compute the
            mean and std of state variables """
        # Compute sum and sum of squares of state variables
        self.state_sum      += prevstate
        self.statesqr_sum   += (prevstate**2)
        # Increment observation count
        self.observation_count += 1

    def learn(self):
        # No learning in Random agent, so return 0 loss
        return 0.0

    def describe_state_variables(self):
        # Calculate mean and standard deviation from sum ans sum squared of state
        # variables
        mean    = self.state_sum / float(self.observation_count)
        sqrmean = self.statesqr_sum / float(self.observation_count)
        std     = np.sqrt(sqrmean - (mean**2))
        return mean, std


class DQNAgent(object):
    """ This agent uses DQN for making action decisions with 1-epsilon probability """

    def __init__(self, name, state_dim, action_dim, epsdecay=0.995,
                 buffersize=500000, samplesize=32, minsamples=10000,
                 gamma=0.99, state_norm_file='../params/state-stats.pkl', update_target_freq=600,
                 nnparams = {  # Basic DQN setting
                     'hidden_layers'  : [ (40, 'relu'), (40, 'relu') ],
                     'loss'           : 'mse',
                     'optimizer'      : Adam(lr=0.00025),
                     'target_network' : False }):
        """ Accepts a unique agent name, number of variables in the state,
            number of actions and parameters of DQN then initialize the agent"""
        # Unique name for the agent
        self.name       = name
        # no:of state and action dimensions
        self.state_dim  = state_dim
        self.action_dim = action_dim
        # Create buffer for experience replay
        self.memory     = Memory(maxsize=buffersize)
        # Set initial epsilon to 1.0
        self.eps        = 1.0
        # Minimum number of samples in the buffer to start learning
        self.minsamples = minsamples
        # Number of random samples to be drawn from the buffer for experience replay
        self.samplesize = samplesize
        # Decay factor for epsilon for each episode
        self.epsdecay   = epsdecay
        # Discount factor for Q learning
        self.gamma      = gamma
        # Dictionary of DQN parameters
        self.nnparams   = nnparams
        # Create the base predictor neural network
        # and if required the target neural network too.
        self._create_nns_()
        # Load the state variable normalizers from pickle file if exists
        self._load_state_normalizer_(state_norm_file)
        # Update frequency of the target network in number of steps
        self.update_target_freq = update_target_freq
        # Boolean flag indicating whether the agent started learning or not
        self.started_learning = False
        # Keeps a count of number of steps.
        self.steps = 0

    def _load_state_normalizer_(self, state_norm_file):
        self.mean = np.zeros(self.state_dim)
        self.std  = np.ones(self.state_dim)
        # Load mean and std of state dimensions from pickle file if possible
        if os.path.isfile(state_norm_file):
            self.mean, self.std = pickle.load( open( state_norm_file, 'rb') )
            print 'Loaded mean and std of state space from %s' % state_norm_file
        else:
            print 'Warning : Not using state space normalization'

    def _preprocess_state_(self, instate):
        # Normalize raw state vector by mean and std normalizers
        return ((instate - self.mean)/self.std)
        
    def _create_nns_(self):
        self.use_target_network = self.nnparams['target_network']
        # Create predictor DQN
        self.model        = self._create_model_()
        # Target network creation is requested, then create it too.
        if self.use_target_network:
            self.target_model = self._create_model_()

    def _create_model_(self):
        # Use Keras' sequential model
        model = Sequential()
        # Layer counter
        layeridx = 0
        for layer_params in self.nnparams['hidden_layers']:
            # Get the number of units in this hidden layer and
            # the name of activation function
            units, activation_name = layer_params[0], layer_params[1]
            if layeridx == 0:
                # Treat the first hidden layer specially where input layer size
                # has to be specified
                model.add(Dense(units, input_dim=self.state_dim))
            else:
                # Add hidden layer
                model.add(Dense(units))
            # Add activation layer as specified by name
            model.add(Activation(activation_name))
        # Add output layer
        model.add(Dense(self.action_dim))
        # Use a linear activation function
        model.add(Activation('linear'))
        # Compile the neural network with specified loss function and optimizer
        model.compile(loss=self.nnparams['loss'], optimizer=self.nnparams['optimizer'])
        return model

    def _update_target_model_(self):
        # Target network is used, then copy weights from predictor NN to
        # target network.
        if self.use_target_network:
            self.target_model.set_weights(self.model.get_weights())

    def decide(self, curstate, testmode=False):
        """ Accepts current state as input and returns action to take """
        # Do not do eps greedy policy for test trials
        if not testmode:
            if (random.random() <= self.eps) or (not self.started_learning):
                return random.randint(0, self.action_dim-1)
        # convert state to a matrix with one row
        s = np.array([self._preprocess_state_(curstate)])
        # Return the action with maximum predicted Q value.
        return np.argmax(self.model.predict(s)[0])

    def observe(self, prevstate, action, reward, curstate, done):
        """ Accepts an observation (s,a,r,s',done) as input, store them in memory buffer for
            experience replay """
        # Normalize both states
        prevstate_normalized = self._preprocess_state_(prevstate)
        curstate_normalized  = self._preprocess_state_(curstate)
        # Save a singe observation
        self.memory.save(prevstate_normalized, action, reward, curstate_normalized, done)
        if done:
            # Finished episode, so time to decay epsilon
            self.eps *= self.epsdecay
        if self.steps % self.update_target_freq == 0 and self.use_target_network:
            # Time to update the weights of target network
            self._update_target_model_()
        # Increment step count
        self.steps += 1

    def learn(self):
        # Do not learn if number of observations in buffer is low
        if self.memory.getsize() <= self.minsamples:
            return 0.0
        # Start training
        if not self.started_learning:
            self.started_learning = True
        # Compute a batch of inputs and targets for training the predictor DQN.
        X, y = self._compute_training_batch_()
        # Do one learning step (epoch=1) with the give (X, y)
        history = self.model.fit(X, y, batch_size=self.samplesize, nb_epoch=1, verbose=False)
        # Return the loss of this training step.
        return history.history['loss'][-1]

    def _compute_training_batch_(self):
        # Get a random sample of specified size from the buffer
        s, a, r, s1, done = self.memory.sample(self.samplesize)
        # Convert plain list of states to numpy matrices
        s  = np.array(s)
        s1 = np.array(s1)
        # Get prediction of s with predictor DQN.
        q  = self.model.predict(s)
        # Get prediction of s1 with target DQN if possible or else do with predictor DQN.
        q1 = self.target_model.predict(s1) if self.use_target_network else self.model.predict(s1)
        # Input batch X has been computed (s)
        X = s
        # Make space for storing targets.
        y = np.zeros((self.samplesize, self.action_dim))
        # Iterate over each observation in the random sample
        for idx in xrange(self.samplesize):
            reward = r[idx]
            action = a[idx]
            target = q[idx]
            # We can improve only the target for the action
            # in the observation <s,a,r,s'>
            target_for_action = reward # correct if state is final.
            if not done[idx]:
                # if not add to it the discounted future rewards per current policy
                target_for_action += ( self.gamma*max(q1[idx]) )
            target[action] = target_for_action
            # Assign computed target for the observation index = idx
            y[idx, :] = target
        return X, y
