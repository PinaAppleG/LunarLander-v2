import os
from datalogger import ExperimentLogger


class Experiment(object):

    def __init__(self, env, agent, logdir="../log", verbose=True, num_episodes=1000):
        """ Takes a gym environment and an agent as inputs
            and initialize an experiment instance """
        # Save gym environment object
        self.env = env
        # Save agent object
        self.agent = agent
        # Total episodes in both test and train phase
        self.num_episodes = num_episodes
        # Compute environment name
        self.envname = env.__str__().split(' ')[0].lstrip('<')
        self.agentname = self.agent.name
        # Unique gym monitor dir name
        self.monitordir = '../monitor/' + self.envname + '_' + self.agentname
        self.prefix = self.envname + '_' + self.agentname
        # Create train ExperimentLogger object
        self.logger_train = ExperimentLogger(logdir=logdir, prefix=(self.prefix  + "_train"), num_episodes=self.num_episodes, verbose=verbose )
        # Create test ExperimentLogger object
        self.logger_test  = ExperimentLogger(logdir=logdir, prefix=(self.prefix + "_test"),   num_episodes=self.num_episodes, verbose=verbose )
        # Create monitor root dirs
        os.makedirs(self.monitordir)
        # Start gym monitor
        self.env.monitor.start(self.monitordir)

    def __del__(self):
        # Stop gym monitor gracefully
        self.env.monitor.close()

    def run(self, testmode=False):
        """ Run num_episodes episodes on self.env with self.agent.
            It will let the agent learn only if testmode==False.
        """
        # Run the experiment with specified number of episodes
        for episodeidx in xrange(self.num_episodes):
            # Get current state from gym env
            curstate = self.env.reset()
            # Flag to indicate if episode has ended.
            done = False
            # Training loss for the episode
            loss = 0.0
            # Episode length of the episode
            numsteps = 0
            # Total reward gained by the agent in current episode
            totreward = 0.0
            # Do till episode finishes
            while not done:
                # Increment step counter (episode length)
                numsteps += 1
                # Make the agent choose an action for current state
                action = self.agent.decide(curstate, testmode=testmode)
                prevstate = curstate
                # Let the gym env know about the action to be taken
                # and get next state, reward
                curstate, reward, done, _ = self.env.step(action)
                # Add the step reward to episode reward
                totreward += reward
                if not testmode:
                    # If in train mode, let the agent get the observation tuple (s,a,r,s')
                    self.agent.observe(prevstate, action, reward, curstate, done)
                    # Let the agent learn and add back this train loss to episode loss
                    loss += self.agent.learn()
            # Log the episode data to the respective data loggers
            if testmode:
                self.logger_test.log_episode(totreward, loss, numsteps)
            else:
                self.logger_train.log_episode(totreward, loss, numsteps)
