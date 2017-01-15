import os
import numpy as np
import matplotlib.pyplot as plt
import random
import time

WINDOW = 100

class ExperimentLogger(object):
    """ Logs per-episode stats and produces a live reward plot """
    def __init__(self, logdir, prefix, num_episodes, verbose=True):
        # Directory where the plot and data log are going to be saved
        self.logdir = logdir
        # prefix of file names used for plot and log file
        self.prefix = prefix
        self.verbose = verbose
        self.num_episodes = num_episodes
        # Create output dir if required
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        # Initialize plotting machinery
        self._init_plots(num_episodes)
        # Initialize data logger
        self._init_logger()

    def _init_logger(self):
        # Create/overwrite data file for writing csv of agent's episode stats.
        self.datafile = open('%s/%s_data.csv' % (self.logdir, self.prefix), 'w')
        # Write header of csv
        self.datafile.write('episodeidx,reward,rewardlow,rewardavg,rewardhigh,loss,steps\n')

    def _init_plots(self, num_episodes):
        # Initialize underyling numpy arrays for plotting
        # Stores the instantaneous rewards for each episode
        self.instrewards = np.zeros(num_episodes) - 300
        # 100-episode rolling average
        self.avgrewards  = np.zeros(num_episodes) - 300
        # Store rolling mean + 1 standard deviation(100 episode rolling)
        self.upstd       = np.zeros(num_episodes) - 300
        # Store rolling mean - 1 standard deviation(100 episode rolling)
        self.downstd     = np.zeros(num_episodes) - 300
        # Store the per episode training losses
        self.losses      = np.zeros(num_episodes)
        # Stores the episode length of each episode
        self.steps       = np.zeros(num_episodes)
        # Counter of episodes
        self.episodeidx = 0
        # Create x axis for plotting
        x = np.arange(1, num_episodes+1)
        # Create interactive plot and figure
        plt.ion()
        self.figure = plt.figure()
        ## Create three subplots
        self.plt1 = plt.subplot(311)
        # First subplot with instantaneous rewards, rolling average,
        # rolling avg + rolling std, rolling avg - rolling std
        self.instrewards_plt = self.plt1.plot(x, self.instrewards, 'k')[0]
        self.avgrewards_plt  = self.plt1.plot(x, self.avgrewards, 'b')[0]
        self.upstd_plt       = self.plt1.plot(x, self.upstd, 'g')[0]
        self.downstd_plt     = self.plt1.plot(x, self.downstd, 'r')[0]
        self.plt1.legend([self.instrewards_plt, self.avgrewards_plt, self.upstd_plt, self.downstd_plt],
                    ['Episode reward', 'Mean reward', 'Mean + 1*stddev', 'Mean - 1*stddev'], bbox_to_anchor=(0, 1), loc='upper left', ncol=4)
        self.plt1.set_xlabel('Episodes')
        self.plt1.set_ylabel('Rewards')
        self.plt1.set_ylim(bottom=-300.0, top=350)
        self.plt1.grid(b=True, which='major', color='k', linestyle='--')
        ## Second subplot with just the losses curve.
        self.plt2 = plt.subplot(312)
        self.losses_plt = self.plt2.plot(x, self.losses, 'r')[0]
        self.plt2.set_xlabel('Episodes')
        self.plt2.set_ylabel('Loss')
        self.plt2.set_ylim(bottom=0, top=350)
        self.plt2.grid(b=True, which='major', color='k', linestyle='--')
        ## Third subplot with episode length curve.
        self.plt3 = plt.subplot(313)
        self.steps_plt = self.plt3.plot(x, self.steps, 'r')[0]
        self.plt3.set_xlabel('Episodes')
        self.plt3.set_ylabel('Steps')
        self.plt3.set_ylim(bottom=0, top=1010)
        self.plt3.grid(b=True, which='major', color='b', linestyle='--')
        
    def __del__(self):
        # Even if the user does Ctrl+C the plots needs to be saved
        self.datafile.close()
        self.figure.savefig('%s/%s_plots.png' % (self.logdir, self.prefix))

    def log_episode(self, reward, loss, numsteps):
        # Skip if episode count has exceeded buffer size.
        if len(self.instrewards) <= self.episodeidx:
            return
        # Add current episode reward to the instantaneous reward numpy array
        self.instrewards[self.episodeidx] = reward
        # Compute rolling mean and standard deviations of
        # inst. rewards of last 100 episodes
        pastidx = max(0, self.episodeidx - WINDOW)
        curmean = np.mean(self.instrewards[pastidx:self.episodeidx+1])
        curstd  = np.std(self.instrewards[pastidx:self.episodeidx+1])
        # Add mean, upstd and downstd to respective arrays
        self.avgrewards[self.episodeidx]   = curmean
        self.upstd[self.episodeidx]        = curmean + curstd
        self.downstd[self.episodeidx]      = curmean - curstd
        # Add current loss and episode length to corresponding arrays
        self.losses[self.episodeidx]       = loss
        self.steps[self.episodeidx]        = numsteps
        # Update the interactive graphs for all subplots
        self.instrewards_plt.set_ydata(self.instrewards)
        self.avgrewards_plt.set_ydata(self.avgrewards)
        self.upstd_plt.set_ydata(self.upstd)
        self.downstd_plt.set_ydata(self.downstd)
        self.losses_plt.set_ydata(self.losses)
        # Rescale the losses plot dynamically
        self.plt2.set_ylim(bottom=min(self.losses), top=max(self.losses))
        self.steps_plt.set_ydata(self.steps)
        plt.draw()
        plt.pause(0.01)
        # Print the epsiode data to screen if in verbose mode.
        if self.verbose:
            print 'Episode #%d : Reward = %.2f (%.2f, %.2f, %.2f), Loss = %f, Steps = %d' % (self.episodeidx+1,
                                                                                             reward,
                                                                                             curmean - curstd,
                                                                                             curmean,
                                                                                             curmean + curstd,
                                                                                             loss,
                                                                                             numsteps)
        # Write current episode stats as a row to the csv file
        row = {'episodeidx' : self.episodeidx + 1,
               'reward'     : reward,
               'rewardlow'  : curmean - curstd,
               'rewardavg'  : curmean,
               'rewardhigh' : curmean + curstd,
               'loss'       : loss,
               'steps'      : numsteps}
        self.datafile.write('%(episodeidx)d,%(reward)f,%(rewardlow)f,%(rewardavg)f,%(rewardhigh)f,%(loss)f,%(steps)d\n' % row)
        self.datafile.flush()
        # Increment episode counter
        self.episodeidx += 1


# Simple tester code for ExperimentLogger class using random numbers
def testrun():
    random.seed(16)
    el = ExperimentLogger(logdir='../log', prefix='testrun', num_episodes=1000)
    reward = -200.0
    loss = 200
    steps = 500
    for _ in xrange(1000):
        el.log_episode(reward, loss, steps)
        reward += 50*(random.random() - 0.5)
        loss += random.gauss(0, 50)
        steps += random.randint(-10, 10)
        time.sleep(0.2)

if __name__ == '__main__':
    testrun()
