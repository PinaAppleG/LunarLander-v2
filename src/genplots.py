# This script takes the data logs of all agent's experiments and evaluates them and
# create free form visualizations (histograms) of episode rewards and episode lengths of each agent
# to help do a comparative study

from evaluator import evaluate
import matplotlib.pyplot as plt
import pandas as pd

# List of input data logs
infiles = ['../log/highbenchmark_test_data.csv',
           '../log/LunarLander_RandomAgent-1_train_data.csv',
           '../log/LunarLander_BasicDQNAgent-1_test_data.csv',
           '../log/LunarLander_FullDQNAgent-exper_test_data.csv']

# Names of Agents
names = ['High Benchmark', 'Low Benchmark', 'Basic DQN', 'Improved DQN']
# Names of agents to appear in the free form visualizations
names_pd_cols = ['a. High Benchmark', 'b. Low Benchmark', 'c. Basic DQN', 'd. Improved DQN']

# List of output evaluation plots
evalplotfiles = ['../report_src/img/highbenchmark_test_evaluation.png',
                 '../report_src/img/randomagent.png',
                 '../report_src/img/basicdqn.png',
                 '../report_src/img/fulldqn.png']

# Free form visualization plot file names
ffviz_er_plotfile = '../report_src/img/ffviz_er.png'
ffviz_en_plotfile = '../report_src/img/ffviz_en.png'

numagents = len(infiles)
numtest_episodes = 500
# Create dataframes for creating free form visualization of episode rewards and episode lengths
df_er = pd.DataFrame(columns=names_pd_cols, index=xrange(1,numtest_episodes+1))
df_en = pd.DataFrame(columns=names_pd_cols, index=xrange(1,numtest_episodes+1))

# Output summary csv file for inclusion in the project report.
summaryfile = open('../report_src/data/summary.csv', 'w')
summaryfile.write('Agent name,E[R],STD[R],E[N],M-score\n')

# Iterate over the agents
for idx in xrange(numagents):
    print '-'*70
    print 'Agent : %s' % names[idx]
    print '-'*70
    # Evaluate the agent using evaluator module's function
    er, stdr, en, mscore = evaluate(infiles[idx], evalplotfiles[idx], names[idx])
    # Write the key stats to summary file
    summaryfile.write('%s,%.5f,%.5f,%.5f,%.5f\n' % (names[idx], er, stdr, en, mscore))
    print '='*70
    df = pd.read_csv(infiles[idx])
    # Store the episode rewards and episode lengths to repective dataframes created earlier
    df_er[names_pd_cols[idx]] = df['reward'].values
    df_en[names_pd_cols[idx]] = df['steps'].values

summaryfile.close()
# compute mean and std of episode rewards.
mean = df_er.mean()
std  = df_er.std()

# Plot histogram of episode rewards for all agents with common X-axis
axs1 = df_er.hist(bins=100, alpha=1, figsize=(10, 8), layout=(4, 1), sharex=True)
for ii in xrange(4):
    axs1[ii][0].axvline(mean[ii],           color='g', linestyle='dashed', linewidth=2)
    axs1[ii][0].axvline(mean[ii] - std[ii], color='r', linestyle='dashed', linewidth=2)
    axs1[ii][0].axvline(mean[ii] + std[ii], color='r', linestyle='dashed', linewidth=2)
fig1 = axs1[0][0].get_figure()
#[ax[0].set_ylabel('Frequency') for ax in axs1]
fig1.suptitle('Histograms of episode rewards of each agent', fontsize=13)
fig1.text(0.5, 0.04, 'Episode reward', ha='center')
fig1.text(0.04, 0.5, 'Frequency', va='center', rotation='vertical')

# compute mean and std of episode lengths.
mean = df_en.mean()
std  = df_en.std()
# Plot histogram of episode lengths for all agents with common X-axis
axs2 = df_en.hist(bins=100, alpha=1, figsize=(10, 8), layout=(4, 1), sharex=True)
for ii in xrange(4):
    axs2[ii][0].axvline(mean[ii],           color='g', linestyle='dashed', linewidth=2)
    axs2[ii][0].axvline(mean[ii] - std[ii], color='r', linestyle='dashed', linewidth=2)
    axs2[ii][0].axvline(mean[ii] + std[ii], color='r', linestyle='dashed', linewidth=2)
fig2 = axs2[0][0].get_figure()
#[ax[0].set_ylabel('Frequency') for ax in axs2]
# Give proper titles and axis labels.
fig2.suptitle('Histograms of episode lengths of each agent', fontsize=13)
fig2.text(0.5, 0.04, 'Episode length', ha='center')
fig2.text(0.04, 0.5, 'Frequency', va='center', rotation='vertical')
# Save figure to approriate files
fig1.savefig(ffviz_er_plotfile)
fig2.savefig(ffviz_en_plotfile)
#plt.show()
#raw_input('press enter...')
