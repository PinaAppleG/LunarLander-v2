import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Calculate key statistics of an agent from its data log and also
# create a plot with reward and episode length panels with Mscore
# printed above
def evaluate(datafile, plotfname, agentname):
    # Load data log for the agent
    df = pd.read_csv(datafile)
    # Calculate expected reward
    er = df["reward"].mean()
    # Calculate standard deviation of rewards
    stdr = df["reward"].std()
    # Expected episode length
    en = df["steps"].mean()
    # Scaling factor used in M-score formula
    psi = 0.00647582
    # Sharpe ratio
    sharpe = er/stdr
    # Print all stats to screen
    print 'E[R] = %.5f' % er
    print 'STD[R] = %.5f' % stdr
    print 'sharpe ratio = %.5f' % sharpe
    print 'E[N] = %.5f' % en
    #print 'psi = %.8f' % (sharpe/en)
    score = 0.0
    if er > 0.0:
        # M-score if expected reward is positive
        score = sharpe - psi*en
    else:
        # M-score if expected reward is negative
        score = er
    # Print M-score to screen
    print 'score = %.5f' % score
    ###
    # create plots of rewards and episode lengths
    x = np.arange(1, df.shape[0]+1)
    fig = plt.figure(figsize=(12,9))
    plt1 = plt.subplot(211)
    instrewards_plt = plt1.plot(x, df['reward'], 'k',     label='Episode reward')[0]
    avgrewards_plt  = plt1.plot(x, df['rewardavg'], 'b',  label='Mean')[0]
    upstd_plt       = plt1.plot(x, df['rewardhigh'], 'g', label='Mean + 1*st.dev' )[0]
    downstd_plt     = plt1.plot(x, df['rewardlow'], 'r',  label='Mean - 1*st.dev')[0]
    plt1.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=4)
    plt1.set_xlabel('Episodes')
    plt1.set_ylabel('Rewards')
    plt1.grid(b=True, which='major', color='k', linestyle='--')
    plt2 = plt.subplot(212)
    steps_plt = plt2.plot(x, df['steps'], 'r')[0]
    plt2.set_xlabel('Episodes')
    plt2.set_ylabel('Episode length')
    plt2.grid(b=True, which='major', color='k', linestyle='--')
    fig.tight_layout()
    plt.suptitle('%s agent score = %.4f' % (agentname, score), fontsize=15)
    plt.subplots_adjust(top=0.92)
    #print "psi = %.5f" % (1.0/score)
    # Save the figure to speficied file name
    fig.savefig(plotfname)
    # Return key statistics to caller
    return er, stdr, en, score
