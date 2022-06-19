import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

import itertools
from scipy.stats import multivariate_normal

def generate_clicks(T=1.0,dt=0.01,rate_r=20,rate_l=20):
    """
    This function generates right and left 'clicks' from two Poisson processes with rates rate_r and rate_l
    over T seconds with bin sizes dt. The outputs are binned clicks into discrete time bins.
    """

    # number of clicks
    num_r = npr.poisson(rate_r*T)
    num_l = npr.poisson(rate_l*T)

    # click times
    click_time_r = np.sort(npr.uniform(low=0.0,high=T,size=[num_r,1]))
    click_time_l = np.sort(npr.uniform(low=0.0,high=T,size=[num_l,1]))

    # binned outputs are arrays with dimensions Tx1
    binned_r = np.histogram(click_time_r,np.arange(0.0,T+dt,dt))[0]
    binned_l = np.histogram(click_time_l,np.arange(0.0,T+dt,dt))[0]

    return binned_r, binned_l

def generate_clicks_D(rates,T=1.0,dt=0.01):
    """
    This function generates 'clicks' from D Poisson processes with rates specified in rates
    over T seconds with bin sizes dt. The outputs are binned clicks into discrete time bins.

    rates is a list of D rates.
    """

    # number of clicks
    num_clicks = [npr.poisson(rate*T) for rate in rates]

    # click times
    click_times = [np.sort(npr.uniform(low=0.0,high=T,size=[num_click,1]))
                   for num_click in num_clicks]

    # binned outputs are arrays with dimensions Tx1
    binned_clicks = [np.histogram(click_time,np.arange(0.0,T+dt,dt))[0]
                     for click_time in click_times]

    return binned_clicks

def factor_analysis(D, ys, num_iters=15):
    # D is number of latent dimensions
    # ys is list of data points

    # concatenate ys
    all_y = np.array(list(itertools.chain(*ys)))

    # observation dimensions
    Nobs,N = np.shape(all_y)

    # compute mean across column
    mu_y = np.mean(all_y,axis=0,keepdims=True)

    # subtract mean
    ys_zero = all_y - mu_y

    # initialize C, Psi
    Cfa = np.random.randn(N,D)
    psi = np.eye(N)

    # run EM
    pbar = trange(num_iters)
    lls = []
    for i in pbar:

        # E-step
        lamb = np.linalg.inv(Cfa.T@np.linalg.inv(psi)@Cfa + np.eye(D))
        mu_x = (lamb@Cfa.T@np.linalg.inv(psi)@(ys_zero.T)).T
        mu_xxT = [lamb + np.outer(mu_x[i,:],mu_x[i,:]) for i in range(Nobs)]

        # M-step
        Cfa = ( np.linalg.inv((mu_x.T@mu_x) + Nobs * lamb) @ mu_x.T @ ys_zero ).T
        np.fill_diagonal(psi,  np.diag( (1.0 / Nobs) * (ys_zero.T @ ys_zero - ys_zero.T @ mu_x @ Cfa.T)))

        # add small elements to diagonal of psi for stability (TODO: add condition)
        np.fill_diagonal(psi,  np.diag(psi +1e-7*np.eye(N)))

        # compute log likelihood
        log_py = np.sum(multivariate_normal.logpdf(all_y, mean=mu_y[0,:], cov=(Cfa@Cfa.T + psi)))
        lls += [log_py]

        pbar.set_description("Itr {} LP: {:.1f}".format(i, lls[-1]))
        pbar.update(1)

    # get xhats
    my_xhats = [(lamb@Cfa.T@np.linalg.inv(psi)@(y - mu_y[0,:]).T).T for y in ys]

    return Cfa, my_xhats, lls , psi

def smooth(xs, window_size=5):
    # window size is number of bins on each side*2, +1

    T,N = np.shape(xs)
    x_smooth = np.zeros(np.shape(xs))

    # win is number of bins on each side
    win = int( (window_size - 1) / 2 )

    for t in range(T):
        smooth_window = np.arange(np.maximum(t-win,0),np.minimum(t+win,T-1))
        x_smooth[t,:] = np.mean(xs[smooth_window,:],axis=0)

    return x_smooth

def simulate_accumulator(model, inputs, num_repeats=1):
    # this function takes in a fit model and inputs, and simulates data ys
    N = len(inputs)
    ys = []
    for r in range(num_repeats):
        for n in range(N):
            T = inputs[n].shape[0]
            z, x, y = model.sample(T, inputs[n])
            ys.append(y)

    return ys

def compute_psths(ys, inputs, num_partitions=3, num_bins_smooth=3):
    # num_partitions - number of partitions above and below 0

    # get time bins, number of neurons
    T, N = ys[0].shape

    # this function plots the input-conditioned PSTH of a neuron
    # assert np.shape(inputs)[2] == 1 or np.shape(inputs)[2] == 2
    if inputs[0].shape[1] == 1:
        u_sums = np.array([np.sum(u) for u in inputs])
    elif inputs[0].shape[1] == 2:
        u_sums = np.array([np.sum(u[:,0] - u[:,1]) for u in inputs])

    # get sorting index
    # import ipdb; ipdb.set_trace()
    idx_sort = np.argsort(u_sums)
    u_sorted = u_sums[idx_sort]
    u_below_0 = np.where(u_sorted<0)[0][-1]
    u_above_0 = np.where(u_sorted>0)[0][0]
    u_0 = np.where(np.abs(u_sorted)<1e-3)[0]

    idx_below_0 = np.array_split(idx_sort[:u_below_0],num_partitions)
    idx_0 = np.copy(idx_sort[u_0]) if u_0.shape[0] > 0 else np.array([])
    idx_above_0 = np.array_split(idx_sort[u_above_0:],num_partitions)

    # compute below 0 psths (assumes same length!)
    bin_size = 0.01
    all_idx = idx_below_0 + [idx_0] + idx_above_0
    y_psths = []
    for idx in all_idx:
        # import ipdb
        # ipdb.set_trace()
        if idx.shape[0] > 0:
            y_idx = [ys[i] for i in idx]
            # y_psths.append(np.mean(y_idx,axis=0) / bin_size)
            y_psths.append(mean_diff_dims(y_idx) / bin_size)
        else:
            y_psths.append(np.zeros((0,N)))

    # rearrange to be neuron by psth
    neuron_psths = [[psth[:,n] for psth in y_psths] for n in range(N)]
    if num_bins_smooth > 1:
        neuron_psths = [[smooth(row[:,None],num_bins_smooth) for row in psth] for psth in neuron_psths]

    return neuron_psths

def mean_diff_dims(xs, min_counts=5):
    # if xs consists of Ti by N arrays, with Ti \neq Tj,
    # compute the mean across the first dimension
    N = xs[0].shape[1]
    T_max = max([x.shape[0] for x in xs])
    counts = np.zeros((T_max,N))
    means = np.zeros((T_max,N))
    for x in xs:
        Ti = x.shape[0]
        means[:Ti,:] += x
        counts[:Ti,:] += 1.0
    
    # min counts for returning mean 
    out = np.divide(means, counts)
    out[counts<min_counts] = np.nan
    return out

def plot_psths(ys, inputs, num_row, num_col, fig=None,linestyle='-', ylim=None):
    if fig is None:
        plt.figure()

    # get time bins, number of neurons
    T, N = ys[0].shape
    neuron_psths = compute_psths(ys, inputs)

    # plot
    colors = [[1.0,0.0,0.0], [1.0,0.3,0.3], [1.0,0.6,0.6],
                'k', [0.6,0.6,1.0], [0.3,0.3,1.0], [0.0,0.0,1.0]]
    for n in range(N):
        plt.subplot(num_row,num_col,n+1)
        for coh in range(len(neuron_psths[n])):
            plt.plot(neuron_psths[n][coh],color=colors[coh],linestyle=linestyle,alpha=0.9, linewidth=1)
            if ylim is not None:
                plt.ylim(ylim)
    return 

def plot_neuron_psth(neuron_psth, linestyle='-', ylim=None, flip_colors=False, colors=None):
    # for 3 coherences on each side, plus zero
    if colors is None:
        colors = [[1.0,0.0,0.0], [1.0,0.3,0.3], [1.0,0.6,0.6],
                    'k', [0.6,0.6,1.0], [0.3,0.3,1.0], [0.0,0.0,1.0]]
    if flip_colors:
        colors.reverse()
    for coh in range(len(neuron_psth)):
        # if coh != 3:
        plt.plot(neuron_psth[coh], color=colors[coh], linestyle=linestyle, alpha=0.9)
        if ylim is not None:
            plt.ylim(ylim)

    return

# compute R^2
# use 10ms time bins
def compute_r2(true_psths, sim_psths):

    # get number of neurons
    assert len(true_psths) == len(sim_psths)
    N = len(true_psths)

    r2 = np.zeros(N)

    for i in range(N):

        true_psth = true_psths[i]
        sim_psth = sim_psths[i]
        true_psth_mean = [true_psth[coh] for coh in range(len(true_psth)) if true_psth[coh].shape[0] > 0]
        mean_PSTH = np.nanmean(np.vstack(true_psth_mean))

        r2_num = 0.0
        r2_den = 0.0

        # number of coherences, loop over
        NC = len(true_psth)
        for j in range(NC):
            T = true_psth[j].shape[0]
            if T > 0:
                r2_num += np.nansum( (true_psth[j] - sim_psth[j][:T])**2)
                r2_den += np.nansum( (mean_PSTH - true_psth[j])**2)

        r2[i] = 1 - r2_num / r2_den

    return r2

def compute_mad(true_psths, sim_psths):

    # get number of neurons
    assert len(true_psths) == len(sim_psths)
    N = len(true_psths)

    mads = np.zeros(N)

    for i in range(N):

        true_psth = true_psths[i]
        sim_psth = sim_psths[i]

        mads_n = 0.0

        # number of coherences, loop over
        NC = len(true_psth)
        for j in range(NC):
            T = true_psth[j].shape[0]
            if T > 0:
                mads_n += np.nansum( np.abs(true_psth[j] - sim_psth[j][:T]) )

        mads[i] = mads_n

    return mads

def plot_multiple_psths(psth_list, neuron_idx=None):
    # takes as input a list of (list of) PSTHs
    # each PSTH is a list of PSTHs for different neurons from the same model
    # plotting is row (neuron) by model
    num_models = len(psth_list)
    if neuron_idx is None:
        neuron_idx = np.arange(0,len(psth_list[0]))
    num_neurons = neuron_idx.shape[0]

    plt.figure()
    for i in range(num_models):
        for j in range(num_neurons):
            plt.subplot(num_neurons, num_models, (j)*num_models + i + 1)
            plot_neuron_psth(psth_list[i][neuron_idx[j]])

    return

def plot_psth_grid(psths, num_row=1, num_col=1, ylim=None, flip_colors=False):
    # takes as input a list of (list of) PSTHs
    # each PSTH is a list of PSTHs for different neurons from the same model
    # plotting is row (neuron) by model
    if np.shape(flip_colors) == ():
        flip_colors = [flip_colors for i in range(num_row*num_col)]
    num_neurons = len(psths)
    plt.figure()
    for i in range(num_neurons):
        plt.subplot(num_row,num_col,i+1)
        plot_neuron_psth(psths[i], ylim=ylim, flip_colors=flip_colors[i])

    return