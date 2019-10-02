import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_context("talk")
sns.set_style("white")

from ssmdm.accumulation import LatentAccumulation, Accumulation
from ssmdm.misc import smooth

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

def plot_psths(ys, inputs, num_row, num_col, fig=None,linestyle='-'):
    if fig is None:
        plt.figure()
    # get time bins, number of neurons
    T, N = ys[0].shape

    # number of partitions above and below 0
    num_partitions = 3

    # this function plots the input-conditioned PSTH of a neuron
    assert np.shape(inputs)[2] == 1 or np.shape(inputs)[2] == 2
    if np.shape(inputs)[2] == 1:
        u_sums = np.array([np.sum(u) for u in inputs])
    elif np.shape(inputs)[2] == 2:
        u_sums = np.array([np.sum(u[:,0] - u[:,1]) for u in inputs])

    # split inputs into thirds above zero and thirds below zero
    # plot "zero" as black, its own category if it exists
    # above zero -> blue
    # below zero -> red

    # get sorting index
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
            y_psths.append(np.mean(y_idx,axis=0) / bin_size)
        else:
            y_psths.append(np.zeros((0,N)))

    # rearrange to be neuron by psth
    neuron_psths = [[psth[:,n] for psth in y_psths] for n in range(N)]
    smoothed_psths = [[smooth(row[:,None],10) for row in psth] for psth in neuron_psths]
    # plot
    colors = [[1.0,0.0,0.0], [1.0,0.3,0.3], [1.0,0.6,0.6],
                'k', [0.6,0.6,1.0], [0.3,0.3,1.0], [0.0,0.0,1.0]]
    # colors = [[1.0,0.0,0.0], [1.0,0.2,0.2], [1.0,0.4,0.4], [1.0,0.6,0.6],
    #             'k', [0.6,0.6,1.0], [0.4,0.4,1.0], [0.2,0.2,1.0], [0.0,0.0,1.0]]
    for n in range(N):
        plt.subplot(num_row,num_col,n+1)
        for coh in range(len(neuron_psths[n])):
            # import ipdb
            # ipdb.set_trace()
            # plt.plot(smooth(neuron_psths[n][coh][:,None],10),color=colors[coh],linestyle=linestyle,alpha=0.9, linewidth=1)
            plt.plot(smoothed_psths[n][coh],color=colors[coh],linestyle=linestyle,alpha=0.9, linewidth=1)
            # plt.plot(neuron_psths[n][coh],color=colors[coh],linestyle=linestyle,alpha=0.9, linewidth=1)
        # plt.subplots_adjust(wspace=0, hspace=0)

    return smoothed_psths

def plot_neuron_psth(neuron_psth, linestyle='-'):
    # for 3 coherences on each side, plus zero
    colors = [[1.0,0.0,0.0], [1.0,0.3,0.3], [1.0,0.6,0.6],
                'k', [0.6,0.6,1.0], [0.3,0.3,1.0], [0.0,0.0,1.0]]
    for coh in range(len(neuron_psth)):
        plt.plot(neuron_psth[coh], color=colors[coh], linestyle=linestyle, alpha=0.9)

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
        mean_PSTH = np.mean(true_psth_mean)

        r2_num = 0.0
        r2_den = 0.0

        # number of coherences, loop over
        NC = len(true_psth)
        for j in range(NC):
            T = true_psth[j].shape[0]
            if T > 0:
                r2_num += np.sum( (true_psth[j] - sim_psth[j])**2)
                r2_den += np.sum( (mean_PSTH - true_psth[j])**2)
            # for t in range(T):
                # r2_num += (true_psth[j][t] - sim_psth[j][t])**2
                # r2_den += (mean_PSTH - true_psth[j][t])**2

        r2[i] = 1 - r2_num / r2_den

    return r2

def plot_multiple_psths(psth_list, neuron_idx=None):
    # takes as input a list of (list of) PSTHs
    # each PSTH is a list of PSTHs for different neurons from the same model
    # plotting is row (neuron) by model
    num_models = len(psth_list)
    if neuron_idx is None:
        neuron_idx = np.arange(0,len(psth_list[0]))
    num_neurons = neuron_idx.shape[0]
    # num_neurons = len(psth_list[0])

    plt.figure()
    for i in range(num_models):
        for j in range(num_neurons):
            plt.subplot(num_neurons, num_models, (j)*num_models + i + 1)
            plot_neuron_psth(psth_list[i][neuron_idx[j]])

    return
