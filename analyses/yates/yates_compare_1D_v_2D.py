import autograd.numpy as np
import autograd.numpy.random as npr

from ssmdm.accumulation import Accumulation, LatentAccumulation
from ssm.util import softplus
from ssmdm.yates_helper_functions import *

import copy
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
sns.set_context("talk")
from ssmdm.misc import smooth, generate_clicks, generate_clicks_D
npr.seed(13)

# import data
from scipy.io import loadmat
# d = loadmat("jake_ssm_sess_n20150312b.mat") # bin_size, us, us2, ys, dprimes
d = loadmat("jake_ssm_sess_n20150326a.mat") # 6 neurons, 2D? bin_size, us, us2, ys, dprimes
# d = loadmat("jake_ssm_sess_p20121104.mat") # 2 neurons
scale_u = 5.0
u1s = [u[:,None] / scale_u for u in d["us"]]
u2s = [u / scale_u for u in d["us2"]]
for i in range(len(u2s)):
	u2s[i][:,1] *= -1.0
ys = [y.astype("int") for y in d["ys"]]
bin_size = d["bin_size"][0][0]
dprime = d["dprimes"][0]
choices = d["cho"][:,0]

# data for fitting
num_trials = len(ys)
ys = [y[:,[3, 4, 5, 6, 7, 8]] for y in ys[:num_trials]]
N = ys[0].shape[1]

# initialize 1D model
K = 3
D = 1
M = 1
latent_acc_1D = LatentAccumulation(N, K, D, M=M,
							  transitions="ddmhard",
							  dynamics_kwargs={"learn_A":False},
							  emissions="poisson",
							  emission_kwargs={"bin_size":0.01})
# latent_acc_1D.emissions.ds = (np.mean([y[0:3] for y in ys],axis=(0,1)) / bin_size).reshape((1,N))

u_sum = np.array([np.sum(u) for u in u1s])
y_end = np.array([y[-3:] for y in ys])
# y_U = y_end[np.where(u_sum>=20)]
# y_L = y_end[np.where(u_sum<=-20)]
y_U = y_end[np.where(choices==1)]
y_L = y_end[np.where(choices==0)]
C_init = (1.0/2.05)*np.mean((np.mean(y_U,axis=0) - np.mean(y_L,axis=0)),axis=0) / bin_size

# latent_acc_1D.emissions.Cs = C_init.reshape((1,N,D))

latent_acc_1D.initialize(ys, inputs=u1s)
init_params = copy.deepcopy(latent_acc_1D.params)

# fit 1D model
init_var = 1e-5
_, q_lem_1D = latent_acc_1D.fit(ys, inputs=u1s, method="laplace_em",
						variational_posterior="structured_meanfield",
						num_iters=0, initialize=False,
						variational_posterior_kwargs={"initial_variance":init_var})
q_elbos1, q_lem_1D = latent_acc_1D.fit(ys, inputs=u1s, method="laplace_em",
							  variational_posterior=q_lem_1D,
							  num_iters=25, alpha=0.75, initialize=False)#,
q_elbos1_2, q_lem_1D = latent_acc_1D.fit(ys, inputs=u1s, method="laplace_em",
							  variational_posterior=q_lem_1D,
							  num_iters=15, alpha=0.75, initialize=False)#,

# initialize 2D model
K = 3
D = 2
M = 2
latent_acc_2D = LatentAccumulation(N, K, D, M=M,
							  transitions="racehard",
							  dynamics_kwargs={"learn_A":False},
							  emissions="poisson",
							  emission_kwargs={"bin_size":0.01})
# latent_acc_2D.emissions.ds = (np.mean([y[0:3] for y in ys],axis=(0,1)) / bin_size).reshape((1,N))
latent_acc_2D.initialize(ys, inputs=u2s)

u_sum = np.array([np.sum(u) for u in u1s])
y_end = np.array([y[-3:] for y in ys])
# y_U = y_end[np.where(u_sum>=20)]
# y_L = y_end[np.where(u_sum<=-20)]
y_U = y_end[np.where(choices==1)]
y_L = y_end[np.where(choices==0)]
C_init = np.hstack((np.mean(y_U,axis=(0,1))[:,None],np.mean(y_L,axis=(0,1))[:,None])) / bin_size - latent_acc_2D.emissions.ds.T

# latent_acc_2D.emissions.Cs = C_init.reshape((1,N,D))

init_params = copy.deepcopy(latent_acc_2D.params)

# fit 2D model
init_var = 1e-5
_, q_lem_2D = latent_acc_2D.fit(ys, inputs=u2s, method="laplace_em",
						variational_posterior="structured_meanfield",
						num_iters=0, initialize=False,
						variational_posterior_kwargs={"initial_variance":init_var})
q_elbos2, q_lem_2D = latent_acc_2D.fit(ys, inputs=u2s, method="laplace_em",
							  variational_posterior=q_lem_2D,
							  num_iters=25, alpha=0.75, initialize=False)#,s
q_elbos2_2, q_lem_2D = latent_acc_2D.fit(ys, inputs=u2s, method="laplace_em",
							  variational_posterior=q_lem_2D,
							  num_iters=10, alpha=0.75, initialize=False)#,s
q_elbos2_3, q_lem_2D = latent_acc_2D.fit(ys, inputs=u2s, method="laplace_em",
							  variational_posterior=q_lem_2D,
							  num_iters=15, alpha=0.75, initialize=False)#,s

plt.ion()

plt.figure()
plt.plot(range(1,len(q_elbos1)),q_elbos1[1:],label="1D")
plt.plot(range(1,len(q_elbos2)),q_elbos2[1:],label="2D")
plt.xlabel("iteration")
plt.ylabel("ELBO")
plt.legend()
plt.tight_layout()

# tr = 0
def plot_sim_results(model, q, us, tr=0):

    xlim = (0, np.shape(ys[tr])[0])
    q_x = q.mean_continuous_states[tr]
    yhat = model.smooth(q_x, ys[tr], input=us[tr])
    plt.ion()
    plt.figure(figsize=[12,6])
    plt.subplot(231)
    if us[tr].shape[1] == 1:
	    plt.plot(us[tr], 'k', label="input")
	    plt.plot(q_x[:,0],'b', label="inferred")
    else:
	    plt.plot(us[tr][:,0] * 1.0,'c', label="input")
	    plt.plot(us[tr][:,1] * 1.0,'m', label=None)
	    plt.plot(q_x[:,0],'b', label="inferred")
	    plt.plot(q_x[:,1],'r')
    plt.xlim(xlim)
    plt.xlabel('time bin')
    plt.ylabel('x')
    plt.legend()

    if np.shape(yhat)[1] < 7:
	    plt.subplot(232)
	    # plt.plot(latent_acc.emissions.mean(np.dot(xs[tr],C.T)+d),'k')
	    plt.plot(smooth(ys[tr],20) / bin_size, 'k')
	    plt.plot(yhat / bin_size,'r--')
	    plt.xlim(xlim)
	    plt.xlabel('time bin')
	    plt.ylabel('y (observations)')

    else:
	    plt.subplot(232)
	    true_y = smooth(ys[tr],20) / bin_size
	    smooth_y = yhat / bin_size
	    lim = max(true_y.max(), smooth_y.max())
	    plt.imshow(true_y.T,aspect="auto", vmin=0, vmax=lim)
	    plt.colorbar()
	    plt.subplot(233)
	    plt.imshow(smooth_y.T,aspect="auto", vmin=0, vmax=lim)
	    plt.colorbar()

    zhat = model.most_likely_states(q_x, ys[tr], input=us[tr])
    plt.subplot(234)
    plt.imshow(zhat[None,:], aspect="auto")
    plt.plot(xlim, [0.5, 0.5], '-k', lw=2)
    plt.xlim(xlim)
    plt.xlabel('time bin')

    plt.subplot(235)
    plt.imshow(model.emissions.Cs[0,:,:],aspect='auto', cmap="inferno")
    plt.xticks([0.5], ["$C_{\\mathrm{inf}}$"])
    plt.colorbar()

    plt.subplot(236)
    plt.imshow(model.emissions.ds.reshape((N,1)),aspect='auto', cmap="inferno")
    plt.xticks([0], ["$d_{\\mathrm{inf}}$"])
    plt.colorbar()
    plt.tight_layout()

plot_sim_results(latent_acc_1D, q_lem_1D, u1s, 0)
plot_sim_results(latent_acc_2D, q_lem_2D, u2s, 0)

# predict choices
pred_choice_1D = np.zeros(num_trials)
pred_choice_2D = np.zeros(num_trials)
for i, (q_x_1, q_x_2) in enumerate(zip(q_lem_1D.mean_continuous_states, q_lem_2D.mean_continuous_states)):
# for i, q_x_2 in enumerate(q_lem_2D.mean_continuous_states):
# for i, q_x_1 in enumerate(q_lem_1D.mean_continuous_states):

	# pred 1D
	q_end = q_x_1[-1]
	if q_end[0] > 0.0:
		pred_choice_1D[i] = 1.0
	else:
		pred_choice_1D[i] = 0.0

	# pred 2D
	q_end = q_x_2[-1]
	if q_end[0] > q_end[1]:
		pred_choice_2D[i] = 1.0
	else:
		pred_choice_2D[i] = 0.0

print("Accuracy 1D: ", np.where(pred_choice_1D == choices)[0].shape[0] / num_trials)
print("Accuracy 2D: ", np.where(pred_choice_2D == choices)[0].shape[0] / num_trials)

tr = 0
plt.close()
tr += 1
# plot_sim_results(tr)
# plot_sim_results(latent_acc_2D, q_lem_2D, u2s, tr)
plot_sim_results(latent_acc_1D, q_lem_1D, u1s, tr)
print("Trial: ", tr)
# print("Predicted choice: ", pred_choice_2D[tr])
print("Predicted choice: ", pred_choice_1D[tr])
print("True choice: ", choices[tr])


# baseline -> u above or below 0
baseline_choice = np.zeros(num_trials)
for i, u in enumerate(u2s):
	u_sum = np.sum(u[:,0] - u[:,1])
	if u_sum > 1e-3:
		baseline_choice[i] = 1.0
	elif np.abs(u_sum) <= 1e-3:
		baseline_choice[i] = 1.0 if npr.rand() < 0.5 else 0.0
	else:
		baseline_choice[i] = 0.0

print("Baseline accuracy: ", np.where(baseline_choice == choices)[0].shape[0] / num_trials)

# for i in range(6):
     # ...:      plt.gcf().axes[i].set_ylim(fig.axes[i].get_ylim())/

# simulate data
sim_ys_1 = simulate_accumulator(latent_acc_1D, u1s, num_repeats=3)
sim_ys_2 = simulate_accumulator(latent_acc_2D, u2s, num_repeats=3)
true_psths = plot_psths(ys, u1s, 1, N);
sim_psths_1 = plot_psths(sim_ys_1, u1s+u1s+u1s, 1, N);
sim_psths_2 = plot_psths(sim_ys_2, u2s+u2s+u2s, 1, N);
r2_1D = compute_r2(true_psths, sim_psths_1)
r2_2D = compute_r2(true_psths, sim_psths_2)
psth_list=[true_psths, sim_psths_1, sim_psths_2]
# plot_multiple_psths(psth_list, np.array([0,2,3])) # second arg could be np.array([0,2,3])
plot_multiple_psths(psth_list) # second arg could be np.array([0,2,3])

plt.gcf().axes[3].set_ylim(plt.gcf().axes[0].get_ylim())
plt.gcf().axes[4].set_ylim(plt.gcf().axes[1].get_ylim())
plt.gcf().axes[5].set_ylim([10,100])
plt.gcf().axes[6].set_ylim(plt.gcf().axes[0].get_ylim())
plt.gcf().axes[7].set_ylim(plt.gcf().axes[1].get_ylim())
plt.gcf().axes[8].set_ylim([10,100])
plt.gcf().axes[2].set_ylim([10,100])
