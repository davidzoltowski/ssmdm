import autograd.numpy as np
import autograd.numpy.random as npr

from ssmdm.accumulation import Accumulation, LatentAccumulation

import copy
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")
sns.set_context("talk")
from ssmdm.misc import smooth, generate_clicks, generate_clicks_D
from ssm.util import softplus
from ssm.primitives import blocks_to_full, convert_lds_to_block_tridiag
npr.seed(12345)

# 2D Accumulator with Poisson observations
D = 2 		# number of accumulation dimensions
K = 3 		# number of discrete states
M = 2 		# number of input dimensions
N = 10		# number of observations
bin_size = 0.01
latent_acc = LatentAccumulation(N, K, D, M=M,
								transitions="race",
								emissions="poisson",
								emission_kwargs={"bin_size":bin_size})

# set params
betas = 0.075*np.ones((D,))
sigmas = np.log(1e-3)*np.ones((D,))
latent_acc.dynamics.params = (betas, sigmas, latent_acc.dynamics.params[2])
latent_acc.emissions.Cs[0] = 4 * npr.randn(N,D) + npr.choice([-15,15],(N,D))
latent_acc.emissions.ds[0] = 40 + 4.0 * npr.randn(N)

# Sample state trajectories
T = 100 # number of time bins
trial_time = 1.0 # trial length in seconds
dt = 0.01 # bin size in seconds
N_samples = 250

# input statistics
total_rate = 40 # the sum of the right and left poisson process rates is 40

us = []
zs = []
xs = []
ys = []

for smpl in range(N_samples):

    # randomly draw right and left rates
    rate_r = np.random.randint(0,total_rate+1)
    # rate_r = npr.choice([10,30])
    rate_l = total_rate - rate_r
    rates = [rate_r,rate_l]

    # generate binned right and left clicks
    u = generate_clicks_D(rates,T=trial_time,dt=dt)

    # input is sum of u_r and u_l
    u = 1.0*np.array(u).T
    z, x, y = latent_acc.sample(T, input=u)

    us.append(u)
    zs.append(z)
    xs.append(x)
    ys.append(y)

# initialize test model
test_acc = LatentAccumulation(N, K, D, M=M,
								transitions="race",
								emissions="poisson",
								emission_kwargs={"bin_size":bin_size})
betas0 = 0.02+0.08*npr.rand()*np.ones((D,))
sigmas0 = np.log((4e-5+3.5e-3*npr.rand()))*np.ones((D,))
test_acc.dynamics.params = (betas0, sigmas0, test_acc.dynamics.params[2])

# Initialize C, d
u_sum = np.array([np.sum(u[:,0] - u[:,1]) for u in us])
y_end = np.array([y[-10:] for y in ys])
y_U = y_end[np.where(u_sum>=25)]
y_L = y_end[np.where(u_sum<=-25)]
d_init = (np.mean([y[:5] for y in ys],axis=(0,1)) / bin_size).reshape((1,N))
C_init = np.hstack((np.mean(y_U,axis=(0,1))[:,None],np.mean(y_L,axis=(0,1))[:,None])) / bin_size - d_init.T
test_acc.emissions.ds[0] = d_init
test_acc.emissions.Cs[0] = C_init
test_acc_mf = copy.deepcopy(test_acc)
init_params = copy.deepcopy(test_acc.params)

# fit model
init_var = 1e-2
# q_elbos_lem, q_lem = test_acc.fit(ys, inputs=us, method="laplace_em",
# 						variational_posterior="structured_meanfield",
# 						num_iters=50, alpha=0.5, initialize=False,
# 						variational_posterior_kwargs={"initial_variance":init_var})
#
# # test_acc50 = copy.deepcopy(test_acc)
# # q_lem50 = copy.deepcopy(q_lem)
# q_elbos_lem2, q_lem = test_acc.fit(ys, inputs=us, method="laplace_em",
# 						variational_posterior=q_lem,
# 						num_iters=25, alpha=0.5, initialize=False)

test_acc_lds.params = init_params
q_elbos_lds, q_lds = test_acc_lds.fit(ys, inputs=us, method="bbvi",
						variational_posterior="lds",
						num_iters=100, initialize=False,
						variational_posterior_kwargs={"initial_variance":init_var})
q_elbos_lds2, q_lds = test_acc_lds.fit(ys, inputs=us, method="bbvi",
						variational_posterior=q_lds,
						num_iters=400, initialize=False)
q_elbos_lds3, q_lds = test_acc_lds.fit(ys, inputs=us, method="bbvi",
						variational_posterior=q_lds,
						num_iters=250, initialize=False)

q_elbos_mf, q_mf = test_acc_mf.fit(ys, inputs=us, method="bbvi",
						variational_posterior="mf",
						num_iters=50, initialize=False,
						variational_posterior_kwargs={"initial_variance":init_var},
						kwargs={"step_size":1e-2})
q_elbos_mf3, q_mf = test_acc_mf.fit(ys, inputs=us, method="bbvi",
						variational_posterior=q_mf,
						num_iters=100, initialize=False)
q_elbos_mf6, q_mf = test_acc_mf.fit(ys, inputs=us, method="bbvi",
						variational_posterior=q_mf,
						num_iters=150, initialize=False)


plt.ion()
plt.figure()
# plt.plot(q_elbos_lem[1:], label="vLEM")
plt.plot(np.concatenate((q_elbos_lem[1:], q_elbos_lem2[1:])), label="vLEM")
# plt.plot(q_elbos_mf[1:], label="BBVI")
# plt.plot(np.concatenate((q_elbos_mf[1:], q_elbos_mf2[1:], q_elbos_mf3[1:])), label="BBVI")
plt.plot(np.concatenate((q_elbos_mf[1:], q_elbos_mf2[1:], q_elbos_mf3[1:], q_elbos_mf4[1:], q_elbos_mf5[1:])), label="BBVI")
plt.xlabel("iteration")
plt.ylabel("ELBO")
plt.legend()
plt.tight_layout()
ylim = plt.gcf().gca().get_ylim()

plt.figure()
# plt.plot(q_elbos_lem[1:], label="vLEM")
plt.plot(np.concatenate((q_elbos_lem, q_elbos_lem2[1:])), label="vLEM")
plt.plot(np.concatenate((q_elbos_lds, q_elbos_lds2[1:], q_elbos_lds3[1:])), label="LDS")
# plt.plot(q_elbos_mf[1:], label="BBVI")
# plt.plot(np.concatenate((q_elbos_mf, q_elbos_mf2[1:], q_elbos_mf3[1:])), label="BBVI")
# plt.plot(np.concatenate((q_elbos_mf, q_elbos_mf2[1:], q_elbos_mf3[1:], q_elbos_mf4[1:], q_elbos_mf5[1:], q_elbos_mf6[1:])), label="BBVI")
plt.xlabel("iteration")
plt.ylabel("ELBO")
plt.ylim(ylim)
plt.legend()
plt.tight_layout()

def plot_trial(model, q, posterior, tr=0, legend=False):

	if posterior is "laplace_em":
		q_x = q.mean_continuous_states[tr]
		J_diag = q._params[tr]["J_diag"]
		J_lower_diag= q._params[tr]["J_lower_diag"]
		J = blocks_to_full(J_diag, J_lower_diag)
		Jinv = np.linalg.inv(J)
		q_lem_std = np.sqrt(np.diag(Jinv))
		q_lem_std = q_lem_std.reshape((T,D))
		q_std_1 = q_lem_std[:,0]
		q_std_2 = q_lem_std[:,1]
	elif posterior is "mf":
		q_x = q.mean[tr]
		q_std_1 = np.sqrt(np.exp(q.params[tr][1])[:,0])
		q_std_2 = np.sqrt(np.exp(q.params[tr][1])[:,1])
	elif posterior is "lds":
		q_x = q.mean[tr]
		J_diag, J_lower_diag, h = convert_lds_to_block_tridiag(*q_lds.params[tr])
		J = blocks_to_full(J_diag, J_lower_diag)
		Jinv = np.linalg.inv(J)
		q_lem_std = np.sqrt(np.diag(Jinv))
		q_lem_std = q_lem_std.reshape((T,D))
		q_std_1 = q_lem_std[:,0]
		q_std_2 = q_lem_std[:,1]

	yhat = model.smooth(q_x, ys[tr], input=us[tr])
	zhat = model.most_likely_states(q_x, ys[tr], input=us[tr])

	f, (a0, a1, a2, a3) = plt.subplots(4, 1, gridspec_kw={'height_ratios': [0.5, 0.5, 3, 1]})
	a0.imshow(np.row_stack((zs[tr], zhat)), aspect="auto", vmin=0, vmax=2)
	a0.set_xticks([])
	a0.set_yticks([0, 1], ["$z_{\\mathrm{true}}$", "$z_{\\mathrm{inf}}$"])
	a0.axis("off")
	a2.plot(xs[tr][:,0],color=[1.0,0.0,0.0],label="$x_1$",alpha=0.9)
	a2.plot(xs[tr][:,1],color=[0.0,0.0,1.0],label="$x_2$",alpha=0.9)
	a2.plot(q_x[:,0],color=[1.0,0.3,0.3],linestyle='--',label="$\hat{x}_1$",alpha=0.9)
	a2.plot(q_x[:,1],color=[0.3,0.3,1.0],linestyle='--',label="$\hat{x}_2$",alpha=0.9)

	a2.fill_between(np.arange(T),q_x[:,0]-q_std_1*2.0, q_x[:,0]+q_std_1*2.0, facecolor='r', alpha=0.3)
	a2.fill_between(np.arange(T),q_x[:,1]-q_std_2*2.0, q_x[:,1]+q_std_2*2.0, facecolor='b', alpha=0.3)
	a2.plot(np.array([0,100]),np.array([1,1]),'k--',linewidth=1.0,label=None)
	a2.set_ylim([-0.4,1.4])
	a2.set_xlim([-1,101])
	a2.set_xticks([])
	a2.set_yticks([0,1])
	a2.set_ylabel("x")
	if legend:
		a2.legend()
	sns.despine()
	for n in range(10):
		a3.eventplot(np.where(ys[tr][:,n]>0)[0], linelengths=0.5, lineoffsets=1+n,color='k')
	sns.despine()
	a3.set_yticks([])
	a3.set_xlim([-1,101])

	a1.plot(0.2*us[tr][:,0],color=[1.0,0.5,0.5], label=None,alpha=0.9)
	a1.plot(0.2*us[tr][:,1],color=[0.5,0.5,1.0], label=None,alpha=0.9)
	a1.set_yticks([])
	a1.set_xticks([])
	a1.axes.get_yaxis().set_visible(False)
	plt.tight_layout()

	return

tr = npr.randint(N_samples)
plot_trial(test_acc, q_lem, "laplace_em", tr=tr)
plot_trial(test_acc_lds, q_lds, "lds", tr=tr)
# plot_trial(test_acc50, q_lem50, "laplace_em", tr=tr)
plot_trial(test_acc_mf, q_mf, "mf", tr=tr)

plt.figure(figsize=[6,3])
plt.subplot(121)
plt.imshow(np.concatenate((latent_acc.emissions.Cs[0],test_acc.emissions.Cs[0,:,:]),axis=1),aspect='auto', cmap="inferno")
plt.xticks([0.5, 2.5], ["$C_{\\mathrm{true}}$", "$C_{\\mathrm{inf}}$"])
plt.colorbar()

plt.subplot(122)
plt.imshow(np.concatenate((latent_acc.emissions.ds[0].reshape((N,1)),test_acc.emissions.ds.reshape((N,1))),axis=1),aspect='auto', cmap="inferno")
plt.xticks([0, 1], ["$d_{\\mathrm{true}}$", "$d_{\\mathrm{inf}}$"])
plt.colorbar()
plt.tight_layout()

plt.show()

sim_ys_1 = simulate_accumulator(test_acc, us, num_repeats=3)
# sim_ys_2 = simulate_accumulator(test_acc_mf, us, num_repeats=3)
sim_ys_2 = simulate_accumulator(test_acc_lds, us, num_repeats=3)
true_psths = plot_psths(ys, us, 1, N);
sim_psths_1 = plot_psths(sim_ys_1, us+us+us, 1, N);
sim_psths_2 = plot_psths(sim_ys_2, us+us+us, 1, N);
r2_lem = compute_r2(true_psths, sim_psths_1)
r2_mf = compute_r2(true_psths, sim_psths_2)
psth_list=[true_psths, sim_psths_1, sim_psths_2]
# psth_list=[true_psths, sim_psths_2]
# plot_multiple_psths(psth_list, np.array([0,1,2]))
plot_neurons2 = npr.permutation(np.arange(10))[:3]
plot_multiple_psths(psth_list, plot_neurons2)

plt.gcf().axes[3].set_ylim(plt.gcf().axes[0].get_ylim())
plt.gcf().axes[6].set_ylim(plt.gcf().axes[0].get_ylim())
plt.gcf().axes[4].set_ylim(plt.gcf().axes[1].get_ylim())
plt.gcf().axes[7].set_ylim(plt.gcf().axes[1].get_ylim())
plt.gcf().axes[5].set_ylim(plt.gcf().axes[2].get_ylim())
plt.gcf().axes[8].set_ylim(plt.gcf().axes[2].get_ylim())

# loop through points to get true and inferred "threshold times"
# z_dir = []
# z_t = []
z_dir_mf = []
z_t_mf = []
# true_z_dir = []
# true_z_t = []
for tr in range(N_samples):
	# q_z = q_lem.mean_discrete_states[tr][0]
	# zhat = np.argmax(q_z,axis=1)
	# z_not0 = np.where(zhat > 0)[0]
	# if z_not0.shape[0] > 0:
	# 	z_t.append(z_not0[0])
	# else:
	# 	z_t.append(T+1)
	# z_dir.append(zhat[-1])

	# true_z_dir.append(zs[tr][-1])
	# true_z_not0 = np.where(zs[tr] > 0)[0]
	# if true_z_not0.shape[0] > 0:
	# 	true_z_t.append(true_z_not0[0])
	# else:
	# 	true_z_t.append(T+1)
	#
	# # BBVI
	# zhat = test_acc_mf.most_likely_states(q_mf.mean[tr], ys[tr], input=us[tr])
	zhat = test_acc_lds.most_likely_states(q_lds.mean[tr], ys[tr], input=us[tr])
	z_not0 = np.where(zhat > 0)[0]
	if z_not0.shape[0] > 0:
		z_t_mf.append(z_not0[0])
	else:
		z_t_mf.append(T+1)
	z_dir_mf.append(zhat[-1])

plt.figure()
plt.plot(np.array([0,100]), np.array([0,100]), 'k--')
plt.plot(np.array(true_z_t), np.array(z_t), '.', color=[0.3,0.3,1.0], alpha=0.75, label="vLEM") #markeredgecolor='w', linewidth=0.5, markersize=10)
plt.plot(np.array(true_z_t), np.array(z_t_mf), '.', color=[1.0,0.3,0.3], alpha=0.75, label="BBVI") #, markeredgecolor='w', linewidth=0.5, markersize=10)
# plt.plot(np.array(true_z_t), np.array(z_t2), 'r.')
plt.xlabel("true state switch time")
plt.ylabel("inferred state switch time")
plt.legend()
plt.axis("equal")
plt.tight_layout()

print(np.sum(np.array(true_z_dir) == np.array(z_dir)))
print(np.sum(np.array(true_z_dir) == np.array(z_dir_mf)))

bins = np.arange(-100,55,5)
plt.figure()
data = np.array(true_z_t) - np.array(z_t)
counts, bins = np.histogram(data, bins)
plt.hist(bins[:-1], bins, weights=counts / np.sum(counts), color=[0.3,0.3,1.0], alpha=0.75, label="vLEM")
data = np.array(true_z_t) - np.array(z_t_mf)
counts, bins = np.histogram(data, bins)
plt.hist(bins[:-1], bins, weights=counts / np.sum(counts), color=[1.0,0.3,0.3], alpha=0.75, label="BBVI")
plt.xlabel("true - inferred state switch times")
plt.ylabel("frequency")
plt.legend()
plt.tight_layout()
