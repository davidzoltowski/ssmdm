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
from ssm.primitives import blocks_to_full
npr.seed(12345)

# 2D Accumulator with Poisson observations
D = 2 		# number of accumulation dimensions
K = 3 		# number of discrete states
M = 2 		# number of input dimensions
N = 15		# number of observations
bin_size = 0.01
latent_acc = LatentAccumulation(N, K, D, M=M,
								transitions="race",
								emissions="poisson",
								emission_kwargs={"bin_size":bin_size})

# set params
betas = 0.05*np.ones((D,))
sigmas = np.log(1e-3)*np.ones((D,))
latent_acc.dynamics.params = (betas, sigmas, latent_acc.dynamics.params[2])
latent_acc.emissions.Cs[0] = 4 * npr.randn(N,D) + npr.choice([-15,15],(N,D))
latent_acc.emissions.ds[0] = 40 + 4.0 * npr.randn(N)

# Sample state trajectories
T = 100 # number of time bins
trial_time = 1.0 # trial length in seconds
dt = 0.01 # bin size in seconds
N_samples = 100

# input statistics
total_rate = 40 # the sum of the right and left poisson process rates is 40

us = []
zs = []
xs = []
ys = []

for smpl in range(N_samples):

    # randomly draw right and left rates
    rate_r = np.random.randint(0,total_rate+1)
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
betas0 = 0.02+0.08*np.ones((D,))
sigmas0 = np.log((4e-5+3.5e-3*npr.rand()))*np.ones((D,))
test_acc.dynamics.params = (betas0, sigmas0, test_acc.dynamics.params[2])

# Initialize C, d
u_sum = np.array([np.sum(u[:,0] - u[:,1]) for u in us])
y_end = np.array([y[-10:] for y in ys])
y_U = y_end[np.where(u_sum>=25)]
y_L = y_end[np.where(u_sum<=-25)]
C_init = np.hstack((np.mean(y_U,axis=(0,1))[:,None],np.mean(y_L,axis=(0,1))[:,None])) / bin_size - test_acc.emissions.ds.T

test_acc.emissions.ds = (np.mean([y[0:3] for y in ys],axis=(0,1)) / bin_size).reshape((1,N))
test_acc.emissions.Cs[0] = C_init

# fit model
init_var = 1e-4
q_elbos, q_lem = test_acc.fit(ys, inputs=us, method="laplace_em",
						variational_posterior="structured_meanfield",
						num_iters=25, alpha=0.5, initialize=False,
						variational_posterior_kwargs={"initial_variance":1e-4})

plt.figure()
plt.plot(q_elbos[1:])
plt.xlabel("iteration")
plt.ylabel("ELBO")
plt.tight_layout()

def plot_trial(tr=0,legend=False):

	q_x = q_lem.mean_continuous_states[tr]
	zhat = test_acc.most_likely_states(q_x, ys[tr], input=us[tr])
	yhat = test_acc.smooth(q_x, ys[tr], input=us[tr])
	zhat = test_acc.most_likely_states(q_x, ys[tr], input=us[tr])

	J_diag = q_lem._params[tr]["J_diag"]
	J_lower_diag= q_lem._params[tr]["J_lower_diag"]
	J = blocks_to_full(J_diag, J_lower_diag)
	Jinv = np.linalg.inv(J)
	q_lem_std = np.sqrt(np.diag(Jinv))
	q_lem_std = q_lem_std.reshape((T,D))
	q_std_1 = q_lem_std[:,0]
	q_std_2 = q_lem_std[:,1]
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
plot_trial(tr=tr)

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
