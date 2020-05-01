import autograd.numpy as np
import autograd.numpy.random as npr

from ssmdm.accumulation import Accumulation, LatentAccumulation

import copy
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")
sns.set_context("talk")
from ssmdm.misc import smooth, generate_clicks, generate_clicks_D
from ssm.primitives import blocks_to_full
from ssm.util import softplus, logistic
npr.seed(2)

# 1D Accumulator with Poisson observations
D = 1 		# number of accumulation dimensions
K = 3 		# number of discrete states
M = 3 		# number of input dimensions
N = 10		# number of observations
bin_size = 0.01
latent_acc = LatentAccumulation(N, K, D, M=M,
								transitions="ddmnlncollapsing",
								dynamics_kwargs={"learn_V":False},
								emissions="poisson",
								emission_kwargs={"bin_size":bin_size})

latent_acc.dynamics._log_sigmasq[0] = np.log(2e-3)*np.ones((D,))
latent_acc.dynamics.Vs[0] = np.array([0.05,0,0])
latent_acc.emissions.Cs[0] = 4 * npr.randn(N,D) + npr.choice([-10,10],(N,D))
latent_acc.emissions.ds[0] = 30 + 5.0 * npr.randn(N)

# Sample state trajectories
T = 100 # number of time bins
trial_time = 1.0 # trial length in seconds
dt = 0.01 # bin size in seconds
N_samples = 200

# input statistics
total_rate = 40 # the sum of the right and left poisson process rates is 40

us = []
zs = []
xs = []
ys = []
cs = []

for smpl in range(N_samples):

    # randomly draw right and left rates
    rate_r = np.random.randint(0,total_rate+1)
    rate_l = total_rate - rate_r
    rates = [rate_r,rate_l]

    # generate binned right and left clicks
    u = generate_clicks_D(rates,T=trial_time,dt=dt)

    # input is sum of u_r and u_l
    u = (1.0*np.array(u[1] - u[0]).T).reshape((T,1))
    u_T = np.arange(0,T)[:,None] * np.ones((1,2)) * 1.0

    u = np.hstack((u, u_T))
    z, x, y = latent_acc.sample(T, input=u)

    us.append(u)
    zs.append(z)
    xs.append(x)
    ys.append(y)

def bound_func(t, a, ap, lamb, k):
	return a - (1 - np.exp(-(t / lamb)**k)) * (0.0 * a + np.exp(ap))

plt.ion()
plt.figure()
for tr in range(15):
	plt.plot(xs[tr],'k',alpha=0.6)
ub = bound_func(np.arange(T), 1.0, latent_acc.transitions.ap, latent_acc.transitions.lamb, 3.0)
plt.plot(np.arange(T), ub, 'b--')
plt.plot(np.arange(T), -1.0 * ub, 'r--')
plt.xlabel("time")
plt.ylabel("x")
sns.despine()
plt.tight_layout()

# fit SLDS model to ys
# initialize
test_acc = LatentAccumulation(N, K, D, M=M, transitions="ddmnlncollapsing", dynamics_kwargs={"learn_V":False, "learn_A":False},
							  emissions="poisson", emission_kwargs={"bin_size":bin_size})
betas = np.array([0.0 + 0.08*npr.rand()])
sigmas = np.log(5e-4+2.5e-3*npr.rand())*np.ones((D,))
test_acc.dynamics.params = (betas, sigmas) + test_acc.dynamics.params[2:]
test_acc.initialize(ys, inputs=us)
init_params = copy.deepcopy(test_acc.params)

# fit
q_elbos, q_lem = test_acc.fit(ys, inputs=us, method="laplace_em",
							  variational_posterior="structured_meanfield",
							  num_iters=50, alpha=0.5, initialize=False)


plt.ion()
plt.figure()
plt.plot(q_elbos[1:])
plt.xlabel("iteration")
plt.ylabel("ELBO")
plt.tight_layout()

plt.figure()
plt.imshow(np.concatenate((latent_acc.emissions.Cs[0,:,:],test_acc.emissions.Cs[0,:,:]),axis=1),aspect='auto')
plt.xticks([0.5, 2.5], ["$C_{\\mathrm{true}}$", "$C_{\\mathrm{FA+Init}}$"])
plt.colorbar()

def plot_trial(model, q, posterior, tr=0,legend=False):

	if posterior is "laplace_em":
		q_x = q.mean_continuous_states[tr]
		q_std = np.sqrt(q._continuous_expectations[tr][2][:,0,0])
	elif posterior is "mf":
		q_x = q_mf.mean[tr]
		# q_std = np.sqrt(np.exp(q_mf.params[tr][1])[:,0])

	yhat = model.smooth(q_x, ys[tr], input=us[tr])
	zhat = model.most_likely_states(q_x, ys[tr], input=us[tr])

	f, (a0, a1, a2, a3) = plt.subplots(4, 1, gridspec_kw={'height_ratios': [0.5, 0.5, 3, 1]})
	# f, (a0, a1, a2) = plt.subplots(3, 1, gridspec_kw={'height_ratios': [0.5, 0.5, 3]})
	a0.imshow(np.row_stack((zs[tr], zhat)), aspect="auto", vmin=0, vmax=2)
	a0.set_xticks([])
	a0.set_yticks([0, 1], ["$z_{\\mathrm{true}}$", "$z_{\\mathrm{inf}}$"])
	a0.axis("off")
	a2.plot(xs[tr][:,0],color='k',label="$x_1$",alpha=0.9)
	a2.plot(q_x[:,0],color=[0.3,0.3,0.3],linestyle='--',label="$\hat{x}_1$",alpha=0.9)
	# for i in range(5):
	# 	a2.plot(q_lem.sample_continuous_states()[tr],'k',alpha=0.5)

	a2.fill_between(np.arange(T),q_x[:,0]-q_std*2.0, q_x[:,0]+q_std*2.0, facecolor='k', alpha=0.3)
	ub = bound_func(np.arange(T), 1.0, latent_acc.transitions.ap, latent_acc.transitions.lamb, 3.0)
	a2.plot(np.arange(T), ub, 'b--')
	a2.plot(np.arange(T), -1.0 * ub, 'r--')
	a2.set_ylim([-1.2,1.2])
	a2.set_xlim([-1,101])
	a2.set_xticks([])
	a2.set_yticks([-1,0,1])
	a2.set_ylabel("x")
	if legend:
		a2.legend()
	sns.despine()
	for n in range(N):
		a3.eventplot(np.where(ys[tr][:,n]>0)[0], linelengths=0.5, lineoffsets=1+n,color='k')
	sns.despine()
	a3.set_yticks([])
	a3.set_xlim([-1,101])

	a1.plot(0.2*us[tr][:,0],color=[1.0,0.5,0.5], label=None,alpha=0.9)
	a1.set_yticks([])
	a1.set_xticks([])
	a1.axes.get_yaxis().set_visible(False)
	plt.tight_layout()

	return

tr = npr.randint(N_samples)
plot_trial(test_acc, q_lem, "laplace_em", tr=tr)
