import autograd.numpy as np
import autograd.numpy.random as npr

from ssmdm.models2 import Accumulation, LatentAccumulation
from ssm.variational import SLDSMeanFieldVariationalPosterior, SLDSTriDiagVariationalPosterior, \
	SLDSStructuredMeanFieldVariationalPosterior
import copy
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
sns.set_context("talk")
from ssmdm.misc import smooth, generate_clicks, generate_clicks_D
from ssm.util import softplus
npr.seed(12345)

# rSLDS with 3D race accumulator dynamics, Gaussian emissions
D = 1 		# number of accumulation dimensions
K = 3 		# number of discrete states = number of accumulator dimensions + 1
M = 1 		# number of input dimensions, same as D
N = 25		# number of observations
latent_acc = LatentAccumulation(N, K, D, M=M,
								transitions="ddm",
								emissions="poisson",
								emission_kwargs={"bin_size":0.01})
latent_acc.dynamics.betas = 0.9*np.ones((1,))
latent_acc.dynamics._log_sigmasq[0] = np.log(2e-3)
# latent_acc.emissions.inv_etas = np.log(1e-3)*np.ones((1,N))

# AR-HMM with ND race accumulator observations
# acc = Accumulation(K, D, M=M)

latent_acc.emissions.Cs[0] = 1 * npr.randn(N,D) + npr.choice([-20,20],(N,D))
latent_acc.emissions.ds[0] = 30 + 3.0 * npr.randn(N)
# ddm
# D = 1
# K = 3
# M = 1
# N = 25
# latent_acc = LatentAccumulation(N, 3, 1, M=1,
# 								transitions="ddm",
# 								emissions="gaussian")
# latent_acc.emissions.inv_etas = np.log(1e-2)*np.ones((1,N))

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
    # rate_r = np.random.randint(0,total_rate+1)
    # rate_r = 10 + np.random.randint(0,20+1)
    rate_r = npr.choice([10,30])
    rate_l = total_rate - rate_r
    rates = [rate_r,rate_l]

    # generate binned right and left clicks
    # rates = [10, 10, 10]
    # idx = int(npr.choice([0,1,2]))
    # rates[idx] = 30
    u = generate_clicks_D(rates,T=trial_time,dt=dt)

    # input is sum of u_r and u_l
    u = (0.075*np.array(u[1] - u[0]).T).reshape((T,1))
	# u = 0.075*np.array(u).T
    # u = npr.choice([-0.05,0.05])
    # u = u*np.ones((T,1))
    z, x, y = latent_acc.sample(T, input=u)

    us.append(u)
    zs.append(z)
    xs.append(x)
    ys.append(y)

plt.ion()
plt.figure()
for tr in range(N_samples):
	plt.plot(xs[tr],'k',alpha=0.5)

test_ddm = Accumulation(3, 1, M=1, transitions="ddm", observation_kwargs={"learn_A":False})
# fit HMM model to xs
# test_acc = Accumulation(K, D, M=M, observation_kwargs={"learn_A":False}) # initialize
# test_acc2 = Accumulation(K, D, M=M, observation_kwargs={"learn_A":True}) # initialize
# change the parameters
# test_acc2.observations.params = (0.5*np.ones(D,), np.log(5e-3)*np.ones(D,), 1.1*np.ones((D,1)))
# test_acc.observations.params = (0.5*np.ones(D,), np.log(5e-3)*np.ones(D,))
# test_acc.fit(xs, inputs=us) #fit
# test_acc2.fit(xs, inputs=us) #fit

# fit SLDS model to ys
# initialize
# test_acc = LatentAccumulation(N, K, D, M=M, emissions="gaussian")
test_acc = LatentAccumulation(N, K, D, M=M, transitions="ddm", dynamics_kwargs={"learn_A":True}, emissions="poisson", emission_kwargs={"bin_size":0.01})
test_acc.initialize(ys, inputs=us)
init_params = copy.deepcopy(test_acc.params)

# fit
q_lem = SLDSStructuredMeanFieldVariationalPosterior(test_acc, ys, inputs=us)
q_elbos = test_acc.fit(q_lem, ys, inputs=us, num_iters=25, method="laplace_em", initialize=False,
						num_samples=1, alpha=0.5, continuous_maxiter=50, emission_optimizer_maxiter=50, continuous_optimizer="newton")
q_elbos2 = test_acc.fit(q_lem, ys, inputs=us, num_iters=5, method="laplace_em", initialize=False,
						num_samples=1, alpha=0.5, continuous_maxiter=50, emission_optimizer_maxiter=50, continuous_optimizer="newton")


plt.figure()
plt.plot(q_elbos[1:])
plt.xlabel("iteration")
plt.ylabel("ELBO")

plt.ion()
plt.figure()
plt.imshow(np.concatenate((latent_acc.emissions.Cs[0,:,:],test_acc.emissions.Cs[0,:,:]),axis=1),aspect='auto')
plt.xticks([0.5, 2.5], ["$C_{\\mathrm{true}}$", "$C_{\\mathrm{FA+Init}}$"])

def plot_sim_results(tr=0):

    xlim = (0, np.shape(xs[tr])[0])
    q_x = q_lem.mean_continuous_states[tr]
    yhat = test_acc.smooth(q_x, ys[tr], input=us[tr])
    plt.ion()
    plt.figure(figsize=[12,6])
    plt.subplot(231)
    # plt.plot(xs[tr],'k',label="true")
    plt.plot(xs[tr][:,0],'r', label="true")
    # plt.plot(xs[tr][:,1],'b')
    # plt.plot(xs[tr][:,2],'k')
    # plt.plot(q_x,'r--',label="inferred")
    plt.plot(q_x[:,0],'m--', label="inferred")
    # plt.plot(q_x[:,1],'c--')
    # plt.plot(q_x[:,2],'g--')
    plt.xlim(xlim)
    plt.xlabel('time bin')
    plt.ylabel('x')
    plt.legend()

    if np.shape(yhat)[1] < 6:
	    plt.subplot(232)
	    true_y = softplus(np.dot(xs[tr], latent_acc.emissions.Cs[0].T) + latent_acc.emissions.ds[0])
	    yhat = softplus(np.dot(q_x, test_acc.emissions.Cs[0].T) + test_acc.emissions.ds[0])
	    # plt.plot(latent_acc.emissions.mean(np.dot(xs[tr],C.T)+d),'k')
	    plt.plot(smooth(ys[tr],3) / latent_acc.emissions.bin_size, 'k')
	    # plt.plot(ys[tr], 'k')
	    # plt.plot(true_y, 'k')
	    plt.plot(yhat,'r--')
	    # plt.plot(yhat / bin_size,'r--')
	    plt.xlim(xlim)
	    plt.xlabel('time bin')
	    plt.ylabel('y (observations)')

    else:
	    plt.subplot(232)
	    # true_y = smooth(ys[tr],20) / test_acc.emissions.bin_size
	    true_y = softplus(np.dot(xs[tr], latent_acc.emissions.Cs[0].T) + latent_acc.emissions.ds[0])
	    # smooth_y = yhat
	    smooth_y = yhat / test_acc.emissions.bin_size
	    lim = max(true_y.max(), smooth_y.max())
	    lim_min = min(true_y.min(), smooth_y.min())
	    plt.imshow(true_y.T,aspect="auto", vmin=lim_min, vmax=lim)
	    plt.title("true rate")
	    plt.colorbar()
	    plt.xlabel('time bin')
	    plt.ylabel('neuron')
	    plt.subplot(233)
	    plt.title("inferred rate")
	    plt.imshow(smooth_y.T,aspect="auto", vmin=lim_min, vmax=lim)
	    plt.colorbar()
	    plt.xlabel('time bin')
	    plt.ylabel('neuron')

    zhat = test_acc.most_likely_states(q_x, ys[tr], input=us[tr])
    plt.subplot(234)
    plt.imshow(np.row_stack((zs[tr], zhat)), aspect="auto")
    plt.plot(xlim, [0.5, 0.5], '-k', lw=2)
    plt.xlim(xlim)
    plt.yticks([0, 1], ["$z_{\\mathrm{true}}$", "$z_{\\mathrm{inf}}$"])
    plt.xlabel('time bin')

    D = latent_acc.D
    plt.subplot(235)
    plt.imshow(np.concatenate((latent_acc.emissions.Cs[0,:,:],test_acc.emissions.Cs[0,:,:]),axis=1),aspect='auto', cmap="inferno")
    plt.axvline(x=D-0.5,color='k',linewidth=1)
    plt.xticks([D/2 - 0.5, 3*D/2 - 0.5], ["$C_{\\mathrm{true}}$", "$C_{\\mathrm{inf}}$"])
    plt.ylabel("neuron")
    plt.xlabel("dimension")
    plt.colorbar()

    plt.subplot(236)
    plt.imshow(np.concatenate((latent_acc.emissions.ds.reshape((N,1)),test_acc.emissions.ds.reshape((N,1))),axis=1),aspect='auto', cmap="inferno")
    plt.axvline(x=0.5,color='k',linewidth=1)
    plt.xticks([0, 1], ["$d_{\\mathrm{true}}$", "$d_{\\mathrm{inf}}$"])
    plt.ylabel("neuron")
    plt.colorbar()
    plt.tight_layout()

plot_sim_results(tr=0)
