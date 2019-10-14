import autograd.numpy as np
import autograd.numpy.random as npr

from ssmdm.models2 import Accumulation, LatentAccumulation
from ssm.variational import SLDSMeanFieldVariationalPosterior, SLDSTriDiagVariationalPosterior, \
	SLDSStructuredMeanFieldVariationalPosterior
import copy
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")
sns.set_context("talk")
from ssmdm.misc import smooth, generate_clicks, generate_clicks_D
from ssm.util import softplus
npr.seed(1)

# 1D Accumulator with Poisson observations
D = 1 		# number of accumulation dimensions
K = 3 		# number of discrete states
M = 1 		# number of input dimensions
N = 5		# number of observations
bin_size = 0.01
latent_acc = LatentAccumulation(N, K, D, M=M,
								transitions="ddm",
								emissions="poisson",
								emission_kwargs={"bin_size":bin_size})
latent_acc.dynamics.Vs[0] = 0.9*np.ones((D,))
latent_acc.dynamics._log_sigmasq[0] = np.log(5e-4)*np.ones((D,))
latent_acc.emissions.Cs[0] = 1 * npr.randn(N,D) + npr.choice([-20,20],(N,D))
latent_acc.emissions.ds[0] = 30 + 3.0 * npr.randn(N)

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
    rate_r = npr.choice([10,30])
    rate_l = total_rate - rate_r
    rates = [rate_r,rate_l]

    # generate binned right and left clicks
    u = generate_clicks_D(rates,T=trial_time,dt=dt)

    # input is sum of u_r and u_l
    u = (0.075*np.array(u[1] - u[0]).T).reshape((T,1))
    z, x, y = latent_acc.sample(T, input=u)

    us.append(u)
    zs.append(z)
    xs.append(x)
    ys.append(y)

# fit SLDS model to ys
# initialize
test_acc = LatentAccumulation(N, K, D, M=M, transitions="ddm",
							  emissions="poisson", emission_kwargs={"bin_size":bin_size})
test_acc.initialize(ys, inputs=us)
init_params = copy.deepcopy(test_acc.params)

# fit
q_elbos, q_lem = test_acc.fit(ys, inputs=us, method="laplace_em",
							  variational_posterior="structured_meanfield",
							  num_iters=25, alpha=0.5, initialize=False)

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
    for d in range(np.shape(q_x)[1]):
	    plt.plot(xs[tr],'k', label="true" if d==0 else None)
	    plt.plot(q_x,'r--',label="inferred" if d==0 else None)
    plt.xlim(xlim)
    plt.xlabel('time bin')
    plt.ylabel('x')
    plt.legend()

    if np.shape(yhat)[1] < 6:
	    plt.subplot(232)
	    true_y = softplus(np.dot(xs[tr], latent_acc.emissions.Cs[0].T) + latent_acc.emissions.ds[0])
	    yhat = softplus(np.dot(q_x, test_acc.emissions.Cs[0].T) + test_acc.emissions.ds[0])
	    # plt.plot(latent_acc.emissions.mean(np.dot(xs[tr],C.T)+d),'k')
	    from scipy.ndimage import gaussian_filter1d
	    y_smooth = smooth(ys[tr],5) / latent_acc.emissions.bin_size
	    for n in range(np.shape(ys[tr])[1]):
		    plt.plot(gaussian_filter1d(y_smooth[:,n],3),'k',alpha=0.75)
	    # plt.plot(smooth(ys[tr],5) / latent_acc.emissions.bin_size, 'k', alpha=0.75)
	    # plt.plot(ys[tr], 'k')
	    # plt.plot(true_y, 'k')
	    plt.plot(yhat,'r',alpha=0.75)
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
