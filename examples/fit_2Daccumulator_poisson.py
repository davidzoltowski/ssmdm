import autograd.numpy as np
import autograd.numpy.random as npr

from ssmdm.models import Accumulator2D, LatentAccumulator2DPoisson
from ssm.variational import SLDSMeanFieldVariationalPosterior, SLDSTriDiagVariationalPosterior, \
	SLDSStructuredMeanFieldVariationalPosterior
from ssm.util import softplus

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
sns.set_context("talk")
from ssmdm.misc import smooth, generate_clicks
npr.seed(1234)

# create 2D Poisson accumulator
betas = np.array([0.9,0.9])
# betas = np.array([0.9,0.9])
a_diag = np.ones((3,2,1))
log_sigma_scale = np.log(20)*np.ones(2,)
acc2 = Accumulator2D(betas=betas,log_sigma_scale=log_sigma_scale,a_diag=a_diag)

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

for smpl in range(N_samples):

    # randomly draw right and left rates
    # rate_r = np.random.randint(0,total_rate+1)
    # rate_r = 10 + np.random.randint(0,20+1)
    rate_r = np.random.choice([10,30])
    rate_l = total_rate - rate_r

    # generate binned right and left clicks
    u_r, u_l = generate_clicks(T=trial_time,dt=dt,rate_r=rate_r,rate_l=rate_l)

    # input is sum of u_r and u_l
    u = 0.075*np.array([u_r,u_l]).T
    z, x = acc2.sample(T, input=u)

    us.append(u)
    zs.append(z)
    xs.append(x)

# generate Poisson observations
bin_size = 0.01
N = 25
D = 2
C = 15 * npr.randn(N,D)
C[0,:] = np.array([-10.0,10.0])
C[1,:] = np.array([10.0,-10.0])
C[2,:] = np.array([15.0,0.0])
C[3,:] = np.array([0.0,15.0])
print("C mean: ", np.mean(C))
d= 25 + 3 * npr.randn(N)
ys = [np.random.poisson(softplus(np.dot(x, C.T) +d )*bin_size) for x in xs]

# initialize model
latent_acc = LatentAccumulator2DPoisson(N, link="softplus", bin_size=bin_size)
latent_acc.initialize(ys, inputs=us)

import copy
init_params = copy.deepcopy(latent_acc.params)

q_laplace_em = SLDSStructuredMeanFieldVariationalPosterior(latent_acc, ys, inputs=us)
for tr in range(N_samples):
    # q_laplace_em._params[tr]["h"] = xs[tr] + 0.1*npr.randn(*xs[tr].shape)
    q_laplace_em.params[tr]["h"] *= 100
    q_laplace_em.params[tr]["J_diag"] *= 100

# q_elbo = latent_acc.fit(q_laplace_em, ys, inputs=us, num_iters=1, method="laplace_em", initialize=False,
#                        num_samples=1, alpha=0.5, continuous_maxiter=25, emission_optimizer_maxiter=25)
print("Variance: ", np.diag(latent_acc.dynamics.Sigmas[1]))

iters = 25
elbos = []
dyn_var = []
# elbos.append(q_elbo)
dyn_var.append(np.diag(latent_acc.dynamics.Sigmas[1]))
for iter in range(iters):
    print("Iteration: ", iter)
    q_elbo = latent_acc.fit(q_laplace_em, ys, inputs=us, num_iters=1, method="laplace_em", initialize=False,
							num_samples=1, alpha=0.5, continuous_maxiter=50, emission_optimizer_maxiter=25)
    elbos.append(q_elbo[-1])
    dyn_var.append(np.diag(latent_acc.dynamics.Sigmas[1]))
    print("Elbo: ", elbos[-1])
    print("Variance: ", np.diag(latent_acc.dynamics.Sigmas[1]))
    print("Beta: ", np.diag(latent_acc.dynamics.Vs[1]))
    print("C: ", np.mean(latent_acc.emissions.Cs[0]))

plt.ion()
plt.figure()
plt.subplot(211)
plt.title("2D Accumulator, Exp")
plt.plot(elbos)
plt.ylabel("ELP")
plt.subplot(212)
plt.axhline(acc2.observations.Sigmas[1,0,0],color='k')
plt.plot(dyn_var,color='b')
plt.legend(["true","inferred"])
plt.yscale("log")
plt.xlabel("iteration")
plt.ylabel("dynamics var")


C_init = init_params[-1][0][0]
plt.ion()
plt.figure()
plt.imshow(np.concatenate((C,C_init,latent_acc.emissions.Cs[0,:,:]),axis=1),aspect='auto', cmap="inferno")
plt.xticks([0.5, 2.5, 4.5], ["$C_{\\mathrm{true}}$", "C_init", "$C_{\\mathrm{inf}}$"])
plt.colorbar()


# tr = 0
def plot_sim_results(tr=0):

    xlim = (0, np.shape(xs[tr])[0])
    q_x = q_laplace_em.mean_continuous_states[tr]
    yhat = latent_acc.smooth(q_x, ys[tr], input=us[tr])
    plt.ion()
    plt.figure(figsize=[12,6])
    plt.subplot(231)
    plt.plot(xs[tr][:,0],'r', label="true")
    plt.plot(xs[tr][:,1],'b')
    plt.plot(q_x[:,0],'m--', label="inferred")
    plt.plot(q_x[:,1],'c--')
    plt.xlim(xlim)
    plt.xlabel('time bin')
    plt.ylabel('x')
    plt.legend()

    if np.shape(yhat)[1] < 6:
	    plt.subplot(232)
	    # plt.plot(latent_acc.emissions.mean(np.dot(xs[tr],C.T)+d),'k')
	    plt.plot(smooth(ys[tr],10) / bin_size, 'k')
	    plt.plot(yhat / bin_size,'r--')
	    plt.xlim(xlim)
	    plt.xlabel('time bin')
	    plt.ylabel('y (observations)')

    else:
	    plt.subplot(232)
	    # true_y = smooth(ys[tr],20) / bin_size
	    true_y = softplus(np.dot(xs[tr], C.T) + d)
	    smooth_y = yhat / bin_size
	    lim = max(true_y.max(), smooth_y.max())
	    plt.imshow(true_y.T,aspect="auto", vmin=0, vmax=lim)
	    plt.colorbar()
	    plt.subplot(233)
	    plt.imshow(smooth_y.T,aspect="auto", vmin=0, vmax=lim)
	    plt.colorbar()

    zhat = latent_acc.most_likely_states(q_x, ys[tr], input=us[tr])
    plt.subplot(234)
    plt.imshow(np.row_stack((zs[tr], zhat)), aspect="auto")
    plt.plot(xlim, [0.5, 0.5], '-k', lw=2)
    plt.xlim(xlim)
    plt.yticks([0, 1], ["$z_{\\mathrm{true}}$", "$z_{\\mathrm{inf}}$"])
    plt.xlabel('time bin')

    plt.subplot(235)
    plt.imshow(np.concatenate((C,latent_acc.emissions.Cs[0,:,:]),axis=1),aspect='auto', cmap="inferno")
    plt.xticks([0.5, 2.5], ["$C_{\\mathrm{true}}$", "$C_{\\mathrm{inf}}$"])
    plt.colorbar()

    plt.subplot(236)
    plt.imshow(np.concatenate((d.reshape((N,1)),latent_acc.emissions.ds.reshape((N,1))),axis=1),aspect='auto', cmap="inferno")
    plt.xticks([0, 1], ["$d_{\\mathrm{true}}$", "$d_{\\mathrm{inf}}$"])
    plt.colorbar()
    plt.tight_layout()

plot_sim_results(tr=0)
