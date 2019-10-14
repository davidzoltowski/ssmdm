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
npr.seed(12345)

# 1D Accumulator with Poisson observations
D = 2 		# number of accumulation dimensions
K = 3 		# number of discrete states
M = 2 		# number of input dimensions
N = 10		# number of observations
bin_size = 0.01
latent_acc = LatentAccumulation(N, K, D, M=M,
								transitions="racehard",
								emissions="poisson",
								emission_kwargs={"bin_size":bin_size})
betas = 0.05*np.ones((D,))
sigmas = np.log(1e-3)*np.ones((D,))
latent_acc.dynamics.params = (betas, sigmas, latent_acc.dynamics.params[2])
# latent_acc.dynamics._log_sigmasq[0] = np.log(1e-3)*np.ones((D,))
latent_acc.emissions.Cs[0] = 4 * npr.randn(N,D) + npr.choice([-15,15],(N,D))
latent_acc.emissions.ds[0] = 40 + 4.0 * npr.randn(N)

# Sample state trajectories
T = 100 # number of time bins
trial_time = 1.0 # trial length in seconds
dt = 0.01 # bin size in seconds
N_samples = 400

# input statistics
total_rate = 40 # the sum of the right and left poisson process rates is 40

us = []
zs = []
xs = []
ys = []

for smpl in range(N_samples):

    # randomly draw right and left rates
    rate_r = np.random.randint(0,total_rate+1)
    # rate_r = 10 + np.random.randint(0,20+1)
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

# initialize model
test_acc = LatentAccumulation(N, K, D, M=M,
								transitions="racehard",
								emissions="poisson",
								emission_kwargs={"bin_size":bin_size})
# betas0 = 0.05*np.ones((D,))
# betas0 = (0.02+0.08*npr.rand())*np.ones((D,))
betas0 = 0.02*np.ones((D,))
sigmas0 = np.log((4e-5+2.5e-3*npr.rand()))*np.ones((D,))
test_acc.dynamics.params = (betas0, sigmas0, test_acc.dynamics.params[2])
test_acc.emissions.ds = (np.mean([y[0:3] for y in ys],axis=(0,1)) / bin_size).reshape((1,N))

u_sum = np.array([np.sum(u[:,0] - u[:,1]) for u in us])
y_end = np.array([y[-10:] for y in ys])
y_U = y_end[np.where(u_sum>=25)]
y_L = y_end[np.where(u_sum<=-25)]
C_init = np.hstack((np.mean(y_U,axis=(0,1))[:,None],np.mean(y_L,axis=(0,1))[:,None])) / bin_size - test_acc.emissions.ds.T

# test_acc.initialize(ys, inputs=us)
test_acc.emissions.Cs[0] = C_init
init_params = copy.deepcopy(test_acc.params)

# init posterior
init_var = 1e-4
_, q_lem = test_acc.fit(ys, inputs=us, method="laplace_em",
						variational_posterior="structured_meanfield",
						num_iters=0, initialize=False,
						variational_posterior_kwargs={"initial_variance":init_var})
q_lem_init = copy.deepcopy(q_lem)
q_elbos2, q_lem = test_acc.fit(ys, inputs=us, method="laplace_em",
							  variational_posterior=q_lem,
							  num_iters=15, alpha=0.5, initialize=False, num_samples=5, parameters_update="sgd", emission_optimizer_maxiter=20)#,
q_elbos2, q_lem = test_acc.fit(ys, inputs=us, method="laplace_em",
							  variational_posterior=q_lem,
							  num_iters=50, alpha=0.5, initialize=False, num_samples=5)#,
model150 = copy.deepcopy(test_acc)
q150 = copy.deepcopy(q_lem)
q_elbos4, q_lem = test_acc.fit(ys, inputs=us, method="laplace_em",
							  variational_posterior=q_lem,
							  num_iters=50, alpha=0.5, initialize=False)#,
model100 = copy.deepcopy(test_acc)
q100 = copy.deepcopy(q_lem)

C_init = init_params[-1][0][0]
plt.ion()
plt.figure()
plt.imshow(np.concatenate((latent_acc.emissions.Cs[0],C_init,test_acc.emissions.Cs[0]),axis=1),aspect='auto', cmap="inferno")
plt.xticks([0.5, 2.5, 4.5], ["$C_{\\mathrm{true}}$", "C_init", "$C_{\\mathrm{inf}}$"])
plt.colorbar()

total_elbos = np.concatenate((q_elbos[1:],q_elbos2[1:],q_elbos3[1:],q_elbos4[1:],q_elbos5[1:]))
plt.ion()
plt.figure()
plt.plot(total_elbos)
plt.xlabel("iteration")
plt.ylabel("ELBO")
plt.tight_layout()

# tr = 0
C = latent_acc.emissions.Cs[0]
d = latent_acc.emissions.ds[0]
def plot_sim_results(tr=0):

    xlim = (0, np.shape(xs[tr])[0])
    q_x = q_lem.mean_continuous_states[tr]
    yhat = test_acc.smooth(q_x, ys[tr], input=us[tr])
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

    if np.shape(yhat)[1] < 11:
	    plt.subplot(232)
	    # plt.plot(latent_acc.emissions.mean(np.dot(xs[tr],C.T)+d),'k')
	    plt.plot(smooth(ys[tr],10) / bin_size, 'k')
	    plt.plot(yhat / bin_size,'r--')
	    plt.xlim(xlim)
	    plt.xlabel('time bin')
	    plt.ylabel('y (observations)')

	    plt.subplot(233)
	    true_y = softplus(np.dot(xs[tr], C.T) + d)
	    smooth_y = yhat / bin_size
	    plt.plot(true_y, 'k')
	    plt.plot(smooth_y, 'r--')

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

    zhat = test_acc.most_likely_states(q_x, ys[tr], input=us[tr])
    plt.subplot(234)
    plt.imshow(np.row_stack((zs[tr], zhat)), aspect="auto", vmin=0, vmax=2)
    plt.plot(xlim, [0.5, 0.5], '-k', lw=2)
    plt.xlim(xlim)
    plt.yticks([0, 1], ["$z_{\\mathrm{true}}$", "$z_{\\mathrm{inf}}$"])
    plt.xlabel('time bin')

    plt.subplot(235)
    plt.imshow(np.concatenate((C,test_acc.emissions.Cs[0,:,:]),axis=1),aspect='auto', cmap="inferno")
    plt.xticks([0.5, 2.5], ["$C_{\\mathrm{true}}$", "$C_{\\mathrm{inf}}$"])
    plt.colorbar()

    plt.subplot(236)
    plt.imshow(np.concatenate((d.reshape((N,1)),test_acc.emissions.ds.reshape((N,1))),axis=1),aspect='auto', cmap="inferno")
    plt.xticks([0, 1], ["$d_{\\mathrm{true}}$", "$d_{\\mathrm{inf}}$"])
    plt.colorbar()
    plt.tight_layout()

plot_sim_results(tr=0)

z_dir = []
z_t = []
true_z_dir = []
true_z_t = []
for tr in range(N_samples):
	q_z = q_lem.mean_discrete_states[tr][0]
	zhat = np.argmax(q_z,axis=1)
	z_not0 = np.where(zhat > 0)[0]
	if z_not0.shape[0] > 0:
		z_t.append(z_not0[0])
	else:
		z_t.append(T+1)
	z_dir.append(zhat[-1])

	true_z_dir.append(zs[tr][-1])
	true_z_not0 = np.where(zs[tr] > 0)[0]
	if true_z_not0.shape[0] > 0:
		true_z_t.append(true_z_not0[0])
	else:
		true_z_t.append(T+1)

from ssm.primitives import blocks_to_full

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
	# q_std_1 = q_lem_std[:T]
	# q_std_2 = q_lem_std[T:]
	q_std_1 = q_lem_std[:,0]
	q_std_2 = q_lem_std[:,1]
	# q_x = q_lem.mean_continuous_states[tr]
	f, (a0, a1, a2, a3) = plt.subplots(4, 1, gridspec_kw={'height_ratios': [0.5, 0.5, 3, 1]})
	# a0.imshow(zs[tr][:,None].T,aspect="auto",vmin=0,vmax=2,cmap=cmap)
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
	# a2.plot(0.2*us[tr][:,0],color=[1.0,0.5,0.5], label=None,alpha=0.9)
	# a2.plot(0.2*us[tr][:,1],color=[0.5,0.5,1.0], label=None,alpha=0.9)
	a2.plot(np.array([0,100]),np.array([1,1]),'k--',linewidth=1.0,label=None)
	a2.set_ylim([-0.4,1.4])
	a2.set_xlim([-1,101])
	a2.set_xticks([])
	a2.set_yticks([0,1])#,["-B","0","B"])
	# a1.set_yticklabels(["0","B"])
	# a0.set_xlabel("t")
	# a1.set_ylabel("x")
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

tr = npr.randint(400)
plot_trial(tr=tr)
print("Trial: ", tr)

plt.figure(figsize=[6,3])
plt.subplot(121)
plt.imshow(np.concatenate((C,test_acc.emissions.Cs[0,:,:]),axis=1),aspect='auto', cmap="inferno")
plt.xticks([0.5, 2.5], ["$C_{\\mathrm{true}}$", "$C_{\\mathrm{inf}}$"])
plt.colorbar()

plt.subplot(122)
plt.imshow(np.concatenate((d.reshape((N,1)),test_acc.emissions.ds.reshape((N,1))),axis=1),aspect='auto', cmap="inferno")
plt.xticks([0, 1], ["$d_{\\mathrm{true}}$", "$d_{\\mathrm{inf}}$"])
plt.colorbar()
plt.tight_layout()


# q_x = np.copy(q_lem.mean_continuous_states)
#
# T_x = 10
# x1to10 = np.vstack([x[:T_x] for x in q_x])
# x2to11 = np.vstack([x[1:T_x+1] for x in q_x])
# u1to11 = np.vstack([u[1:T_x+1] for u in us])
# x_diff = x2to11 - x1to10
# beta1 = 1.0 / np.dot(u1to11[:,0], u1to11[:,0]) * np.dot(u1to11[:,0], x_diff[:,0])
# beta2 = 1.0 / np.dot(u1to11[:,1], u1to11[:,1]) * np.dot(u1to11[:,1], x_diff[:,1])
