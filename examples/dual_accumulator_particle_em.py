import autograd.numpy as np
import autograd.numpy.random as npr

from ssmdm.accumulation import Accumulation, LatentAccumulation

import copy
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")
sns.set_context("talk")
from ssmdm.misc import smooth, generate_clicks, generate_clicks_D
from ssmdm.misc import *
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
								emission_kwargs={"bin_size":bin_size},
								dynamics_kwargs={"learn_A":False})
betas0 = 0.02+0.08*npr.rand()*np.ones((D,))
sigmas0 = np.log((4e-5+3.5e-3*npr.rand()))*np.ones((D,))
test_acc.dynamics.params = (betas0, sigmas0) # test_acc.dynamics.params[2])

# Initialize C, d
u_sum = np.array([np.sum(u[:,0] - u[:,1]) for u in us])
y_end = np.array([y[-10:] for y in ys])
y_U = y_end[np.where(u_sum>=25)]
y_L = y_end[np.where(u_sum<=-25)]
d_init = (np.mean([y[:5] for y in ys],axis=(0,1)) / bin_size).reshape((1,N))
C_init = np.hstack((np.mean(y_U,axis=(0,1))[:,None],np.mean(y_L,axis=(0,1))[:,None])) / bin_size - d_init.T
test_acc.emissions.ds[0] = d_init
test_acc.emissions.Cs[0] = C_init
init_params = copy.deepcopy(test_acc.params)
test_acc_vlem = copy.deepcopy(test_acc)

# fit model with particle em
logjoints, all_particles, all_weights = test_acc._fit_particle_em(ys, inputs=us,
											N_particles=10, num_iters=2)
final_params = test_acc.params

ssmdm_dir = os.path.expanduser('/tigress/dz5/ssmdm')
save_name = "dual_accumulator_particle_em.npz"
np.savez(os.path.join(ssmdm_dir, save_name), logjoints=logjoints,
		all_particles=all_particles, all_weights=all_weights)
        final_params=final_params, init_params=init_params)#,
	#q_params=q_params, all_ys=all_ys, all_xs=all_xs, all_us=all_us, all_zs=all_zs)

d = np.load("dual_accumulator_particle_em_100.npz", allow_pickle=True)
ys = d["ys"]
us = d["us"]
xs = d["xs"]
zs = d["zs"]
ys = [np.array(y) for y in ys]
us = [np.array(u) for u in us]
init_params = d["init_params"]
final_params = d["final_params"]
all_particles = d["all_particles"]
all_weights = d["all_weights"]
logjoints = d["logjoints"]

test_acc_pem = LatentAccumulation(N, K, D, M=M,
								transitions="race",
								emissions="poisson",
								emission_kwargs={"bin_size":bin_size},
								dynamics_kwargs={"learn_A":False})
test_acc_pem.params = final_params
new_particles, new_weights = test_acc_pem.smc(ys, inputs=us, N_particles=50)
new_particles2, new_weights2 = test_acc_pem.smc(ys, inputs=us, N_particles=100)

ys_trials = [ys[31], ys[55], ys[82], ys[99]]
us_trials = [us[31], us[55], us[82], us[99]]
new_particles, new_weights = test_acc_pem.smc(ys_trials, inputs=us_trials, N_particles=100)
new_particles2, new_weights2 = test_acc_pem.smc(ys_trials, inputs=us_trials, N_particles=200)

test_acc = LatentAccumulation(N, K, D, M=M,
								transitions="race",
								emissions="poisson",
								emission_kwargs={"bin_size":bin_size},
								dynamics_kwargs={"learn_A":False})
test_acc.params = init_params
# fit model
# init_var = 1e-3
ys = [np.array(y) for y in ys]
us = [np.array(u) for u in us]
q_elbos_lem, q_lem = test_acc.fit(ys, inputs=us, method="laplace_em",
						variational_posterior="structured_meanfield",
						num_iters=25, alpha=0.5, initialize=False) #,
						# variational_posterior_kwargs={"initial_variance":init_var})
q_elbos_lem2, q_lem = test_acc.fit(ys, inputs=us, method="laplace_em",
						variational_posterior=q_lem,
						num_iters=25, alpha=0.5, initialize=False) #,
# plt.ion()
plt.figure()
# plt.plot(np.concatenate((q_elbos_lem[1:], q_elbos_lem2[1:])), label="vLEM")
plt.plot(q_elbos_lem[1:], label="vLEM")
# plt.plot(logjoints, label="particle EM")
plt.xlabel("iteration")
plt.ylabel("ELBO")
plt.legend()
plt.tight_layout()

tr=0
tr += 1
particles = all_particles[tr]
q_x_pem = np.mean(particles, axis=0)
q_x_lem = q_lem.mean_continuous_states[tr]
plt.figure()
plt.plot(np.array([0,100]),np.array([1,1]),'k--',linewidth=1.0,label=None)
plt.plot(xs[tr][:,0],'k', label="True")
plt.plot(xs[tr][:,1],'k', label=None)
plt.plot(q_x_lem[:,0],'r', label="vLEM")
plt.plot(q_x_lem[:,1],'r', label=None)
plt.plot(q_x_pem[:,0],'b', label="pEM")
plt.plot(q_x_pem[:,1],'b', label=None)
plt.legend()

plt.figure()
trials = npr.choice(100,6, replace=False)
for i in range(6):
	plt.subplot(2,3,i+1)
	tr = trials[i]
	particles = all_particles[tr]
	q_x_pem = np.mean(particles, axis=0)
	q_x_lem = q_lem.mean_continuous_states[tr]
	plt.plot(np.array([0,100]),np.array([1,1]),'k--',linewidth=1.0,label=None)
	plt.plot(xs[tr][:,0],'k', label="True")
	plt.plot(xs[tr][:,1],'k', label=None)
	plt.plot(q_x_lem[:,0],'r', label="vLEM")
	plt.plot(q_x_lem[:,1],'r', label=None)
	plt.plot(q_x_pem[:,0],'b', label="pEM")
	plt.plot(q_x_pem[:,1],'b', label=None)
	if i == 5:
		plt.legend()

plt.figure()
# trials = npr.choice(100,6, replace=False)
for i in range(6):
	# particles = all_particles[tr]
	particles = new_particles3[i]
	plt.subplot(2,3,i+1)
	tr = trials[i]
	plt.plot(xs[tr][:,0],'k', label="True")
	plt.plot(xs[tr][:,1],'k', label=None)
	for particle in particles:
		plt.plot(particle[:,0],'b', alpha=0.5)
		plt.plot(particle[:,1],'b', alpha=0.5)

# compute MSE in xs
mse_lem = 0.0
mse_pem = 0.0
# mse_pem2 = 0.0
errors_pem = []
errors_lem = []
for tr in range(N_samples):
	particles = all_particles[tr]
	q_x_pem = np.mean(particles, axis=0)
	mse_pem += np.mean((q_x_pem - xs[tr])**2)
	q_x_lem = q_lem.mean_continuous_states[tr]
	mse_lem += np.mean((q_x_lem - xs[tr])**2)
	errors_pem.append((q_x_pem - xs[tr])**2)
	errors_lem.append((q_x_lem - xs[tr])**2)
	# q_x_pem2 = np.mean(new_particles[tr],axis=0)
	# mse_pem2 += np.mean((q_x_pem2 - xs[tr])**2)


tr=-1
tr+=1
plt.figure()
plt.plot(xs[tr],'k')
q_x_pem_smoothed = np.mean(all_particles[tr],axis=0)
q_x_pem_filtered = np.mean(new_particles[tr],axis=0)
plt.plot(q_x_pem_smoothed,'r')
plt.plot(q_x_pem_filtered,'b')


# plot C and d
C_true = latent_acc.emissions.Cs[0]
C_pem = test_acc_pem.emissions.Cs[0]
C_lem = test_acc.emissions.Cs[0]
C_init = init_params[3][0][0]
plt.figure(figsize=[6,3])
plt.subplot(121)
plt.imshow(np.concatenate((C_true, C_lem, C_pem, C_init),axis=1),aspect='auto', cmap="inferno")
plt.xticks([0.5, 2.5, 4.5, 6.5], ["$C_{\\mathrm{true}}$", "$C_{\\mathrm{LEM}}$", "$C_{\\mathrm{pEM}}$", "$C_{\\mathrm{init}}$"])
plt.colorbar()

d_true = latent_acc.emissions.ds.T
d_pem = test_acc_pem.emissions.ds.T
d_lem = test_acc.emissions.ds.T
d_init = init_params[3][1].T
plt.subplot(122)
plt.imshow(np.concatenate((d_true, d_lem, d_pem, d_init),axis=1),aspect='auto', cmap="inferno")
plt.xticks([0, 1, 2, 3], ["$d_{\\mathrm{true}}$", "$d_{\\mathrm{LEM}}$","$d_{\\mathrm{pEM}}$", "$d_{\\mathrm{init}}$"])
plt.colorbar()
plt.tight_layout()



q = copy.deepcopy(q_lem)
def plot_trial(tr=0, legend=False):

	q_x = q.mean_continuous_states[tr]
	J_diag = q._params[tr]["J_diag"]
	J_lower_diag= q._params[tr]["J_lower_diag"]
	J = blocks_to_full(J_diag, J_lower_diag)
	Jinv = np.linalg.inv(J)
	q_lem_std = np.sqrt(np.diag(Jinv))
	q_lem_std = q_lem_std.reshape((T,D))
	q_std_1 = q_lem_std[:,0]
	q_std_2 = q_lem_std[:,1]

	yhat = test_acc.smooth(q_x, ys[tr], input=us[tr])
	zhat = test_acc.most_likely_states(q_x, ys[tr], input=us[tr])

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

def plot_trial_particles(tr=0, legend=False):

	idx = np.where(trials==tr)[0][0]
	q_x = np.mean(new_particles2[idx],axis=0)
	q_std = np.std(new_particles2[idx], axis=0)
	q_std_1 = q_std[:,0]
	q_std_2 = q_std[:,1]

	yhat = test_acc_pem.smooth(q_x, ys[tr], input=us[tr])
	zhat = test_acc_pem.most_likely_states(q_x, ys[tr], input=us[tr])

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
print(tr)
plot_trial(tr)
plot_trial_particles(tr)





sim_ys_1 = simulate_accumulator(test_acc, us, num_repeats=3)
sim_ys_2 = simulate_accumulator(test_acc_pem, us, num_repeats=3)
true_psths = plot_psths(ys, us, 1, N);
sim_psths_1 = plot_psths(sim_ys_1, us+us+us, 1, N);
sim_psths_2 = plot_psths(sim_ys_2, us+us+us, 1, N);
r2_lem = compute_r2(true_psths, sim_psths_1)
r2_mf = compute_r2(true_psths, sim_psths_2)
psth_list=[true_psths, sim_psths_1, sim_psths_2]
plot_neurons2 = npr.permutation(np.arange(N))[:3]
plot_multiple_psths(psth_list, plot_neurons2)

plt.gcf().axes[3].set_ylim(plt.gcf().axes[0].get_ylim())
plt.gcf().axes[6].set_ylim(plt.gcf().axes[0].get_ylim())
plt.gcf().axes[4].set_ylim(plt.gcf().axes[1].get_ylim())
plt.gcf().axes[7].set_ylim(plt.gcf().axes[1].get_ylim())
plt.gcf().axes[5].set_ylim(plt.gcf().axes[2].get_ylim())
plt.gcf().axes[8].set_ylim(plt.gcf().axes[2].get_ylim())

# compare z
z_dir = []
z_t = []
z_dir_bbvi = []
z_t_bbvi = []
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

	# BBVI
	q_x = np.mean(all_particles[tr],axis=0)
	zhat = test_acc_pem.most_likely_states(q_x, ys[tr], input=us[tr])
	z_not0 = np.where(zhat > 0)[0]
	if z_not0.shape[0] > 0:
		z_t_bbvi.append(z_not0[0])
	else:
		z_t_bbvi.append(T+1)
	z_dir_bbvi.append(zhat[-1])
