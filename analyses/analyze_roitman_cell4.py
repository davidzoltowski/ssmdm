import autograd.numpy as np
import autograd.numpy.random as npr

from ssmdm.ramping import ObservedRamping, Ramping, RampingHard, RampingLowerHard

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
sns.set_context("talk")
sns.set_style("ticks")
from ssmdm.misc import smooth
npr.seed(1)
import copy

# load neuron, setup data
from scipy.io import loadmat
d = loadmat("roitman_cell4.mat")
# d = loadmat("yates_cell1.mat")
ys = [y[0].astype(int) for y in d["ys"]]
us = [u[0] for u in d["us"]]
numTrials = len(ys)
cohs = np.array(d["cohs"]).astype(float)
choices = np.array(d["choices"])
N = 1 # 1 neuron
bin_size = 0.01
M = us[0].shape[1]
# cut off "longer" trials?

# initialize params
def initialize_ramp(ys,cohs, bin_size):
	coh5 = np.where(cohs==np.argmax(cohs))[0]
	# coh4 = np.where(cohs==3)[0]
	# coh5 = np.concatenate((np.where(cohs==4)[0],np.where(cohs==3)[0]))
	y_end = np.array([y[-5:] for y in ys])
	# y_end_5 = y_end[np.concatenate((coh5,coh4))]
	y_end_5 = y_end[coh5]
	C = np.mean(y_end_5) / bin_size
	y0_mean = np.mean([y[0] for y in ys])
	x0 = y0_mean / C / bin_size
	return C, x0

# C, x0 = initialize_ramp(ys, cohs, bin_size)

## TODO -> initialize based on choices?
def initialize_ramp_choice(ys,choices, bin_size):
	choice_0 = np.where(choices==0)[0]
	choice_1 = np.where(choices==1)[0]
	y_end = np.array([y[-5:] for y in ys])
	C0 = np.mean(y_end[choice_0]) / bin_size
	C1 = np.mean(y_end[choice_1]) / bin_size
	C_in = max(C0, C1)
	C_out = min(C0, C1)
	y0_mean = np.mean([y[:3] for y in ys])
	x0 = y0_mean / C_in / bin_size
	return C_in, x0, C_out

C_in, x0, C_out = initialize_ramp_choice(ys, choices, bin_size)

beta = np.array([-0.015,-0.005,0.0,0.01,0.02]) + 0.001*npr.randn(5)
if M == 6:
	beta = np.array([-0.015,-0.005,-0.005,0.0,0.01,0.02]) + 0.01*npr.randn(6)
# x0 = 0.5 + 0.05 * npr.randn(1)
log_sigma_scale = np.log(5e-4+5e-4*npr.rand())

# latent_ddm = Ramping(N, M=5, link="softplus", beta=beta, log_sigma_scale=log_sigma_scale, x0=x0, bin_size=bin_size)
ramp = RampingHard(N, M=M, link="softplus", beta=beta, log_sigma_scale=log_sigma_scale, x0=x0, bin_size=bin_size)
# ramp.initialize(ys, inputs=us, choices=choices)
ramp_lower = RampingLowerHard(N, M=M, link="softplus", beta=beta, log_sigma_scale=log_sigma_scale, x0=x0, bin_size=bin_size)
ramp.emissions.Cs[0] = C_in
ramp_lower.emissions.Cs[0] = C_in
lb_loc = max(min(0.3,C_out / C_in),0.0)
# lb_loc = C_out / C_in
ramp_lower.transitions.params = (lb_loc,ramp_lower.transitions.lb_scale)

_, q_lem = ramp.fit(ys, inputs=us, method="laplace_em",
							  variational_posterior="structured_meanfield",
							  num_iters=0, alpha=0.5, initialize=False,
							  variational_posterior_kwargs={"initial_variance":1e-5},
							  num_samples=1)
q_lem_init = copy.deepcopy(q_lem)
# q_elbos, q_lem = ramp.fit(ys, inputs=us, method="laplace_em",
# 							  variational_posterior=q_lem,
# 							  num_iters=50, alpha=0.75, initialize=False,
# 							  num_samples=1)

init_ramp_lower = copy.deepcopy(ramp_lower)
_, q_lem_lower = ramp_lower.fit(ys, inputs=us, method="laplace_em",
							  variational_posterior="structured_meanfield",
							  num_iters=0, alpha=0.5, initialize=False,
							  variational_posterior_kwargs={"initial_variance":1e-5},
							  num_samples=1)
q_lem_lower_init = copy.deepcopy(q_lem_lower)
q_elbos_lower, q_lem_lower = ramp_lower.fit(ys, inputs=us, method="laplace_em",
							  variational_posterior=q_lem_lower,
							  num_iters=10, alpha=0.75, initialize=False,
							  num_samples=1)
# q_elbos_lower, q_lem_lower = ramp_lower.fit(ys, inputs=us, method="laplace_em",
# 							  variational_posterior=q_lem_lower,
# 							  parameters_update="sgd", emission_optimizer_maxiter=25,
# 							  num_iters=5, alpha=0.75, initialize=False,
# 							  num_samples=2)
q_elbos_lower3, q_lem_lower = ramp_lower.fit(ys, inputs=us, method="laplace_em",
							  variational_posterior=q_lem_lower,
							  num_iters=25, alpha=0.75, initialize=False,
							  num_samples=1)
q_elbos_lower4, q_lem_lower = ramp_lower.fit(ys, inputs=us, method="laplace_em",
							  variational_posterior=q_lem_lower,
							  num_iters=10, alpha=0.75, initialize=False,
							  num_samples=1)
#
# plot elbos
sns.set_style("ticks")
plt.ion()
plt.figure()
# plt.plot(q_elbos[1:],label="Ramp")
plt.plot(q_elbos_lower[1:],label="Ramp Lower")
plt.legend()
plt.ylabel("ELBO")
plt.xlabel("iteration")
plt.tight_layout()


#
def plot_trial(q,model,ys,us,method="lem",true_xs=None,tr=0):

	if method == "lem":
		q_lem_x = q.mean_continuous_states[tr]
		from ssm.primitives import blocks_to_full
		J_diag = q._params[tr]["J_diag"]
		J_lower_diag= q._params[tr]["J_lower_diag"]
		J = blocks_to_full(J_diag, J_lower_diag)
		Jinv = np.linalg.inv(J)
		q_lem_std = np.sqrt(np.diag(Jinv))
	else:
		q_lem_x = q.mean[tr]
		q_lem_std = np.sqrt(np.exp(q.params[tr][1]))

	q_lem_z = model.most_likely_states(q_lem_x, ys[tr])

	max_rate = np.log1p(np.exp(model.emissions.Cs[0]+model.emissions.ds[0]))[0][0]

	yhat = model.smooth(q_lem_x, ys[tr], input=us[tr])
	# print("Coherence level: ", np.where(us[tr][0]==1.0)[0][0]+1)
	# plt.figure(figsize=[4,6])
	plt.figure(figsize=[12,4])
	plt.subplot(121)
	if true_xs is not None:
		plt.plot(true_xs[tr],'k',label="true")
	plt.plot(q_lem_x,'b',label="inferred")
	plt.fill_between(np.arange(np.shape(ys[tr])[0]),(q_lem_x-q_lem_std*2.0)[:,0], (q_lem_x+q_lem_std*2.0)[:,0], facecolor='b', alpha=0.3)
	plt.plot(np.array([0,np.shape(yhat)[0]]),np.array([1.0,1.0]),'r--')
	plt.ylim([-0.2,1.2])
	plt.legend()
	plt.xlabel("time bin")
	plt.ylabel("$x$")
	# plt.plot((q_lem_z[:,None]-1)*0.5)

	if np.shape(yhat)[1] < 6:
		plt.subplot(122)
		plt.plot(smooth(ys[tr],10) / bin_size,'k');
		plt.plot(yhat / bin_size,'b');
		plt.plot(np.array([0,np.shape(yhat)[0]]),np.array([max_rate,max_rate]))
		plt.eventplot(np.where(ys[tr]>0)[0], linelengths=2)
		plt.ylabel("$y$")
		plt.xlabel("time bin")
		plt.legend(["true","inferred"])

	else:
		plt.subplot(122)
		# true_y = smooth(ys[tr],20) / bin_size
		# smooth_y = yhat / bin_size
		# plt.imshow(np.concatenate((true_y, smooth_y),axis=1).T,aspect="auto")
		# plt.colorbar()
		# plt.legend(["true","inferred"])
		rand_neurons = npr.randint(0,np.shape(y)[1], (5))
		plt.plot(smooth(ys[tr][:,rand_neurons],10) / bin_size, 'k-');
		plt.plot(yhat[:,rand_neurons] / bin_size,'b');
		plt.plot(np.array([0,np.shape(yhat)[0]]),np.array([max_rate,max_rate]))
		plt.eventplot(np.where(ys[tr]>0)[0], linelengths=2)
		plt.ylabel("$y$")
		plt.xlabel("time bin")
		plt.legend(["true","inferred"])

	plt.tight_layout()
	plt.show()

plot_trial(q_lem_lower,ramp_lower,ys,us,method="lem",true_xs=None,tr=0)


# loop through coh 5 trials, get average changes before 1.0
coh5 = np.where(cohs==4)[0]
diffs = np.array([])
for trial in coh5:
	qx = q_lem.mean_continuous_states[trial]
	pre1 = np.where(qx<1)[0]
	if pre1.shape[0] < qx.shape[0]:
		pre1 = np.append(pre1, pre1[-1]+1)
	qx_pre1 = qx[pre1]
	diffs = np.append(diffs, qx_pre1[1:] - qx_pre1[:-1])


# simulate data
T = 100
num_sims = 1000
from ssm.util import one_hot
sim_zs, sim_xs, sim_ys, sim_us, sim_cohs = [], [], [], [], []
numCoh = np.copy(M)
model_sim = copy.deepcopy(ramp_lower)
for iter in range(num_sims):

	coh = npr.randint(numCoh)
	u_tr = one_hot(coh,numCoh)
	tr_length = np.copy(T)
	u = u_tr * np.ones((tr_length,1))

	# sample from Observed ramping and them map to observations? fit that instead???
	z, x, y = model_sim.sample(T, input=u)

	sim_zs.append(z)
	sim_xs.append(x)
	y = model_sim.emissions.mean(model_sim.emissions.forward(x, input=u, tag=None))
	sim_ys.append(y)
	sim_us.append(u)
	sim_cohs.append(coh)

# tr = 0
# tr+=1
# print(sim_us[tr][0])
# plt.figure()
# plt.plot(sim_xs[tr])

####
def plot_psth(ys,us,linestyle='-'):

	num_coh = np.shape(us[0])[1]
	# T = 1000
	T = max([y.shape[0] for y in ys])

	psth = np.zeros((num_coh,T))
	psth_count = np.zeros((num_coh,T))
	numTrials = len(ys)

	for tr in range(numTrials):

		u_trial = np.where(us[tr][0]==1)[0][0]
		y_trial = ys[tr]
		t_trial = np.shape(y_trial)[0]

		psth[u_trial,0:t_trial] += y_trial.flatten()
		psth_count[u_trial,0:t_trial] += 1.0

	psth = psth / psth_count
	psth[psth_count < 8] = None

	# plt.figure()
	plt.plot(smooth(psth[0][:,None],5)/0.01,linestyle=linestyle,color=[1.0,0.0,0.0],alpha=0.8)
	plt.plot(smooth(psth[1][:,None],5)/0.01,linestyle=linestyle,color=[1.0,0.5,0.5],alpha=0.8)
	plt.plot(smooth(psth[2][:,None],5)/0.01,linestyle=linestyle,color=[0.0,0.0,0.0],alpha=0.8)
	plt.plot(smooth(psth[3][:,None],5)/0.01,linestyle=linestyle,color=[0.5,0.5,1.0],alpha=0.8)
	plt.plot(smooth(psth[4][:,None],5)/0.01,linestyle=linestyle,color=[0.0,0.0,1.0],alpha=0.8)
	if num_coh == 6:
		plt.plot(smooth(psth[5][:,None],5)/0.01,linestyle=linestyle,color=[0.0,0.5,1.0],alpha=0.8)
	plt.xlabel("time bin")
	plt.ylabel("firing rate (sp/s)")
	plt.show()

plt.ion()
plt.figure(figsize=[6,4])
plot_psth(ys, us)
plot_psth(sim_ys, sim_us, '--')
plt.tight_layout()

def plot_choice_psth(ys,choices):

	num_coh = np.shape(us[0])[0]
	T = 1000

	psth = np.zeros((2,T))
	psth_count = np.zeros((2,T))
	numTrials = len(ys)

	for tr in range(numTrials):

		u_trial = choices[tr][0]-1
		y_trial = ys[tr]
		t_trial = np.shape(y_trial)[0]

		psth[u_trial,0:t_trial] += y_trial.flatten()
		psth_count[u_trial,0:t_trial] += 1.0

	psth = psth / psth_count

	plt.figure()
	plt.plot(smooth(psth[0][:,None],5)/0.01,color=[0.0,0.0,1.0])
	plt.plot(smooth(psth[1][:,None],5)/0.01,color=[1.0,0.0,0.0])
	plt.xlabel("time bin")
	plt.ylabel("firing rate (sp/s)")
	plt.legend(["in-RF choice","out-RF choice"])

# analyze variance
def compute_dyn_var(q_laplace_em, num_runs=1):

	numTrials = len(q_laplace_em.mean_continuous_states)
	dyn_vars = np.zeros(numTrials)
	num_runs = 1
	for j in range(num_runs):
		x_samples = q_laplace_em.sample_continuous_states()
		for i in range(numTrials):
			xi = x_samples[i]
			x_range = np.where(xi<1.0)[0]
			x_range = x_range[:-1]
			x_diffs = xi[x_range+1] - xi[x_range]
			if len(x_diffs)>0:
				dyn_vars[i] += np.var(x_diffs,ddof=1)

	dyn_vars /= num_runs
	dyn_vars_nonzero = dyn_vars[dyn_vars>0]
	return np.mean(dyn_vars_nonzero)


# plt.figure(figsize=[12,4])
# plt.hist(dyn_vars_nonzero,40)
# plt.axvline(np.exp(np.log(1e-4)+latent_ddm.dynamics.log_sigma_scale),color='k')
# plt.xlabel("variance")
# plt.ylabel("frequency")
# plt.tight_layout()
# plt.savefig("sample_dynamics_var_N100.png")


def plot_posterior_spikes(q,model,ys,us,tr=0):

	q_lem_x = q.mean_continuous_states[tr]
	from ssm.primitives import blocks_to_full
	J_diag = q._params[tr]["J_diag"]
	J_lower_diag= q._params[tr]["J_lower_diag"]
	J = blocks_to_full(J_diag, J_lower_diag)
	Jinv = np.linalg.inv(J)
	q_lem_std = np.sqrt(np.diag(Jinv))

	q_lem_z = model.most_likely_states(q_lem_x, ys[tr])

	# max_rate = np.log1p(np.exp(model.emissions.Cs[0]+model.emissions.ds[0]))[0][0]
	f, (a0, a1, a2) = plt.subplots(3, 1, gridspec_kw={'height_ratios': [0.25, 3, 1]})

	yhat = model.smooth(q_lem_x, ys[tr], input=us[tr])
	zhat = model.most_likely_states(q_lem_x, ys[tr], input=us[tr])
	a0.imshow(zhat[:,None].T,aspect="auto",vmin=0,vmax=2)
	a0.set_xticks([])
	a0.set_yticks([])
	a0.set_xlim([0,ys[tr].shape[0]-1])
	a0.axis("off")
	a1.plot(q_lem_x,'k',label="inferred")
	a1.fill_between(np.arange(np.shape(ys[tr])[0]),(q_lem_x-q_lem_std*2.0)[:,0], (q_lem_x+q_lem_std*2.0)[:,0], facecolor='k', alpha=0.3)
	for j in range(5):
		x_sample = q.sample_continuous_states()[tr]
		a1.plot(x_sample, 'k', alpha=0.3)
	a1.plot(np.array([0,np.shape(ys[tr])[0]]),np.array([1.0,1.0]),'k--', linewidth=1)
	a1.set_ylim([-0.2,1.1])
	a1.set_ylabel("$x$")
	a1.set_xlim([0,ys[tr].shape[0]-1])
	# a2.eventplot(np.where(ys[tr]>0)[0], linelengths=0.02, color='k', lineoffsets=1.1)
	a2.plot(ys[tr])
	a2.set_ylim([-0.1,max(ys[tr]+0.1)])
	a2.set_yticks([0.0,max(ys[tr])])
	a2.set_xlim([0,ys[tr].shape[0]-1])
	a2.set_ylabel("$y$")
	plt.tight_layout()
	plt.show()

plot_posterior_spikes(q_lem_lower,ramp_lower,ys,us,tr=0)
