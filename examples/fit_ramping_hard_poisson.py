import autograd.numpy as np
import autograd.numpy.random as npr

from ssmdm.ramping import ObservedRamping, Ramping, simulate_ramping, RampingHard, RampingLowerHard

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
sns.set_context("talk")
from ssmdm.misc import smooth
from ssm.util import one_hot, softplus
npr.seed(2)

# setup parameters
# numTrials = 375
numTrials = 200
bin_size = 0.01
# bin_size = 1.0
N = 25
# beta = np.array([-0.005,-0.0025,0.0,0.01,0.015]) #+ 0.01*npr.randn(5)
beta = np.array([-0.01,-0.0025,0.0,0.01,0.015]) #+ 0.01*npr.randn(5)
log_sigma_scale = np.log(5e-4)
x0 = 0.5
latent_ddm = RampingHard(N, M=5, link="softplus", beta = beta, log_sigma_scale=log_sigma_scale, x0=x0, bin_size=bin_size)
# latent_ddm = RampingLowerHard(N, M=5, link="softplus", beta = beta, log_sigma_scale=log_sigma_scale, x0=x0, bin_size=bin_size)
latent_ddm.emissions.Cs[0] = 40.0 + 3.0 * npr.randn(N,1)
print("True C: ", np.mean(latent_ddm.emissions.Cs[0]))
ys = []
xs = []
zs = []
us = []
cohs = []
numCoh = 5
M = 5
# sample from model
for tr in range(numTrials):

	coh = npr.randint(numCoh)
	u_tr = one_hot(coh,numCoh)
	tr_length = npr.randint(50)+50
	u = u_tr * np.ones((tr_length,1))
	# u = all_us[tr]
	T = np.shape(u)[0]

	# sample from Observed ramping and them map to observations? fit that instead???
	z, x, y = latent_ddm.sample(T, input=u)

	zs.append(z)
	xs.append(x)
	ys.append(y)
	us.append(u)
	cohs.append(coh)

cohs = np.array(cohs)

sns.set_style("ticks")
plt.ion()
plt.figure()
colors=['r','m','k','c','b']
# for tr in range(numTrials):
for tr in np.arange(1,50):
	color=colors[cohs[tr]]
	if cohs[tr] == 4 or cohs[tr] == 0:
		z_switch = np.where(np.abs(zs[tr])>0)[0]
		z_switch = z_switch[0] if z_switch.shape[0] > 0 else xs[tr].shape[0]
		plt.plot(np.arange(0,z_switch),xs[tr][:z_switch],'k',alpha=0.75)
		plt.plot(np.arange(z_switch-1,xs[tr].shape[0]),xs[tr][z_switch-1:],color,alpha=0.75)
	# plt.plot(xs[tr],color,alpha=0.6)
plt.xlabel("time")
plt.ylabel("x")
sns.despine()
plt.tight_layout()

sns.set_style("ticks")
plt.figure()
plt.plot([0,100],[0,0],'k',linewidth=1,alpha=0.75)
for tr in np.arange(5,26):
	color=colors[cohs[tr]]
	if cohs[tr] == 4 or cohs[tr] == 0:
		y_trial = latent_ddm.emissions.mean(latent_ddm.emissions.forward(xs[tr],input=us[tr],tag=None)[:,0,:])[:,0] / bin_size
	# plt.plot(latent_ddm.emissions.forward(xs[tr],input=us[tr],tag=None)[:,0,:][:,0],alpha=0.75)
		plt.plot(y_trial,alpha=0.75,color=color)
plt.ylim([-1.0,45])
plt.xlabel("time")
plt.ylabel("x")
sns.despine()
plt.tight_layout()


# initialize
beta = np.array([-0.005,-0.0025,0.0,0.01,0.02]) + 0.01*npr.randn(5)
# beta = np.array([-0.01,-0.005,0.0,0.001,0.001]) + 0.001*npr.randn(5)
log_sigma_scale = np.log(5e-4+1.5e-3*npr.rand())
# log_sigma_scale = np.log(3e-3)

def initialize_ramp(ys,cohs, bin_size):
	coh5 = np.where(cohs==4)[0]
	# coh5 = np.concatenate((np.where(cohs==4)[0],np.where(cohs==3)[0]))
	y_end = np.array([y[-10:] for y in ys])
	y_end_5 = y_end[coh5]
	C = np.mean(y_end_5,axis=(0,1)) / bin_size
	y0_mean = np.mean([y[0:2] for y in ys],axis=0) / bin_size
	x0 = np.mean(np.divide(y0_mean, C))
	return C, x0

C, x0 = initialize_ramp(ys, cohs, bin_size)

test_ddm = RampingHard(N, M=5, link="softplus", beta=beta, log_sigma_scale=log_sigma_scale, x0=x0, bin_size=bin_size)
test_ddm.emissions.Cs[0] =  C.reshape((N,1))
# test_ddm.emissions.Cs[0] =  latent_ddm.emissions.Cs[0] + 2.0 * npr.randn(N,1)
# test_ddm.emissions.Cs[0] =  latent_ddm.emissions.Cs[0]
# test_ddm.initialize(ys,inputs=us)

import copy
init_params = copy.deepcopy(test_ddm.params)

# initialize variational posterior
init_var = 1e-5
_, q_lem = test_ddm.fit(ys, inputs=us, method="laplace_em",
								  variational_posterior="structured_meanfield",
								  variational_posterior_kwargs={"initial_variance":init_var},
								  num_iters=0, initialize=False)
# for tr in range(numTrials):
# 	q_lem.params[tr]["h"] = xs[tr] / init_var
q_lem_elbos, q_lem = test_ddm.fit(ys, inputs=us, method="laplace_em",
								  variational_posterior=q_lem,
								  alpha=0.5,
								  num_iters=5, initialize=False, num_samples=1)
q_lem_elbos2, q_lem = test_ddm.fit(ys, inputs=us, method="laplace_em",
								  variational_posterior=q_lem,
								  alpha=0.5,
								  num_iters=5, initialize=False, num_samples=1)
q_lem_elbos3, q_lem = test_ddm.fit(ys, inputs=us, method="laplace_em",
								  variational_posterior=q_lem,
								  alpha=0.5,
								  num_iters=10, initialize=False)
q_lem_elbos4, q_lem = test_ddm.fit(ys, inputs=us, method="laplace_em",
								  variational_posterior=q_lem,
								  alpha=0.5,
								  num_iters=10, initialize=False)
q_lem_elbos6, q_lem = test_ddm.fit(ys, inputs=us, method="laplace_em",
								  variational_posterior=q_lem,
								  alpha=0.5,
								  num_iters=25, initialize=False)

plt.ion()
plt.figure()
plt.plot(q_lem_elbos[1:])
plt.plot()
plt.ylabel("ELBO")
plt.xlabel("iteration")
plt.tight_layout()

plt.figure()
plt.imshow(np.concatenate( (latent_ddm.emissions.Cs[0], init_params[-1][0], test_ddm.emissions.Cs[0]),axis=1),aspect="auto")
plt.tight_layout()
plt.colorbar()

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
	for i in range(5):
		q_sample = q.sample_continuous_states()[tr]
		plt.plot(q_sample,'b',alpha=0.5)
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
		plt.plot(softplus(xs[tr]*latent_ddm.emissions.Cs[0]))
		plt.plot(yhat / bin_size,'b');
		plt.plot(np.array([0,np.shape(yhat)[0]]),np.array([max_rate,max_rate]))
		plt.eventplot(np.where(ys[tr]>0)[0], linelengths=2)
		plt.ylabel("$y$")
		plt.xlabel("time bin")
		plt.legend(["true smoothed", "true rate","inferred rate"])

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

plot_trial(q_lem, test_ddm, ys, us, method="lem", true_xs=xs, tr=0)
plt.show()


def plot_trials(q,model,ys,us,method="lem",true_xs=None):
	numTrials = len(ys)
	rand_trials = npr.choice(range(numTrials),3)
	plt.figure(figsize=[12,12])
	for idx, tr in enumerate(rand_trials):
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
		# plt.subplot(121)
		plt.subplot(5,2,(idx*2)+1)
		if true_xs is not None:
			plt.plot(true_xs[tr],'k',label="true")
		plt.plot(q_lem_x,'b',label="inferred")
		for i in range(5):
			q_sample = q.sample_continuous_states()[tr]
			plt.plot(q_sample,'b',alpha=0.5)
		plt.fill_between(np.arange(np.shape(ys[tr])[0]),(q_lem_x-q_lem_std*2.0)[:,0], (q_lem_x+q_lem_std*2.0)[:,0], facecolor='b', alpha=0.3)
		plt.plot(np.array([0,np.shape(yhat)[0]]),np.array([1.0,1.0]),'r--')
		plt.ylim([-0.2,1.2])
		if idx == 2:
			plt.legend()
			plt.xlabel("time bin")
			plt.ylabel("$x$")

		if np.shape(yhat)[1] < 6:
			plt.subplot(5,2,(idx*2)+2)
			plt.plot(smooth(ys[tr],10) / bin_size,'k');
			plt.plot(softplus(xs[tr]*latent_ddm.emissions.Cs[0]))
			plt.plot(yhat / bin_size,'b');
			plt.plot(np.array([0,np.shape(yhat)[0]]),np.array([max_rate,max_rate]))
			plt.eventplot(np.where(ys[tr]>0)[0], linelengths=2)
			if idx == 2:
				plt.ylabel("$y$")
				plt.xlabel("time bin")
				plt.legend(["true smoothed", "true rate","inferred rate"])

		else:
			plt.subplot(5,2,(idx*2)+2)
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
			# plt.ylabel("$y$")
			# plt.xlabel("time bin")
			if i == 0:
				plt.legend(["true","inferred"])

	plt.tight_layout()
	plt.show()

# tr = 0
# tr += 1
# q_z = test_ddm.most_likely_states(q_laplace_em.mean_continuous_states[tr], ys[tr])
# plt.figure()
# plt.plot(zs[tr],'k')
# plt.plot(q_z,'r--')

T = 100
num_sims = 1000
from ssm.util import one_hot
sim_zs, sim_xs, sim_ys, sim_us, sim_cohs = [], [], [], [], []
numCoh = np.copy(M)
for iter in range(num_sims):

	coh = npr.randint(numCoh)
	u_tr = one_hot(coh,numCoh)
	tr_length = np.copy(T)
	u = u_tr * np.ones((tr_length,1))

	# sample from Observed ramping and them map to observations? fit that instead???
	z, x, y = test_ddm.sample(T, input=u)

	sim_zs.append(z)
	sim_xs.append(x)
	y = test_ddm.emissions.mean(test_ddm.emissions.forward(x, input=u, tag=None))
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

	num_coh = np.shape(us[0])[0]
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

	# plt.figure()
	plt.plot(smooth(psth[0][:,None],5)/0.01,linestyle=linestyle,color=[1.0,0.0,0.0],alpha=0.8)
	plt.plot(smooth(psth[1][:,None],5)/0.01,linestyle=linestyle,color=[1.0,0.0,0.5],alpha=0.8)
	plt.plot(smooth(psth[2][:,None],5)/0.01,linestyle=linestyle,color=[0.0,0.0,0.0],alpha=0.8)
	plt.plot(smooth(psth[3][:,None],5)/0.01,linestyle=linestyle,color=[0.5,0.0,1.0],alpha=0.8)
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
