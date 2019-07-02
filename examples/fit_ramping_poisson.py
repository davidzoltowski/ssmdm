import autograd.numpy as np
import autograd.numpy.random as npr

from ssmdm.ramping import ObservedRamping, Ramping, simulate_ramping, RampingHard
from ssm.variational import SLDSMeanFieldVariationalPosterior, SLDSTriDiagVariationalPosterior, \
	SLDSStructuredMeanFieldVariationalPosterior

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
sns.set_context("talk")
from ssmdm.misc import smooth
from ssm.util import one_hot, softplus
npr.seed(0)

# setup parameters
numTrials = 200
bin_size = 0.01
# bin_size = 1.0
N = 1
beta = np.array([-0.005,-0.0025,0.0,0.01,0.02]) #+ 0.01*npr.randn(5)
# beta = np.array([-0.01,-0.005,0.0,0.001,0.001]) #+ 0.01*npr.randn(5)
# log_sigma_scale = np.log(1000)
log_sigma_scale = np.log(1e-3)
x0 = 0.5
latent_ddm = Ramping(N, M=5, link="softplus", beta = beta, log_sigma_scale=log_sigma_scale, x0=x0, bin_size=bin_size)
latent_ddm.emissions.Cs[0] = 40.0 + 3.0 * npr.randn(N,1)
print("True C: ", np.mean(latent_ddm.emissions.Cs[0]))
ys = []
xs = []
zs = []
us = []
cohs = []
numCoh = 5
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

# initialize
beta = np.array([-0.01,-0.005,0.0,0.01,0.02]) + 0.01*npr.randn(5)
# beta = np.array([-0.01,-0.005,0.0,0.001,0.001]) + 0.001*npr.randn(5)
x0 = 0.5 + 0.05 * npr.randn(1)
log_sigma_scale = np.log(5e-4+1.5e-3*npr.rand())
# beta = beta
# x0 = x0
# log_sigma_scale = log_sigma_scale

def initialize_ramp(ys,cohs, bin_size):
	coh5 = np.where(cohs==4)[0]
	y_end = np.array([y[-3:-1] for y in ys])
	y_end_5 = y_end[coh5]
	C = np.mean(y_end_5) / bin_size
	y0_mean = np.mean([y[0] for y in ys])
	x0 = y0_mean / C / bin_size
	return C, x0

C, x0 = initialize_ramp(ys, cohs, bin_size)

test_ddm = Ramping(N, M=5, link="softplus", beta=beta, log_sigma_scale=log_sigma_scale, x0=x0, bin_size=bin_size)
test_ddm.emissions.Cs[0] =  C
# test_ddm.emissions.Cs[0] =  latent_ddm.emissions.Cs[0] + 2.0 * npr.randn(N,1)
# test_ddm.emissions.Cs[0] =  latent_ddm.emissions.Cs[0]
# test_ddm.initialize(ys,inputs=us)

import copy
init_params = copy.deepcopy(test_ddm.params)

q_laplace_em = SLDSStructuredMeanFieldVariationalPosterior(test_ddm, ys, inputs=us, initial_variance=1e-7)

# for tr in range(numTrials):
# 	q_laplace_em.params[tr]["h"] *= 1e7
# 	# q_laplace_em.params[tr]["h"] = 1e7 * xs[tr]
# 	q_laplace_em.params[tr]["J_diag"] *= 1e7

num_iters = 25
Cs, dynamics_var, q_elbos = [], [], []
x_mean, x_sample_mean, qx_diff = [], [], []

current_mean = np.vstack(q_laplace_em.mean_continuous_states)
x_mean.append(np.mean(current_mean))
x_sample_mean.append(np.mean(np.vstack(q_laplace_em.sample_continuous_states())))
dynamics_var.append(test_ddm.dynamics.Sigmas[0][0][0])
Cs.append(test_ddm.emissions.Cs[0][0])

print("Initial C: ", test_ddm.emissions.Cs[0][0])
print("True x mean: ", np.mean(np.vstack(xs)))
print("X mean, X_sample mean:", x_mean[-1], x_sample_mean[-1])
params = []
params.append(test_ddm.params)
for iter in range(num_iters):
	print("Iteration: ", iter)
	q_lem_elbos = test_ddm.fit(q_laplace_em, ys, inputs=us, method="laplace_em", num_iters=1, initialize=False,
							   num_samples=3, alpha=0.5, emission_optimizer_maxiter=10, continuous_maxiter=50,
							   continuous_optimizer="newton", continuous_tolerance=1e-6,
							   parameters_update="mstep")
	# q_lem_elbos = test_ddm.fit(q_laplace_em, ys, inputs=us, method="laplace_em", num_iters=1, initialize=False,
							   # num_samples=1, alpha=0.75, emission_optimizer_maxiter=50, continuous_maxiter=100)
	q_elbos.append(q_lem_elbos[1])
	params.append(test_ddm.params)
	dynamics_var.append(test_ddm.dynamics.Sigmas[0][0][0])
	new_mean = np.vstack(q_laplace_em.mean_continuous_states)
	qx_diff.append(np.linalg.norm(current_mean - new_mean))
	current_mean = new_mean
	x_mean.append(np.mean(current_mean))
	x_sample_mean.append(np.mean(np.vstack(q_laplace_em.sample_continuous_states())))
	Cs.append(test_ddm.emissions.Cs[0][0])
	# print("Variance: ", dynamics_var[-1])
	# print("C: ", np.mean(test_ddm.emissions.Cs[0][0]))
	print("Params: ", params[-1])
	print("X mean, X_sample mean:", x_mean[-1], x_sample_mean[-1])

plt.figure()
plt.subplot(211)
# plt.title("Softplus, Binsize=0.01, N=1, Nsamp=5")
plt.plot(np.arange(1,len(q_elbos)+1),q_elbos)
plt.ylabel("ELBO")
# plt.subplot(312)
# plt.axhline(np.mean(latent_ddm.emissions.Cs[0]),color='k')
# plt.plot(Cs)
# # plt.axhline(1e-4,color='r')
# plt.ylabel("C")
plt.subplot(212)
plt.axhline(np.mean(np.vstack(xs)),color='k')
plt.plot(x_mean)
plt.ylabel("mean q(x)")
plt.xlabel("iteration")
plt.legend(["true","inferred"])
plt.tight_layout()
# plt.savefig("ramping_qx_fix_params_ns1_1e7.png")

plt.ion()
plt.figure()
plt.plot(q_elbos)
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

plot_trial(q_laplace_em, test_ddm, ys, us, method="lem", true_xs=xs, tr=0)
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
		# plt.plot((q_lem_z[:,None]-1)*0.5)

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
