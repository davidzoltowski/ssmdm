import autograd.numpy as np
import autograd.numpy.random as npr

from ssmdm.ramping import ObservedRamping, Ramping, simulate_ramping
from ssm.variational import SLDSMeanFieldVariationalPosterior, SLDSTriDiagVariationalPosterior, \
	SLDSStructuredMeanFieldVariationalPosterior

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
sns.set_context("talk")
from ssmdm.misc import smooth
from ssm.util import one_hot
npr.seed(0)

# setup parameters
numTrials = 200
bin_size = 0.01
N = 1
beta = np.array([-0.01,-0.005,0.0,0.01,0.02]) #+ 0.01*npr.randn(5)
log_sigma_scale = np.log(100)
x0 = 0.5
latent_ddm = Ramping(N, M=5, link="softplus", beta = beta, log_sigma_scale=log_sigma_scale, x0=x0, bin_size=bin_size)
latent_ddm.emissions.Cs[0] = 40.0 + 3.0 * npr.randn(N,1)
print("True C: ", np.mean(latent_ddm.emissions.Cs[0]))
ys = []
xs = []
zs = []
us = []
# sample from model
for tr in range(numTrials):

	u_tr = one_hot(npr.randint(5),5)
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

# initialize
beta = np.array([-0.01,-0.005,0.0,0.01,0.02]) + 0.01*npr.randn(5)
x0 = 0.5 + 0.05 * npr.randn(1)
log_sigma_scale = np.log(50 + 150*npr.rand())

test_ddm = Ramping(N, M=5, link="softplus", beta=beta, log_sigma_scale=log_sigma_scale, x0=x0, bin_size=bin_size)
test_ddm.emissions.Cs[0] =  latent_ddm.emissions.Cs[0] + 2.0 * npr.randn(N,1)
# test_ddm.initialize(ys,inputs=us)

import copy
init_params = copy.deepcopy(test_ddm.params)

q_laplace_em = SLDSStructuredMeanFieldVariationalPosterior(test_ddm, ys, inputs=us)

for tr in range(numTrials):
	# q_laplace_em.params[tr]["h"] *= 100.0
	q_laplace_em.params[tr]["h"] *= 1000.0 * xs[tr]
	q_laplace_em.params[tr]["J_diag"] *= 1000.0

num_iters = 25
Cs, dynamics_var, q_elbos = [], [], []
x_mean, x_sample_mean, qx_diff = [], [], []

current_mean = np.vstack(q_laplace_em.mean_continuous_states)
x_mean.append(np.mean(current_mean))
x_sample_mean.append(np.mean(np.vstack(q_laplace_em.sample_continuous_states())))
dynamics_var.append(test_ddm.dynamics.Sigmas[0][0][0])
Cs.append(test_ddm.emissions.Cs[0][0])

print("Initial C: ", test_ddm.emissions.Cs[0][0])
print("X mean, X_sample mean:", x_mean[-1], x_sample_mean[-1])

for iter in range(num_iters):
	print("Iteration: ", iter)
	q_lem_elbos = test_ddm.fit(q_laplace_em, ys, inputs=us, method="laplace_em", num_iters=1, initialize=False,
							   num_samples=5, alpha=0.5, emission_optimizer_maxiter=10, continuous_maxiter=50,
							   continuous_optimizer="newton", continuous_tolerance=1e-6,
							   parameters_update="mstep")
	# q_lem_elbos = test_ddm.fit(q_laplace_em, ys, inputs=us, method="laplace_em", num_iters=1, initialize=False,
							   # num_samples=1, alpha=0.75, emission_optimizer_maxiter=50, continuous_maxiter=100)
	q_elbos.append(q_lem_elbos[1])
	dynamics_var.append(test_ddm.dynamics.Sigmas[0][0][0])
	new_mean = np.vstack(q_laplace_em.mean_continuous_states)
	qx_diff.append(np.linalg.norm(current_mean - new_mean))
	current_mean = new_mean
	x_mean.append(np.mean(current_mean))
	x_sample_mean.append(np.mean(np.vstack(q_laplace_em.sample_continuous_states())))
	Cs.append(test_ddm.emissions.Cs[0][0])
	print("Variance: ", dynamics_var[-1])
	print("C: ", np.mean(test_ddm.emissions.Cs[0][0]))
	print("X mean, X_sample mean:", x_mean[-1], x_sample_mean[-1])

plt.figure(figsize=[6,6])
plt.subplot(311)
plt.title("Softplus, Binsize=0.01, N=1, Nsamp=5")
plt.plot(np.arange(1,len(q_elbos)+1),q_elbos)
plt.ylabel("ELBO")
plt.subplot(312)
plt.axhline(np.mean(latent_ddm.emissions.Cs[0]),color='k')
plt.plot(Cs)
# plt.axhline(1e-4,color='r')
plt.ylabel("C")
plt.subplot(313)
plt.axhline(np.mean(np.vstack(xs)),color='k')
plt.plot(x_mean)
plt.ylabel("mean q(x)")
plt.xlabel("iteration")
plt.legend(["true","inferred"])
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

plot_trial(q_laplace_em, test_ddm, ys, us, method="lem", true_xs=xs, tr=0)
plt.show()
