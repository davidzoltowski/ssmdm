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
npr.seed(0)

# 1D Accumulator with Poisson observations
D = 1 		# number of accumulation dimensions
K = 3 		# number of discrete states
M = 1 		# number of input dimensions
N = 5		# number of observations
bin_size = 0.01
latent_acc = LatentAccumulation(N, K, D, M=M,
								transitions="ddmhard",
								emissions="poisson",
								emission_kwargs={"bin_size":bin_size})
# latent_acc.dynamics.Vs[0] = 0.05*np.ones((D,))
beta = 0.05*np.ones((D,))
log_sigmasq = np.log(1e-3)*np.ones((D,))
A = np.ones((D,D))
latent_acc.dynamics.params = (beta, log_sigmasq, A)
latent_acc.emissions.Cs[0] = 4 * npr.randn(N,D) + npr.choice([-15,15],(N,D))
latent_acc.emissions.ds[0] = 30 + 5 * npr.randn(N)

# loop through number of runs
num_runs = 2
init_params, final_params = [], []
q_params, elbos = [], []
all_xs, all_ys, all_us, all_zs, = [], [], [], []

for run in range(num_runs):

	# simulate data
	# Sample state trajectories
	T = 100 # number of time bins
	trial_time = 1.0 # trial length in seconds
	dt = 0.01 # bin size in seconds
	N_samples = 10

	# input statistics
	total_rate = 40 # the sum of the right and left poisson process rates is 40

	us = []
	zs = []
	xs = []
	ys = []

	for smpl in range(N_samples):

	    # randomly draw right and left rates
	    rate_r = np.random.randint(0,total_rate+1)
	    rate_l = total_rate - rate_r
	    rates = [rate_r,rate_l]

	    # generate binned right and left clicks
	    u = generate_clicks_D(rates,T=trial_time,dt=dt)

	    # input is sum of u_r and u_l
	    u = (1.0*np.array(u[1] - u[0]).T).reshape((T,1))
	    z, x, y = latent_acc.sample(T, input=u)

	    us.append(u)
	    zs.append(z)
	    xs.append(x)
	    ys.append(y)

	# store data
	all_xs += [xs]
	all_ys += [ys]
	all_zs += [zs]
	all_us += [us]

	# fit SLDS model to ys
	# initialize
	test_acc = LatentAccumulation(N, K, D, M=M, transitions="ddmhard",
								  emissions="poisson", emission_kwargs={"bin_size":bin_size})
	beta0 = np.array([0.0 + 0.1*npr.rand()])
	log_sigmasq0 = np.log([5e-4 + 4.5e-3*npr.rand()])
	A0 = np.ones((D,D))
	test_acc.dynamics.params = (beta0, log_sigmasq0, A0)
	test_acc.initialize(ys, inputs=us)
	init_params += [copy.deepcopy(test_acc.params)]

	# fit
	q_elbos, q_lem = test_acc.fit(ys, inputs=us, method="laplace_em",
								  variational_posterior="structured_meanfield",
								  num_iters=2, alpha=0.5, initialize=False,
								  variational_posterior_kwargs={"initial_variance":1e-4})

	print("Final params:", test_acc.params)
	final_params += [test_acc.params]
	q_params += [q_lem.params]
	elbos += [q_elbos]

np.savez("fit_1Daccumulator_results_200.npz", true_params=latent_acc.params,
	final_params=final_params, init_params=init_params,
	q_params=q_params, elbos=elbos, N_samples=N_samples,
	all_ys=all_ys, all_xs=all_xs, all_us=all_us, all_zs=all_zs)
