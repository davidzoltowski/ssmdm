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

# fit model
# init_var = 1e-2
# q_elbos_lem, q_lem = test_acc.fit(ys, inputs=us, method="laplace_em",
# 						variational_posterior="structured_meanfield",
# 						num_iters=20, alpha=0.5, initialize=False,
# 						variational_posterior_kwargs={"initial_variance":init_var})
