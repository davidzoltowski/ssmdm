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
# npr.seed(123)
npr.seed(123456)

# 1D Accumulator with Poisson observations
D = 1 		# number of accumulation dimensions
K = 3 		# number of discrete states
M = 1 		# number of input dimensions
N = 10		# number of observations
bin_size = 0.01
latent_acc = LatentAccumulation(N, K, D, M=M,
								transitions="ddmhard",
								emissions="poisson",
								emission_kwargs={"bin_size":bin_size})
# latent_acc.dynamics.Vs[0] = 0.05*np.ones((D,))
beta = 0.075*np.ones((D,))
log_sigmasq = np.log(2e-3)*np.ones((D,))
A = np.ones((D,D))
latent_acc.dynamics.params = (beta, log_sigmasq, A)
latent_acc.emissions.Cs[0] = 4 * npr.randn(N,D) + npr.choice([-15,15],(N,D))
latent_acc.emissions.ds[0] = 40 + 5 * npr.randn(N)

# simulate data
# Sample state trajectories
T = 100 # number of time bins
trial_time = 1.0 # trial length in seconds
dt = 0.01 # bin size in seconds
N_samples = 200

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

init_var = 1e-5
q_elbos, q_lem = latent_acc.fit(ys, inputs=us, method="laplace_em",
						variational_posterior="structured_meanfield",
						num_iters=1, initialize=False,
						variational_posterior_kwargs={"initial_variance":init_var})

# test hessian
tr = 1
discrete_expectations = q_lem.mean_discrete_states
Ez, Ezzp1, _ = discrete_expectations[tr]
input = us[tr]
masks = [np.ones_like(y, dtype=bool) for y in ys]
tags = [None] * len(ys)
mask = masks[tr]
tag = tags[tr]
data = ys[tr]
x = xs[tr]
T = data.shape[0]

def neg_expected_log_joint(x, Ez, Ezzp1, scale=1):
    # The "mask" for x is all ones
    x_mask = np.ones_like(x, dtype=bool)
    log_pi0 = latent_acc.init_state_distn.log_initial_state_distn(x, input, x_mask, tag)
    log_Ps = latent_acc.transitions.log_transition_matrices(x, input, x_mask, tag)
    log_likes = latent_acc.dynamics.log_likelihoods(x, input, x_mask, tag)
    log_likes += latent_acc.emissions.log_likelihoods(data, input, mask, tag, x)

    # Compute the expected log probability
    elp = np.sum(Ez[0] * log_pi0)
    elp += np.sum(Ezzp1 * log_Ps)
    elp += np.sum(Ez * log_likes)
    # assert np.all(np.isfinite(elp))

    return -1 * elp / scale

def hessian_neg_expected_log_joint(x, Ez, Ezzp1, scale=1):
    T, D = np.shape(x)
    x_mask = np.ones((T, D), dtype=bool)
    hessian_diag, hessian_lower_diag = latent_acc.dynamics.hessian_expected_log_dynamics_prob(Ez, x, input, x_mask, tag)
    hessian_diag[:-1] += latent_acc.transitions.hessian_expected_log_trans_prob(x, input, x_mask, tag, Ezzp1)
    hessian_diag += latent_acc.emissions.hessian_log_emissions_prob(data, input, mask, tag, x)

    # The Hessian of the log probability should be *negative* definite since we are *maximizing* it.
    # hessian_diag -= 1e-8 * np.eye(D)

    # Return the scaled negative hessian, which is positive definite
    return -1 * hessian_diag / scale, -1 * hessian_lower_diag / scale

from autograd import hessian
from ssm.primitives import blocks_to_full

hess = hessian(neg_expected_log_joint)
H_autograd = hess(x, Ez, Ezzp1).reshape((T*D,T*D))

H_diag, H_lower_diag = hessian_neg_expected_log_joint(x, Ez, Ezzp1)
H = blocks_to_full(H_diag, H_lower_diag)

assert np.allclose(H,H_autograd)
