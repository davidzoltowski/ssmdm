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

# create 2D Poisson accumulator
betas = np.array([1.0,1.0])
a_diag = np.ones((3,2,1))
log_sigma_scale = np.log(10)*np.ones(2,)
bin_size = 0.01
N = 5
D = 2
latent_acc = LatentAccumulator2DPoisson(N, link="softplus", bin_size=bin_size, log_sigma_scale=log_sigma_scale, betas=betas, a_diag=a_diag)
latent_acc.emissions.Cs[0] = 15 * npr.randn(N,D)
latent_acc.emissions.ds[0] = 40 + 3 * npr.randn(N)

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
    rate_l = total_rate - rate_r

    # generate binned right and left clicks
    u_r, u_l = generate_clicks(T=trial_time,dt=dt,rate_r=rate_r,rate_l=rate_l)

    # input is sum of u_r and u_l
    u = 0.1*np.array([u_r,u_l]).T
    z, x, y = latent_acc.sample(T, input=u)

    us.append(u)
    zs.append(z)
    xs.append(x)
    ys.append(y)

q_laplace_em = SLDSStructuredMeanFieldVariationalPosterior(latent_acc, ys, inputs=us)
for tr in range(N_samples):
    q_laplace_em._params[tr]["h"] *= 100
    q_laplace_em._params[tr]["J_diag"] *= 100

q_lem_elbos = latent_acc.fit(q_laplace_em, ys, inputs=us, method="laplace_em", num_iters=1, initialize=False,
						   num_samples=1, alpha=0.5, emission_optimizer_maxiter=5, continuous_maxiter=50)


# test hessian
tr = 12
discrete_expectations = q_laplace_em.mean_discrete_states
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
