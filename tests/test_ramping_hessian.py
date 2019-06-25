import autograd.numpy as np
import autograd.numpy.random as npr

from ssmdm.ramping import ObservedRamping, Ramping
from ssm.variational import SLDSMeanFieldVariationalPosterior, SLDSTriDiagVariationalPosterior, \
	SLDSStructuredMeanFieldVariationalPosterior

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
sns.set_context("talk")
from ssmdm.misc import smooth

# import neuron 1
d = np.load("../analyses/neuron1.npz")
us = list(d["us"])
ys = list(d["ys"])
all_ys = [y.astype(int) for y in ys]
all_us = us
num_trials = len(us)

# setup parameters
numTrials = 100
bin_size = 0.01
# bin_size = 1.0
N = 1
beta = np.array([-0.01,-0.005,0.0,0.01,0.02])
latent_ddm = Ramping(N, M=5, link="softplus", beta = beta, log_sigma_scale=np.log(30), x0=0.5, bin_size=bin_size)
latent_ddm.emissions.Cs[0] = 40.0 + 3.0 * npr.randn(N,1)

ys = []
xs = []
zs = []
us = []
# sample from model
for tr in range(numTrials):

	u = all_us[tr]
	T = np.shape(u)[0]
	z, x, y = latent_ddm.sample(T, input=u)

	zs.append(z)
	xs.append(x)
	ys.append(y)
	us.append(u)

q_laplace_em = SLDSStructuredMeanFieldVariationalPosterior(latent_ddm, ys, inputs=us)
q_lem_elbos = latent_ddm.fit(q_laplace_em, ys, inputs=us, method="laplace_em", num_iters=1, initialize=False,
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
    log_pi0 = latent_ddm.init_state_distn.log_initial_state_distn(x, input, x_mask, tag)
    log_Ps = latent_ddm.transitions.log_transition_matrices(x, input, x_mask, tag)
    log_likes = latent_ddm.dynamics.log_likelihoods(x, input, x_mask, tag)
    log_likes += latent_ddm.emissions.log_likelihoods(data, input, mask, tag, x)

    # Compute the expected log probability
    elp = np.sum(Ez[0] * log_pi0)
    elp += np.sum(Ezzp1 * log_Ps)
    elp += np.sum(Ez * log_likes)
    # assert np.all(np.isfinite(elp))

    return -1 * elp / scale

def hessian_neg_expected_log_joint(x, Ez, Ezzp1, scale=1):
    T, D = np.shape(x)
    x_mask = np.ones((T, D), dtype=bool)
    hessian_diag, hessian_lower_diag = latent_ddm.dynamics.hessian_expected_log_dynamics_prob(Ez, x, input, x_mask, tag)
    hessian_diag[:-1] += latent_ddm.transitions.hessian_expected_log_trans_prob(x, input, x_mask, tag, Ezzp1)
    hessian_diag += latent_ddm.emissions.hessian_log_emissions_prob(data, input, mask, tag, x)

    # The Hessian of the log probability should be *negative* definite since we are *maximizing* it.
    # hessian_diag -= 1e-8 * np.eye(D)

    # Return the scaled negative hessian, which is positive definite
    return -1 * hessian_diag / scale, -1 * hessian_lower_diag / scale

from autograd import hessian
from ssm.primitives import blocks_to_full

hess = hessian(neg_expected_log_joint)
H_autograd = hess(x, Ez, Ezzp1).reshape((T,T))

H_diag, H_lower_diag = hessian_neg_expected_log_joint(x, Ez, Ezzp1)
H = blocks_to_full(H_diag, H_lower_diag)

assert np.allclose(H,H_autograd)
