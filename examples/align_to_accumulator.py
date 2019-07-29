import autograd.numpy as np
import autograd.numpy.random as npr

import matplotlib.pyplot as plt

from ssmdm.accumulation import Accumulation, LatentAccumulation
from ssmdm.misc import smooth, generate_clicks, generate_clicks_D, factor_analysis
from ssm.util import softplus
from ssm.preprocessing import factor_analysis_with_imputation
from tqdm.auto import trange

# for initialization
from ssm.optimizers import adam_step, rmsprop_step, sgd_step, convex_combination
from autograd.scipy.misc import logsumexp
from autograd.tracer import getval
from autograd.misc import flatten
from autograd import value_and_grad

# create 1D accumulator
D = 1 		# number of accumulation dimensions
K = 3 		# number of discrete states
M = 1 		# number of input dimensions
acc = Accumulation(K, D, M=M, transitions="ddmhard")
acc.observations.Vs[0] = 0.075*np.ones((D,))
# acc.observations._log_sigmasq[0] = np.log(2e-3)*np.ones((D,))


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

for smpl in range(N_samples):

    # randomly draw right and left rates
    rate_r = np.random.randint(0,total_rate+1)
    rate_l = total_rate - rate_r
    rates = [rate_r,rate_l]

    # generate binned right and left clicks
    u = generate_clicks_D(rates,T=trial_time,dt=dt)

    # input is sum of u_r and u_l
    u = (1.0*np.array(u[1] - u[0]).T).reshape((T,1))
    z, x = acc.sample(T, input=u)

    us.append(u)
    zs.append(z)
    xs.append(x)

# 1. Generate data R x + r + noise. Can you recover R, r w/ fit parameters?
R = 2 * npr.randn(D,D)
r = 1 * npr.randn(D,1)
xhats = [np.dot(x, R.T) + r + 0.01 * npr.randn(*x.shape) for x in xs]

# plot data and transformed data
plt.ion()
plt.figure()
for tr in range(5):
    plt.plot(xs[tr],'k',alpha=0.5)
    plt.plot(xhats[tr],'r',alpha=0.5)

# define objective
r = -1.0 * np.mean([x[0] for x in xhats])
def objective(params, itr):
    T = sum([x.shape[0] for x in xhats])
    new_datas = [np.dot(x,params[0].T)+params[1] for x in xhats]
    # new_datas = [np.dot(x,params[0].T) for x in xhats]
    # new_datas = [np.dot(x+r,params[0].T) for x in xhats]
    acc2 = Accumulation(K, D, M=M, transitions="ddm")
    # acc2.observations._log_sigmasq[0] = np.log(1e-2)*np.ones((D,))
    acc2.observations.betas = np.array([0.075])
    # obj = Accumulation(K, D, M=M, transitions="ddmhard").log_likelihood(new_datas,inputs=us)
    obj = acc2.log_likelihood(new_datas,inputs=us)
    return -obj / T

# initialize params
# r_init
# R_init = 1 * npr.randn(D,D)
# R_init = -2*np.ones((1,1))
# R_init = -5.reshape((1,1))
# r_init = 0.1 * npr.randn(D,1)
max_xhat = np.max(np.vstack([x+r for x in xhats]))
R_init = 1.25 / max_xhat
r_init = np.copy(r)
# params = [R_init]
params = [R_init, r_init]

# optimize marginal likelihood of p(Rx+r|theta_arhmm), with respect to R, r
num_iters = 100
state = None
lls = [-objective(params, 0) * T]
pbar = trange(num_iters)
pbar.set_description("Epoch {} Itr {} LP: {:.1f}".format(0, 0, lls[-1]))
for itr in pbar:
    params, val, g, state = adam_step(value_and_grad(objective), params, itr, state, step_size=0.05)
    lls.append(-val * T)
    pbar.set_description("LP: {:.1f}".format(lls[-1]))
    pbar.update(1)

x_transformed = [np.dot(x, params[0].T) + params[1] for x in xhats]
# x_transformed2 = [x * 1.0 / R) + params[1] for x in xhats]
# x_transformed = [np.dot(x+r, params[0].T) for x in xhats]
# x_transformed = [np.dot(x, params[0].T) for x in xhats]
# plot a transformed trial

tr = 0
tr += 1
plt.figure()
plt.plot(xs[tr],'k')
# plt.plot(xhats[tr],'r')
plt.plot(x_transformed[tr],'b')

# try with simulating neural activity, then doing e.g. PCA or FA?

# generate neural data, shifted in time (with random noise before hand?)
# C = 1 * npr.randn(N,D) + npr.choice([-20,20],(N,D))
# 30 + 3.0 * npr.randn(N)
