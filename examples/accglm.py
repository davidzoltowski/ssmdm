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
from scipy.linalg import hankel

# create 1D accumulator
D = 1 		# number of accumulation dimensions
K = 3 		# number of discrete states
M = 10 		# number of input dimensions
acc = Accumulation(K, D, M=M, transitions="ddmhard", observations="accglm")
V0 = np.zeros((D,M))
V0[0,-1] = 0.075
acc.observations.params = (V0, acc.observations.params[1], acc.observations.params[2])
acc.observations._log_sigmasq[0] = np.log(1e-3)*np.ones((D,))
acc.observations._log_sigmasq[1] = np.log(1e-3)*np.ones((D,))
acc.observations._log_sigmasq[2] = np.log(1e-3)*np.ones((D,))
acc.observations.As[1] = 0.985
acc.observations.As[2] = 0.985



# Sample state trajectories
T = 100 # number of time bins
trial_time = 1.0 # trial length in seconds
dt = 0.01 # bin size in seconds
N_samples = 50

# input statistics
total_rate = 40 # the sum of the right and left poisson process rates is 40

us = []
inputs = []
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
    u[-50:,:] *= 0.0
    
    # first column
    c = np.vstack((np.zeros((M-1,1)),u[:-M+1,:]))
    # last row
    r = u[-M:,:]
    U = hankel(c,r)

    z, x = acc.sample(T, input=U)

    us.append(u)
    inputs.append(U)
    zs.append(z)
    xs.append(x)

plt.ion()

tr = 0
tr += 1
plt.figure()
plt.plot(us[tr]*0.1,'k')
plt.plot(xs[tr],'b')
plt.xlabel("time")
plt.ylabel("x")
plt.tight_layout()

test_acc = Accumulation(K, D, M=M, transitions="ddmhard", observations="accglm")
test_acc.fit(xs, inputs=inputs)

plt.figure()
plt.plot(acc.observations._V0.ravel(),'k')
plt.plot(test_acc.observations._V0.ravel(),'r--')
