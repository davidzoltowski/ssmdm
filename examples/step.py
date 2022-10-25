import autograd.numpy as np
import autograd.numpy.random as npr
npr.seed(0)

import matplotlib
import matplotlib.pyplot as plt

import ssm
from ssm.util import find_permutation

from ssmdm.step import Step 

# Set the parameters of the HMM
T = 100     # number of time bins
K = 2       # number of discrete states
D = 1       # number of observed dimensions
num_trials = 100 

# Make an HMM with the true parameters
# true_hmm = Step(K, D, M=0, 
#                 observations="trial_poisson", observation_kwargs={"num_trials":num_trials})
true_hmm = Step(K, D, M=0, 
                transitions="trial_step", transition_kwargs={"num_trials":num_trials},
                observations="trial_poisson", observation_kwargs={"num_trials":num_trials})
# update params
state_rates = npr.randn(K, D, num_trials)
state_rates[0] += 10
state_rates[1] += 20
true_hmm.observations.log_lambdas = np.log(state_rates)

outs = [true_hmm.sample(T, tag=tr) for tr in range(num_trials)]
zs = [out[0] for out in outs]
ys = [out[1] for out in outs]

# visualize trials
tr = npr.randint(num_trials)
tr = 26
plt.figure()
plt.subplot(211)
plt.plot(zs[tr])
plt.subplot(212)
plt.plot(ys[tr])

# fit test hmm
test_hmm = Step(K, D, M=0, 
                transitions="trial_step", transition_kwargs={"num_trials":num_trials},
                observations="trial_poisson", observation_kwargs={"num_trials":num_trials})
tags = list(np.arange(num_trials))
# initialize log lambdas
rate_init = np.array([np.mean(y[:10], axis=0) for y in ys])
rate_end = np.array([np.mean(y[-10:], axis=0) for y in ys])
lambdas = np.hstack((rate_init, rate_end)).T[:, None, :]
test_hmm.observations.log_lambdas = np.log(lambdas)
test_hmm.fit(ys, tags=tags)

test_step = Step(K, D, M=0, observations="poisson")
step_lambdas = np.array([np.mean(rate_init), np.mean(rate_end)])[:, None]
test_step.observations.log_lambdas = step_lambdas
test_step.fit(ys)


# z, y = true_hmm.sample(T)
# z_test, y_test = true_hmm.sample(T)
# true_ll = true_hmm.log_probability(y)
plt.figure()
plt.subplot(121)
plt.imshow(np.exp(true_hmm.observations.log_lambdas[:, 0, :]).T, aspect="auto", vmin=5.0, vmax=25.0)
plt.title("true")
plt.ylabel("trial")
plt.xticks([])
plt.colorbar()
plt.subplot(122)
plt.imshow(np.exp(test_hmm.observations.log_lambdas[:, 0, :]).T, aspect="auto", vmin=5.0, vmax=25.0)
plt.title("inferred")
plt.xticks([])
plt.colorbar()

tr = 15
z_test = test_hmm.most_likely_states(ys[tr], tag=tags[tr])
z_test2 = test_step.most_likely_states(ys[tr])
y_smooth = test_hmm.smooth(ys[tr], tag=tags[tr])
plt.figure()
plt.subplot(211)
plt.plot(zs[tr])
plt.plot(z_test)
plt.plot(z_test2)
plt.subplot(212)
plt.plot(ys[tr])
plt.plot(y_smooth)
