import autograd.numpy as np
import autograd.numpy.random as npr

import ssm
from ssm.hmm import HMM
from ssm.lds import SLDS

from ssmdm.misc import smooth

from ssm.util import random_rotation, ensure_args_are_lists
from ssm.observations import Observations, AutoRegressiveDiagonalNoiseObservations
from ssm.transitions import Transitions, RecurrentOnlyTransitions, RecurrentTransitions
from ssm.init_state_distns import InitialStateDistribution
from ssm.emissions import _LinearEmissions, GaussianEmissions, PoissonEmissions
from ssm.preprocessing import factor_analysis_with_imputation
import ssm.stats as stats

# preprocessing
# for initialization
from tqdm.auto import trange
from ssm.optimizers import adam_step, rmsprop_step, sgd_step, convex_combination
from autograd.scipy.misc import logsumexp
from autograd.tracer import getval
from autograd.misc import flatten
from autograd import value_and_grad
from scipy.stats import gamma, norm

from ssm.util import ensure_args_are_lists, \
    logistic, logit, softplus, inv_softplus
from ssm.preprocessing import interpolate_data, pca_with_imputation

class RampingTransitions(RecurrentOnlyTransitions):
    def __init__(self, K, D, M=0, scale=200):
        assert K == 2
        assert D == 1
        # assert M == 1
        super(RampingTransitions, self).__init__(K, D, M)

        # Parameters linking past observations to state distribution
        self.Ws = np.zeros((K, M))
        self.Rs = np.array([0, scale]).reshape((K, 1))
        self.r = np.array([0, -scale])

    @property
    def params(self):
        return ()

    @params.setter
    def params(self, value):
        pass

    def initialize(self, datas, inputs=None, masks=None, tags=None):
        pass

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        pass


class RampingHardTransitions(RecurrentTransitions):
    def __init__(self, K, D, M=0, scale=10):
        assert K == 2
        assert D == 1
        # assert M == 1
        super(RampingHardTransitions, self).__init__(K, D, M)

        # Parameters linking past observations to state distribution
        self.log_Ps = np.array([[0, -scale], [-10, 10]])
        self.Ws = np.zeros((K, M))
        self.Rs = np.array([0, scale*0.75]).reshape((K, D))

    @property
    def params(self):
        return ()

    @params.setter
    def params(self, value):
        pass

    def initialize(self, datas, inputs=None, masks=None, tags=None):
        pass

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        pass


class RampingObservations(AutoRegressiveDiagonalNoiseObservations):
    def __init__(self, K, D=1, M=5, lags=1, beta=np.linspace(-0.02,0.02,5), log_sigma_scale=np.log(100), x0=0.4):
        assert K == 2
        assert D == 1
        assert M == 5
        super(RampingObservations, self).__init__(K, D, M)

        # The only free parameters of the DDM are the ramp rate...
        self.beta = beta
        self.log_sigma_scale = log_sigma_scale
        self.x0 = x0 # mu init

        # and the noise variances, which are initialized in the AR constructor
        # self._log_sigmasq[0] = np.log(1e-5) + np.exp(self.log_sigma_scale)
        self._log_sigmasq[0] = np.log(1e-5) + self.log_sigma_scale
        self._log_sigmasq[1] = np.log(1e-5)

        # set init params
        self.mu_init = self.x0 * np.ones((2,1))
        # self._log_sigmasq_init[0] = np.log(1e-5) + np.exp(self.log_sigma_scale)
        self._log_sigmasq_init[0] = np.log(1e-5) + self.log_sigma_scale
        self._log_sigmasq_init[1] = np.log(1e-5)

        # They only differ in their input
        self.Vs[0] = self.beta    # ramp
        self.Vs[1] = 0            # bound

        # Set the remaining parameters to fixed values
        self._As = np.ones((K, 1, 1))
        self.bs = np.zeros((K, 1))

        # self.l2_penalty_V = 1.0 / (0.1**2)

    @property
    def params(self):
        return self.beta, self.log_sigma_scale, self.x0

    @params.setter
    def params(self, value):
        self.beta, self.log_sigma_scale, self.x0 = value
        mask = np.reshape(np.array([1, 0]), (2, 1, 1))
        self.Vs = mask * self.beta
        sig_mask = np.reshape(np.array([1, 0]), (2, 1))
        # self._log_sigmasq = np.log(1e-5)*np.ones((2,1)) + sig_mask * np.exp(self.log_sigma_scale)
        # self._log_sigmasq_init = np.log(1e-5)*np.ones((2,1)) + sig_mask * np.exp(self.log_sigma_scale)
        self._log_sigmasq = np.log(1e-5)*np.ones((2,1)) + sig_mask * self.log_sigma_scale
        self._log_sigmasq_init = np.log(1e-5)*np.ones((2,1)) + sig_mask * self.log_sigma_scale
        self.mu_init = self.x0 * np.ones((2,1))

    def log_prior(self):
        beta_mean = np.zeros(np.shape(self.beta)[0])
        beta_cov = (0.1**2)*np.eye(np.shape(self.beta)[0])
        return np.sum(stats.multivariate_normal_logpdf(np.array([self.beta]), beta_mean, beta_cov)) \
                + np.sum(-0.5 * (self.x0 - 0.5)**2 / 1.0)

    def initialize(self, datas, inputs=None, masks=None, tags=None):
        pass

    def m_step(self, expectations, datas, inputs, masks, tags, optimizer="lbfgs", **kwargs):
        Observations.m_step(self, expectations, datas, inputs, masks, tags, optimizer=optimizer, **kwargs)


class RampingEmissions(PoissonEmissions):
    def __init__(self, N, K, D, M=0, single_subspace=True, link="softplus", bin_size=0.01):
        super(RampingEmissions, self).__init__(N, K, D, M=M, single_subspace=single_subspace, link=link, bin_size=bin_size)
        # Make sure the input matrix Fs is set to zero and never updated
        self.Fs *= 0
        self.ds *= 0

    # Construct an emissions model
    @property
    def params(self):
        return self.Cs

    @params.setter
    def params(self, value):
        self.Cs = value

    def log_prior(self):
        a = 2.0
        b = 0.05
        return np.sum((a-1.0) * np.log(self.Cs[0]) - b*self.Cs[0])
        # return np.sum(gamma.logpdf(self.Cs, 2, scale=1/0.025))

    def invert(self, data, input=None, mask=None, tag=None):
#         yhat = self.link(np.clip(data, .1, np.inf))
        yhat = smooth(data,20)
        xhat = self.link(np.clip(yhat, 0.01, np.inf))
        xhat = self._invert(xhat, input=input, mask=mask, tag=tag)
        num_pad = 10
        xhat = smooth(np.concatenate((np.zeros((num_pad,1)),xhat)),10)[num_pad:,:]
        xhat[xhat > 1.05] = 1.05
        if np.abs(xhat[0])>1.0:
                xhat[0] = 0.4 + 0.01*npr.randn(1,np.shape(xhat)[1])
        return xhat

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None, num_em_iters=10, num_tr_iters=50):
#         print("Initializing...")
        # datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        # yhats = [smooth(self.link(np.clip(d, .1, np.inf)),10) for d in datas]
        # take the strongest motion trials (e.g. max inputs u) and compute rate at the end of the trials
        # self.Cs = # set C to the above rate
        pass

# define new initial state distribution so the init parameters are not learned!

class ObservedRamping(HMM):
    def __init__(self, K=2, D=1, *, M=5, beta=np.linspace(-0.02,0.02,5), log_sigma_scale=np.log(10), x0=0.4):

        K, D, M = 2, 1, 5

        init_state_distn = InitialStateDistribution(K, D, M=M)
        init_state_distn.log_pi0 = np.log([0.999, 0.001])
        transition_distn = RampingTransitions(K, D, M=M)
        observation_distn = RampingObservations(K, D, M=M, beta=beta, log_sigma_scale=log_sigma_scale, x0=x0)

        super().__init__(K, D, M=M, init_state_distn=init_state_distn,
                                transitions=transition_distn,
                                observations=observation_distn)

class ObservedRampingHard(HMM):
    def __init__(self, K=2, D=1, *, M=5, beta=np.array([-0.01,-0.005,0.0,0.01,0.02]), log_sigma_scale=np.log(10), x0=0.4):

        K, D, M = 2, 1, 5

        init_state_distn = InitialStateDistribution(K, D, M=M)
        init_state_distn.log_pi0 = np.log([0.999, 0.001])
        transition_distn = RampingHardTransitions(K, D, M=M)
        observation_distn = RampingObservations(K, D, M=M, beta=beta, log_sigma_scale=log_sigma_scale, x0=x0)

        super().__init__(K, D, M=M, init_state_distn=init_state_distn,
                                transitions=transition_distn,
                                observations=observation_distn)


class Ramping(SLDS):
    def __init__(self, N, K=2, D=1, *, M=5, beta=np.linspace(-0.02,0.02,5), log_sigma_scale=np.log(100), x0=0.4, link="softplus", bin_size=1.0):

        K, D, M = 2, 1, 5

        init_state_distn = InitialStateDistribution(K, D, M=M)
        init_state_distn.log_pi0 = np.log([0.999, 0.001])
        transition_distn = RampingTransitions(K, D, M=M)
        dynamics_distn = RampingObservations(K, D, M=M, beta=beta, log_sigma_scale=log_sigma_scale, x0=x0)
        emission_distn = RampingEmissions(N, K, D, M=M, single_subspace=True, link=link, bin_size=bin_size)

        super().__init__(N, K=K, D=D, M=M, init_state_distn=init_state_distn,
                                transitions=transition_distn,
                                dynamics=dynamics_distn,
                                emissions=emission_distn)
