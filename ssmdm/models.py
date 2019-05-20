import autograd.numpy as np
import autograd.numpy.random as npr

import ssm
from ssm.hmm import HMM
from ssm.lds import SLDS

from ssmdm.misc import smooth

from ssm.util import random_rotation, ensure_args_are_lists
from ssm.observations import Observations, AutoRegressiveDiagonalNoiseObservations
from ssm.transitions import Transitions, RecurrentOnlyTransitions
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

from ssm.util import ensure_args_are_lists, \
    logistic, logit, softplus, inv_softplus
from ssm.preprocessing import interpolate_data, pca_with_imputation

class DDMTransitions(RecurrentOnlyTransitions):
    def __init__(self, K, D, M=0, scale=50):
        assert K == 3
        assert D == 1
        # assert M == 1
        super(DDMTransitions, self).__init__(K, D, M)

        # Parameters linking past observations to state distribution
        self.Ws = np.zeros((3, M))
        self.Rs = np.array([scale, 0, -scale]).reshape((3, 1))
        self.r = np.array([-scale, 0, -scale])

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

class DDMObservations(AutoRegressiveDiagonalNoiseObservations):
    def __init__(self, K, D=1, M=1, lags=1, beta=1.0, log_sigma_scale=np.log(10)):
        assert K == 3
        assert D == 1
        assert M == 1
        super(DDMObservations, self).__init__(K, D, M)

        # The only free parameters of the DDM are the ramp rate...
        self.beta = beta
        self.log_sigma_scale = log_sigma_scale

        # and the noise variances, which are initialized in the AR constructor
        self._log_sigmasq[0] = np.log(1e-4)
        self._log_sigmasq[1] = np.log(1e-4) + self.log_sigma_scale
        self._log_sigmasq[2] = np.log(1e-4)

        # Set the remaining parameters to fixed values
        self._As = np.ones((3, 1, 1))
        self.bs = np.zeros((3, 1))
        self.mu_init = np.zeros((3, 1))
        self._log_sigmasq_init = np.log(.01 * np.ones((3, 1)))

        # They only differ in their input
        self.Vs[0] = 0            # left bound
        self.Vs[1] = self.beta    # ramp
        self.Vs[2] = 0            # right bound

    @property
    def params(self):
        return self.beta, self.log_sigma_scale

    def log_prior(self):
        beta_mean = np.ones(self.D,)
        beta_cov = 0.5*np.eye(self.D)
        return stats.multivariate_normal_logpdf(np.array([self.beta]), beta_mean, beta_cov)

    @params.setter
    def params(self, value):
        self.beta, self.log_sigma_scale = value
        mask = np.reshape(np.array([0, 1, 0]), (3, 1, 1))
        self.Vs = mask * self.beta
        sig_mask = np.reshape(np.array([0, 1, 0]), (3, 1))
        self._log_sigmasq = np.log(1e-4)*np.ones((3,1)) + sig_mask * self.log_sigma_scale

    def initialize(self, datas, inputs=None, masks=None, tags=None):
        pass

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        Observations.m_step(self, expectations, datas, inputs, masks, tags, **kwargs)

class DDM(HMM):
    def __init__(self, K=3, D=1, *, M=1, betas=1.0, log_sigma_scale=np.log(10)):

        K, D, M = 3, 1, 1

        init_state_distn = InitialStateDistribution(K, D, M=M)
        init_state_distn.log_pi0 = np.log([0.01, 0.98, 0.01])
        transition_distn = DDMTransitions(K, D, M)
        observation_distn = DDMObservations(K, D, M, beta=beta, log_sigma_scale=log_sigma_scale)

        super().__init__(K, D, M=M, init_state_distn=init_state_distn,
                                transitions=transition_distn,
                                observations=observation_distn)

class DDMPoissonEmissions(PoissonEmissions):
    def __init__(self, N, K, D, M=0, single_subspace=True, link="softplus", bin_size=0.01):
        super(DDMPoissonEmissions, self).__init__(N, K, D, M=M, single_subspace=single_subspace, link=link, bin_size=bin_size)
        # Make sure the input matrix Fs is set to zero and never updated
        self.Fs *= 0

    # Construct an emissions model
    @property
    def params(self):
        return self._Cs, self.ds

    @params.setter
    def params(self, value):
        self._Cs, self.ds = value


    def invert(self, data, input=None, mask=None, tag=None):
#         yhat = self.link(np.clip(data, .1, np.inf))
        yhat = smooth(data,20)
        xhat = self.link(np.clip(yhat, 0.01, np.inf))
        xhat = self._invert(xhat, input=input, mask=mask, tag=tag)
        num_pad = 10
        xhat = smooth(np.concatenate((np.zeros((num_pad,1)),xhat)),10)[num_pad:,:]
        xhat[xhat > 1.05] = 1.05
        xhat[xhat < -1.05] = -1.05
        if np.abs(xhat[0])>1.0:
                xhat[0] = 0.3*npr.randn(1,np.shape(xhat)[1])
        return xhat

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None, num_em_iters=10, num_tr_iters=50):
#         print("Initializing...")

        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
#         yhats = [self.link(np.clip(d, .1, np.inf)) for d in datas]
        yhats = [smooth(self.link(np.clip(d, .1, np.inf)),10) for d in datas]

        print("First with FA using {} steps of EM.".format(num_em_iters))
        fa, xhats, Cov_xhats, lls = factor_analysis_with_imputation(self.D, yhats, masks=masks, num_iters=num_em_iters)

        # define objective
        def _objective(params, itr):
            Td = sum([x.shape[0] for x in xhats])
            new_datas = [np.dot(x,params[0].T)+params[1] for x in xhats]
#             new_datas = [np.dot(x,params[0].T) for x in xhats]
            obj = RampingDDM().log_likelihood(new_datas,inputs=inputs)
            return -obj / Td

        # initialize R and r
        R = 0.5*np.random.randn(self.D,self.D)
        r = 0.0
        params = [R,r]
#         params = [R]
        Td = sum([x.shape[0] for x in xhats])

        print("Next by transforming latents to match DDM prior using {} steps of max log likelihood.".format(num_tr_iters))

        state = None
        lls = [-_objective(params, 0) * Td]
        pbar = trange(num_tr_iters)
        pbar.set_description("Epoch {} Itr {} LP: {:.1f}".format(0, 0, lls[-1]))

        for itr in pbar:
            params, val, g, state = sgd_step(value_and_grad(_objective), params, itr, state)
            lls.append(-val * Td)
            pbar.set_description("LP: {:.1f}".format(lls[-1]))
            pbar.update(1)

        R = params[0]
        r = params[1]

        x_transformed = [np.dot(x,R.T)+r for x in xhats]
        x_all = np.concatenate(np.array([x.flatten() for x in x_transformed]))
        max_x = np.max(np.abs(x_all))
        # scale to make it go to 1.05 instead?

        R *= 1.05 / max_x
        r *= 1.05 / max_x

        self.Cs = (fa.W @ np.linalg.inv(R)).reshape([1,self.N,self.D])
        self.ds = fa.mean - np.squeeze(fa.W @ np.linalg.inv(R) * r)

class LatentDDMPoisson(SLDS):
    def __init__(self, N, K=3, D=1, *, M=1, beta=1.0, log_sigma_scale=np.log(10), link="softplus"):

        K, D, M = 3, 1, 1

        init_state_distn = InitialStateDistribution(K, D, M=M)
        init_state_distn.log_pi0 = np.log([0.01, 0.98, 0.01])
        transition_distn = DDMTransitions(K, D, M=M)
        dynamics_distn = DDMObservations(K, D, M=M, beta=beta, log_sigma_scale=log_sigma_scale)
        emission_distn = DDMPoissonEmissions(N, K, D, M=M, single_subspace=True, link=link)

        super().__init__(N, K=K, D=D, M=M, init_state_distn=init_state_distn,
                                transitions=transition_distn,
                                dynamics=dynamics_distn,
                                emissions=emission_distn)

class RampingObservations(AutoRegressiveDiagonalNoiseObservations):
    def __init__(self, K, D=1, M=5, lags=1, beta=np.linspace(-0.02,0.02,5), log_sigma_scale=np.log(10)):
        assert K == 3
        assert D == 1
        assert M == 5
        super(RampingObservations, self).__init__(K, D, M)

        # The only free parameters of the DDM are the ramp rate...
        self.beta = beta
        self.log_sigma_scale = log_sigma_scale

        # and the noise variances, which are initialized in the AR constructor
        self._log_sigmasq[0] = np.log(1e-4)
        self._log_sigmasq[1] = np.log(1e-4) + self.log_sigma_scale
        self._log_sigmasq[2] = np.log(1e-4)

        # Set the remaining parameters to fixed values
        self._As = np.ones((3, 1, 1))
        self.bs = np.zeros((3, 1))
        self.mu_init = np.zeros((3, 1))
        self._log_sigmasq_init = np.log(.01 * np.ones((3, 1)))

        # They only differ in their input
        self.Vs[0] = 0            # left bound
        self.Vs[1] = self.beta    # ramp
        self.Vs[2] = 0            # right bound

    @property
    def params(self):
        return self.beta, self.log_sigma_scale

    # def log_prior(self):
    #     beta_mean = np.ones(self.D,)
    #     beta_cov = 0.5*np.eye(self.D)
    #     return stats.multivariate_normal_logpdf(np.array([self.beta]), beta_mean, beta_cov)

    @params.setter
    def params(self, value):
        self.beta, self.log_sigma_scale = value
        mask = np.reshape(np.array([0, 1, 0]), (3, 1, 1))
        self.Vs = mask * self.beta
        sig_mask = np.reshape(np.array([0, 1, 0]), (3, 1))
        self._log_sigmasq = np.log(1e-4)*np.ones((3,1)) + sig_mask * self.log_sigma_scale

    def initialize(self, datas, inputs=None, masks=None, tags=None):
        pass

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        Observations.m_step(self, expectations, datas, inputs, masks, tags, **kwargs)

class RampingDDM(HMM):
    def __init__(self, K=3, D=1, *, M=5, beta=np.linspace(-0.02,0.02,5), log_sigma_scale=np.log(10)):

        K, D, M = 3, 1, 5

        init_state_distn = InitialStateDistribution(K, D, M=M)
        init_state_distn.log_pi0 = np.log([0.01, 0.98, 0.01])
        transition_distn = DDMTransitions(K, D, M=M)
        observation_distn = RampingObservations(K, D, M=M, beta=beta, log_sigma_scale=log_sigma_scale)

        super().__init__(K, D, M=M, init_state_distn=init_state_distn,
                                transitions=transition_distn,
                                observations=observation_distn)

class LatentRampingDDMPoisson(SLDS):
    def __init__(self, N, K=3, D=1, *, M=5, beta=np.linspace(-0.02,0.02,5), log_sigma_scale=np.log(10), link="softplus", bin_size=1.0):

        K, D, M = 3, 1, 5

        init_state_distn = InitialStateDistribution(K, D, M=M)
        init_state_distn.log_pi0 = np.log([0.01, 0.98, 0.01])
        transition_distn = DDMTransitions(K, D, M=M)
        dynamics_distn = RampingObservations(K, D, M=M, beta=beta, log_sigma_scale=log_sigma_scale)
        emission_distn = DDMPoissonEmissions(N, K, D, M=M, single_subspace=True, link=link, bin_size=bin_size)

        super().__init__(N, K=K, D=D, M=M, init_state_distn=init_state_distn,
                                transitions=transition_distn,
                                dynamics=dynamics_distn,
                                emissions=emission_distn)
