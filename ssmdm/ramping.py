import autograd.numpy as np
import autograd.numpy.random as npr

import ssm
from ssm import hmm, lds
from ssm.hmm import HMM
from ssm.lds import SLDS
from ssm.util import random_rotation, ensure_args_are_lists, one_hot, softplus
from ssm.observations import Observations, AutoRegressiveDiagonalNoiseObservations
from ssm.transitions import Transitions, RecurrentOnlyTransitions, RecurrentTransitions
from ssm.init_state_distns import InitialStateDistribution
from ssm.emissions import _LinearEmissions, GaussianEmissions, PoissonEmissions
from ssm.preprocessing import factor_analysis_with_imputation, interpolate_data, pca_with_imputation
from ssm.optimizers import adam_step, rmsprop_step, sgd_step, convex_combination
import ssm.stats as stats

from ssmdm.misc import smooth

import copy

from tqdm.auto import trange
from autograd.scipy.special import logsumexp
from autograd.tracer import getval
from autograd.misc import flatten
from autograd import value_and_grad
from scipy.stats import gamma, norm, invgamma
from scipy.special import gammaln


class RampingSoftTransitions(RecurrentOnlyTransitions):
    def __init__(self, K, D, M=0, scale=200):
        assert K == 2
        assert D == 1
        super(RampingSoftTransitions, self).__init__(K, D, M)

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


class RampingTransitions(RecurrentTransitions):
    def __init__(self, K, D, M=0, scale=100):
        assert K == 2
        assert D == 1
        super(RampingTransitions, self).__init__(K, D, M)

        # Parameters linking past observations to state distribution
        self.log_Ps = np.array([[0, -scale], [-scale, scale]])
        self.Ws = np.zeros((K, M))
        self.Rs = np.array([0, scale]).reshape((K, D))

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

class RampingLowerBoundTransitions(RecurrentTransitions):
    def __init__(self, K, D, M=0, scale=200.0):
        assert K == 3
        assert D == 1
        super(RampingLowerBoundTransitions, self).__init__(K, D, M)

        # Parameters linking past observations to state distribution
        self.log_Ps = -scale*np.ones((K,K)) + np.diag(np.concatenate(([scale],2.0*scale*np.ones(K-1))))
        self.scale = scale
        self.lb_loc = 0.0
        self.lb_scale = 10.0
        self.log_Ps[0,2] = self.lb_loc * self.lb_scale
        self.Ws = np.zeros((K, M))
        self.Rs = np.array([0, scale, -self.lb_scale]).reshape((3, 1))

    @property
    def params(self):
        return self.lb_loc, self.lb_scale

    @params.setter
    def params(self, value):
        self.lb_loc, self.lb_scale = value
        mask = np.vstack((np.array([1.0,1.0,0.0]), np.ones((self.K-1,self.K))))
        log_Ps = -self.scale*np.ones((self.K,self.K)) + np.diag(np.concatenate(([self.scale],2.0*self.scale*np.ones(self.K-1))))
        self.log_Ps = mask * log_Ps + (1.0 - mask) * self.lb_loc * self.lb_scale * np.ones((self.K,self.K))
        self.Rs = np.array([0.0, self.scale, -self.lb_scale]).reshape((3, 1))

    def log_prior(self):
        loc_mean = 0.0
        loc_var = 0.5
        return np.sum(-0.5 * (self.lb_loc - loc_mean)**2 / loc_var)

    def initialize(self, datas, inputs=None, masks=None, tags=None):
        pass

    def m_step(self, expectations, datas, inputs, masks, tags, optimizer="bfgs", **kwargs):
        Transitions.m_step(self, expectations, datas, inputs, masks, tags, optimizer=optimizer, **kwargs)


class RampingObservations(AutoRegressiveDiagonalNoiseObservations):
    def __init__(self, K, D=1, M=5, lags=1, beta=None, log_sigma_scale=np.log(1e-3), x0=0.4):
        assert D == 1
        super(RampingObservations, self).__init__(K, D, M)

        # The only free parameters are the ramp rate...
        if beta is None:
            beta = np.linspace(-0.02,0.02,M)
        self.beta = beta
        self.log_sigma_scale = log_sigma_scale # log variance
        self.x0 = x0 # mu init

        # and the noise variances
        self.base_var=1e-5
        mask1 = np.vstack( (np.ones(D,), np.zeros((K-1,D))) )
        mask2 = np.vstack( (np.zeros(D), np.ones((K-1,D))) )
        self._log_sigmasq = self.log_sigma_scale * mask1 + np.log(self.base_var) * mask2

        # set init params
        self.mu_init = self.x0 * np.ones((K,1))
        self._log_sigmasq_init = self.log_sigma_scale * mask1 + np.log(self.base_var) * mask2

        # They only differ in their input
        self.Vs[0] = self.beta    # ramp
        for k in range(1,K):
            self.Vs[k] = 0        # bound

        # Set the remaining parameters to fixed values
        self._As = np.ones((K, 1, 1))
        self.bs = np.zeros((K, 1))

    @property
    def params(self):
        return self.beta, self.log_sigma_scale, self.x0

    @params.setter
    def params(self, value):
        self.beta, self.log_sigma_scale, self.x0 = value
        mask = np.reshape(np.concatenate(([1],np.zeros(self.K-1))), (self.K, 1, 1))
        self.Vs = mask * self.beta
        mask1 = np.vstack( (np.ones(self.D,), np.zeros((self.K-1,self.D))) )
        mask2 = np.vstack( (np.zeros(self.D), np.ones((self.K-1,self.D))) )
        self._log_sigmasq = self.log_sigma_scale * mask1 + np.log(self.base_var) * mask2
        self.mu_init = self.x0 * np.ones((self.K,1))
        self._log_sigmasq_init = self.log_sigma_scale * mask1 + np.log(self.base_var) * mask2

    def log_prior(self):
        beta_mean = np.zeros(np.shape(self.beta)[0])
        beta_cov = (0.1**2)*np.eye(np.shape(self.beta)[0])
        dyn_var = np.exp(self.log_sigma_scale)
        alpha = 1.1 # or 0.02
        beta = 1e-3 # or 0.02
        return np.sum(stats.multivariate_normal_logpdf(np.array([self.beta]), beta_mean, beta_cov)) \
                + np.sum(-0.5 * (self.x0 - 0.5)**2 / 1.0) \
                + alpha*np.log(beta) - gammaln(alpha) - (alpha+1)*np.log(dyn_var) - beta/dyn_var

    def initialize(self, datas, inputs=None, masks=None, tags=None):
        pass

    def m_step(self, expectations, datas, inputs, masks, tags, optimizer="lbfgs", **kwargs):
        Observations.m_step(self, expectations, datas, inputs, masks, tags, optimizer=optimizer, **kwargs)


class RampingPoissonEmissions(PoissonEmissions):
    def __init__(self, N, K, D, M=0, single_subspace=True, link="softplus", bin_size=0.01):
        super(RampingPoissonEmissions, self).__init__(N, K, D, M=M, single_subspace=single_subspace, link=link, bin_size=bin_size)
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

    # this thresholds at 1.0. Can keep (or not)
    # def forward(self, x, input, tag):
    #     return np.matmul(self.Cs[None, ...], np.minimum(x[:, None, :, None],1.0))[:, :, :, 0] \
    #         + np.matmul(self.Fs[None, ...], input[:, None, :, None])[:, :, :, 0] \
    #         + self.ds

    def invert(self, data, input=None, mask=None, tag=None):
        yhat = smooth(data,20)
        xhat = self.link(np.clip(yhat, 0.01, np.inf))
        xhat = self._invert(xhat, input=input, mask=mask, tag=tag)
        for t in range(xhat.shape[0]):
            if np.all(xhat[np.max([0,t-2]):t+3]>0.99) and t>2:
                xhat[np.minimum(0,t-2):] = 1.01*np.ones(np.shape(xhat[np.minimum(0,t-2):]))
                xhat[:np.maximum(0,t-2)] = np.clip(xhat[:np.maximum(0,t-2)], -0.5,0.95)

        if np.abs(xhat[0])>1.0:
                xhat[0] = 0.5 + 0.01*npr.randn(1,np.shape(xhat)[1])
        return xhat

    def initialize(self, datas, inputs=None, masks=None, tags=None, choices=None):

        def initialize_ramp_choice(ys,choices, bin_size):
        	choice_0 = np.where(choices==0)[0]
        	choice_1 = np.where(choices==1)[0]
        	y_end = np.array([y[-5:] for y in ys])
        	C0 = np.mean(y_end[choice_0]) / bin_size
        	C1 = np.mean(y_end[choice_1]) / bin_size
        	C = max(C0, C1)
        	y0_mean = np.mean([y[:3] for y in ys])
        	x0 = y0_mean / C / bin_size
        	return C, x0

        choices = np.array(choices)
        C, x0 = initialize_ramp_choice(datas, choices, self.bin_size)

        self.Cs[0] = C.reshape((self.N,self.D))

        return C, x0

    # need this hessian if thresholding at 1
    # def hessian_log_emissions_prob(self, data, input, mask, tag, x):
    #     """
    #     d/dx log p(y | x) = d/dx [y * (Cx + Fu + d) - exp(Cx + Fu + d)
    #                       = y * C - lmbda * C
    #                       = (y - lmbda) * C
    #
    #     d/dx  (y - lmbda)^T C = d/dx -exp(Cx + Fu + d)^T C
    #         = -C^T exp(Cx + Fu + d)^T C
    #     """
    #     if self.link_name == "log":
    #         assert self.single_subspace
    #         lambdas = self.mean(self.forward(x, input, tag))
    #         return np.einsum('tn, ni, nj ->tij', -lambdas[:, 0, :], self.Cs[0], self.Cs[0])
    #
    #     elif self.link_name == "softplus":
    #         assert self.single_subspace
    #         lambdas = self.mean(self.forward(x, input, tag))[:, 0, :] / self.bin_size
    #         expterms = np.exp(-np.dot(x,self.Cs[0].T)-np.dot(input,self.Fs[0].T)-self.ds[0])
    #         diags = (data / lambdas * (expterms - 1.0 / lambdas) - expterms * self.bin_size) / (1.0+expterms)**2
    #         diags[np.tile(x>=1,(1,self.N))] *= 0.0
    #         return np.einsum('tn, ni, nj ->tij', diags, self.Cs[0], self.Cs[0])

class RampingGaussianEmissions(GaussianEmissions):
    def __init__(self, N, K, D, M=0, single_subspace=True):
        super(RampingGaussianEmissions, self).__init__(N, K, D, M=M, single_subspace=single_subspace)
        # Make sure the input matrix Fs is set to zero and never updated
        self.Fs *= 0

    # Construct an emissions model
    @property
    def params(self):
        return self._Cs, self.ds, self.inv_etas

    @params.setter
    def params(self, value):
        self._Cs, self.ds, self.inv_etas = value

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None, num_em_iters=10, num_tr_iters=50):
        # here, init params using data
        # estimate boundary rate
        # use that to estimate initial x0
        # then init w2, betas
        pass

class RampingInitialStateDistribution(InitialStateDistribution):
    def __init__(self, K, D, M=0):
        self.K, self.D, self.M = K, D, M
        self.log_pi0 = -np.log(K) * np.ones(K)

    @property
    def params(self):
        return (self.log_pi0,)

    @params.setter
    def params(self, value):
        self.log_pi0 = value[0]

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        pass

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        self.log_pi0 = self.log_pi0[perm]

    @property
    def initial_state_distn(self):
        return np.exp(self.log_pi0 - logsumexp(self.log_pi0))

    @property
    def log_initial_state_distn(self):
        return self.log_pi0 - logsumexp(self.log_pi0)

    def log_prior(self):
        return 0

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        self.log_pi0 = self.log_pi0


class ObservedRamping(HMM):
    def __init__(self, K, D, *, M,
                transitions="ramp",
                transition_kwargs=None,
                observation_kwargs=None,
                **kwargs):

        init_state_distn = RampingInitialStateDistribution(K, D, M=M)
        init_state_distn.log_pi0 = np.log(np.concatenate(([0.999],(0.001/(K-1))*np.ones(K-1))))

        if transitions == "ramplower":
            assert K == 3
        transition_classes = dict(
            ramp=RampingTransitions,
            ramplower=RampingLowerBoundTransitions,
            rampsoft=RampingSoftTransitions)
        transition_kwargs = transition_kwargs or {}
        transitions = transition_classes[transitions](K, D, M=M, **transition_kwargs)

        observation_kwargs = observation_kwargs or {}
        observations = RampingObservations(K, D, M=M, **observation_kwargs)

        super().__init__(K, D, M=M,
                            init_state_distn=init_state_distn,
                            transitions=transitions,
                            observations=observations)


class Ramping(SLDS):
    def __init__(self, N, K=2, D=1, *, M=5,
            transitions="ramp",
            transition_kwargs=None,
            dynamics_kwargs=None,
            emissions="poisson",
            emission_kwargs=None,
            single_subspace=True,
            **kwargs):

        init_state_distn = RampingInitialStateDistribution(K, D, M=M)
        init_state_distn.log_pi0 = np.log(np.concatenate(([0.999],(0.001/(K-1))*np.ones(K-1))))

        if transitions == "ramplower":
            assert K == 3
        transition_classes = dict(
            ramp=RampingTransitions,
            ramplower=RampingLowerBoundTransitions,
            rampsoft=RampingSoftTransitions)
        transition_kwargs = transition_kwargs or {}
        transitions = transition_classes[transitions](K, D, M=M, **transition_kwargs)

        dynamics_kwargs = dynamics_kwargs or {}
        dynamics = RampingObservations(K, D, M=M, **dynamics_kwargs)

        emission_classes = dict(
            poisson=RampingPoissonEmissions,
            gaussian=RampingGaussianEmissions)
        emission_kwargs = emission_kwargs or {}
        emissions = emission_classes[emissions](N, K, D, M=M,
            single_subspace=single_subspace, **emission_kwargs)

        super().__init__(N, K=K, D=D, M=M,
                            init_state_distn=init_state_distn,
                            transitions=transitions,
                            dynamics=dynamics,
                            emissions=emissions)

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None, num_em_iters=25, choices=None):

        C, x0 = self.emissions.initialize(datas, inputs, masks, tags, choices)

        self.dynamics.params = (self.dynamics.params[0], self.dynamics.params[1], x0)

        # Get the initialized variational mean for the data
        xs = [self.emissions.invert(data, input, mask, tag)
              for data, input, mask, tag in zip(datas, inputs, masks, tags)]
        xmasks = [np.ones_like(x, dtype=bool) for x in xs]

        # Now run a few iterations of EM on a ARHMM with the variational mean
        print("Initializing with an ARHMM using {} steps of EM.".format(num_em_iters))
        arhmm = hmm.HMM(self.K, self.D, M=self.M,
                        init_state_distn=copy.deepcopy(self.init_state_distn),
                        transitions=copy.deepcopy(self.transitions),
                        observations=copy.deepcopy(self.dynamics))

        arhmm.fit(xs, inputs=inputs, masks=xmasks, tags=tags,
                  method="em", num_em_iters=num_em_iters)

        self.init_state_distn = copy.deepcopy(arhmm.init_state_distn)
        self.transitions = copy.deepcopy(arhmm.transitions)
        self.dynamics = copy.deepcopy(arhmm.observations)

def simulate_ramping(beta=np.linspace(-0.02,0.02,5), w2=3e-3, x0=0.5, C=40, T=100, bin_size=0.01):

    NC = 5 # number of trial types
    cohs = np.arange(NC)
    trial_cohs = np.repeat(cohs, int(T / NC))
    tr_lengths = np.random.randint(50,size=(T))+50
    us = []
    xs = []
    zs = []
    ys = []
    for t in range(T):
        tr_coh = trial_cohs[t]
        betac = beta[tr_coh]

        tr_length = tr_lengths[t]
        x = np.zeros(tr_length)
        z = np.zeros(tr_length)
        x[0] = x0 + np.sqrt(w2)*npr.randn()
        z[0] = 0
        for i in np.arange(1,tr_length):

            if x[i-1] >= 1.0:
                x[i] = 1.0
                z[i] = 1
            else:
                x[i] = np.min((1.0, x[i-1] + betac + np.sqrt(w2)*npr.randn()))
                if x[i] >= 1.0:
                    z[i] = 1
                else:
                    z[i] = 0

        y = npr.poisson(np.log1p(np.exp(C*x))*bin_size)

        u = np.tile(one_hot(tr_coh,5), (tr_length,1))
        us.append(u)
        xs.append(x.reshape((tr_length,1)))
        zs.append(z.reshape((tr_length,1)))
        ys.append(y.reshape((tr_length,1)))

    return ys, xs, zs, us, tr_lengths, trial_cohs
