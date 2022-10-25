import autograd.numpy as np
import autograd.numpy.random as npr

import ssm
from ssm import hmm, lds
from ssm.hmm import HMM
from ssm.lds import SLDS
from ssm.util import random_rotation, ensure_args_are_lists, softplus
from ssm.observations import Observations, PoissonObservations
from ssm.transitions import Transitions, StationaryTransitions
from ssm.init_state_distns import InitialStateDistribution
from ssm.emissions import _LinearEmissions, GaussianEmissions, PoissonEmissions
from ssm.preprocessing import factor_analysis_with_imputation, interpolate_data, pca_with_imputation
from ssm.optimizers import adam_step, rmsprop_step, sgd_step, lbfgs, bfgs, convex_combination
from ssm.primitives import hmm_normalizer

import ssm.stats as stats

from ssmdm.misc import smooth

import copy

from tqdm import tqdm
from tqdm.auto import trange
from autograd.scipy.special import logsumexp
from autograd.tracer import getval
from autograd.misc import flatten
from autograd import value_and_grad

class TrialPoissonObservations(PoissonObservations):
    def __init__(self, K, D, M=0, num_trials=1, bin_size=1.0):
        super(PoissonObservations, self).__init__(K, D, M)

        # one set of lambdas for each trial
        self.bin_size = bin_size
        self.log_lambdas = npr.randn(K, D, num_trials) - np.log(self.bin_size)

    def log_likelihoods(self, data, input, mask, tag):
        lambdas = np.exp(self.log_lambdas[:, :, tag]) * self.bin_size # tag is trial number
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        return stats.poisson_logpdf(data[:, None, :], lambdas, mask=mask[:, None, :])

    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        lambdas = np.exp(self.log_lambdas[:, :, tag]) * self.bin_size
        return npr.poisson(lambdas[z])

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        for expectation, data, input, tag in zip(expectations, datas, inputs, tags):
            weights = expectation[0]
            for k in range(self.K):
                weighted_data = np.log(
                    np.average(data, axis=0, weights=weights[:,k]) + 1e-16) 
                self.log_lambdas[k, :, tag] = weighted_data / self.bin_size

    def smooth(self, expectations, data, input, tag):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        return expectations.dot(np.exp(self.log_lambdas[:, :, tag]))

class GammaDistributionObservations():
    def __init__(self, K, D, M=0):
        super(GammaDistributionObservations, self).__init__(K, D, M)

        # one gamma distribution per state
        # TODO -> change init?
        self.inv_alphas = np.log(np.ones((K, D)))
        self.inv_betas = np.log(0.01 * np.ones((K, D)))

    @property
    def params(self):
        return self.inv_alphas, self.inv_betas

    @params.setter
    def params(self, value):
        self.inv_alphas, self.inv_betas = value

    def log_likelihoods(self, data, input, mask, tag):
        lambdas = np.exp(self.log_lambdas)
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        return stats.poisson_logpdf(data[:, None, :], lambdas, mask=mask[:, None, :])

    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        alphas = np.exp(self.inv_alphas)
        betas = np.exp(self.inv_betas)
        lambdas = npr.gamma(alphas, 1.0 / betas)
        # lambdas = sample from Gamma(alphas, betas)
        return npr.poisson(lambdas[z])

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        pass

    def smooth(self, expectations, data, input, tag):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        mean_lambdas = np.exp(self.inv_alphas - self.inv_betas)
        return expectations.dot(mean_lambdas)


class StepTransitions(StationaryTransitions):
    def __init__(self, K, D, M=0):
        super(StationaryTransitions, self).__init__(K, D, M)
        assert K == 2

        # single step param
        self.log_p_step = np.log(0.02)
        self.eps = 1e-80
        Ps = np.array([[1.0 - np.exp(self.log_p_step), np.exp(self.log_p_step)],
                         [self.eps, 1.0-self.eps]])
        self.log_Ps = np.log(Ps)

    @property
    def params(self):
        return (self.log_p_step,)

    @params.setter
    def params(self, value):
        self.log_p_step = value[0]
        mask1 = np.array([-1.0, 1.0])
        r1 = np.array([1.0, 0.0]) + mask1 * np.exp(self.log_p_step)
        r2 = np.array([self.eps, 1.0 - self.eps])
        self.log_Ps = np.log(np.vstack((r1, r2)))

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        Transitions.m_step(self, expectations, datas, inputs, masks, tags, **kwargs)

class TrialStepTransitions(StationaryTransitions):
    def __init__(self, K, D, M=0, num_trials=1):
        super(StationaryTransitions, self).__init__(K, D, M)
        assert K == 2

        # single step param
        self.log_p_steps = np.log(0.02) * np.ones(num_trials)
        self.eps = 1e-80
        Ps = []
        for log_p_step in self.log_p_steps:
            P = np.array([[1.0 - np.exp(log_p_step), np.exp(log_p_step)],
                            [self.eps, 1.0-self.eps]])
            Ps.append(P)
        self.log_Ps = np.array([np.log(P) for P in Ps])

    @property
    def params(self):
        return (self.log_p_steps,)

    @params.setter
    def params(self, value):
        self.log_p_steps = value[0]
        mask1 = np.array([-1.0, 1.0])
        r1s = [np.array([1.0, 0.0]) + mask1 * np.exp(log_p_step) for log_p_step in self.log_p_steps]
        r2 = np.array([self.eps, 1.0 - self.eps])
        self.log_Ps = np.array([np.log(np.vstack((r1, r2))) for r1 in r1s])

    def log_transition_matrices(self, data, input, mask, tag):
        log_Ps = self.log_Ps[tag] - logsumexp(self.log_Ps[tag], axis=1, keepdims=True)
        return log_Ps[None, :, :]

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        Transitions.m_step(self, expectations, datas, inputs, masks, tags, **kwargs)

class Step(HMM):
    def __init__(self, K, D, *, M=0,
                 transitions="step",
                 transition_kwargs=None,
                 observations="trial_poission",
                 observation_kwargs=None,
                 **kwargs):
        assert K == 2

        init_state_distn = InitialStateDistribution(K, D, M=M)
        eps = 1e-80
        init_state_distn.log_pi0 = np.log(np.concatenate(([1.0-eps],(eps)*np.ones(K-1))))

        transition_classes = dict(
            step=StepTransitions,
            trial_step=TrialStepTransitions)
        transition_kwargs = transition_kwargs or {}
        transitions = transition_classes[transitions](K, D, M=M, **transition_kwargs)   

        observation_classes = dict(
            trial_poisson=TrialPoissonObservations,
            poisson=PoissonObservations)
        observation_kwargs = observation_kwargs or {}
        observation_distn = observation_classes[observations](K, D, M=M, **observation_kwargs)

        super().__init__(K, D, M=M,
                            init_state_distn=init_state_distn,
                            transitions=transitions,
                            observations=observation_distn)