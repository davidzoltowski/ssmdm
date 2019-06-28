import autograd.numpy as np
import autograd.numpy.random as npr

import ssm
from ssm import hmm
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
import copy

# preprocessing
# for initialization
from tqdm.auto import trange
from ssm.optimizers import adam_step, rmsprop_step, sgd_step, lbfgs, bfgs, convex_combination
from autograd.scipy.misc import logsumexp
from autograd.tracer import getval
from autograd.misc import flatten
from autograd import value_and_grad

from ssm.util import ensure_args_are_lists, \
    logistic, logit, softplus, inv_softplus
from ssm.preprocessing import interpolate_data, pca_with_imputation

class AccumulationRaceTransitions(RecurrentOnlyTransitions):
    def __init__(self, K, D, M=0, scale=200):
        assert K == D+1
        assert D >= 1
        super(AccumulationRaceTransitions, self).__init__(K, D, M)

        # "Race" transitions with D+1 states
        # Transition to state d when x_d > 1.0
        # State 0 is the accumulation state
        # scale determines sharpness of the threshold
        self.Ws = np.zeros((K,M))
        self.Rs = np.vstack((np.zeros(D),scale*np.eye(D)))
        self.r = np.concatenate(([0],-scale*np.ones(D)))

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

class DDMTransitions(RecurrentOnlyTransitions):
    def __init__(self, K, D, M=0, scale=200):
        assert K == 3
        assert D == 1
        # assert M == 1
        super(DDMTransitions, self).__init__(K, D, M)

        # Parameters linking past observations to state distribution
        self.Ws = np.zeros((3, M))
        self.Rs = np.array([0, scale, -scale]).reshape((3, 1))
        self.r = np.array([0, -scale, -scale])

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

class AccumulationObservations(AutoRegressiveDiagonalNoiseObservations):
    def __init__(self, K, D, M, lags=1, learn_A=True):
        super(AccumulationObservations, self).__init__(K, D, M)

        # diagonal dynamics for each state
        # only learn dynamics for accumulation state
        self.learn_A = learn_A
        if self.learn_A:
            a_diag=np.ones((D,1))
            self._a_diag = a_diag
            mask1 = np.vstack( (np.eye(D)[None,:,:],np.zeros((K-1,D,D))) ) # for accum state
            mask2 = np.vstack( (np.zeros((1,D,D)), np.tile(np.eye(D),(K-1,1,1)) ))
            self._As = self._a_diag*mask1 + mask2
        else:
            self._As = np.tile(np.eye(D),(K,1,1))

        # learn dynamics for each state
        # a_diag=np.ones((K,D,1))
        # self._a_diag = a_diag
        # mask = np.tile(np.eye(D),(K,1,1))
        # self._As = self._a_diag*mask

        # set input Accumulation params, one for each dimension
        # They only differ in their input
        self._betas = np.ones(D,)
        self.Vs[0] = self._betas*np.eye(D)       # ramp
        for d in range(1,K):
            self.Vs[d] *= np.zeros((D,M))

        # set noise variances
        self.accum_log_sigmasq = np.log(1e-3)*np.ones(D,)
        mask1 = np.vstack( (np.ones(D,), np.zeros((K-1,D))) )
        mask2 = np.vstack( (np.zeros(D), np.ones((K-1,D))) )
        self._log_sigmasq = self.accum_log_sigmasq * mask1 + np.log(1e-4) * mask2

        # Set the remaining parameters to fixed values
        self.bs = np.zeros((K, D))
        self.mu_init = np.zeros((K, D))
        self._log_sigmasq_init = np.log(.001 * np.ones((K, D)))

    @property
    def params(self):
        if self.learn_A:
            params = self._betas, self.accum_log_sigmasq, self._a_diag
        else:
            params = self._betas, self.accum_log_sigmasq
        return params

    @params.setter
    def params(self, value):
        if self.learn_A:
            self._betas, self.accum_log_sigmasq, self._a_diag = value
        else:
            self._betas, self.accum_log_sigmasq = value

        K, D = self.K, self.D
        # update V
        mask = np.vstack((np.eye(D)[None,:,:], np.zeros((K-1,D,D))))
        self.Vs = self._betas * mask

        # update sigma
        mask1 = np.vstack( (np.ones(D,), np.zeros((K-1,D))) )
        mask2 = np.vstack( (np.zeros(D), np.ones((K-1,D))) )
        self._log_sigmasq = self.accum_log_sigmasq * mask1 + np.log(1e-4) * mask2

        # update A
        if self.learn_A:
            mask1 = np.vstack( (np.eye(D)[None,:,:],np.zeros((K-1,D,D))) ) # for accum state
            mask2 = np.vstack( (np.zeros((1,D,D)), np.tile(np.eye(D),(K-1,1,1)) ))
            self._As = self._a_diag*mask1 + mask2

    @property
    def betas(self):
        return self._betas

    @betas.setter
    def betas(self, value):
        assert value.shape == (self.D,)
        self._betas = value
        mask = np.vstack((np.eye(self.D)[None,:,:], np.zeros((self.K-1,self.D,self.D))))
        self.Vs = self._betas * mask

    def initialize(self, datas, inputs=None, masks=None, tags=None):
        pass

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        Observations.m_step(self, expectations, datas, inputs, masks, tags, **kwargs)

class Accumulation(HMM):
    def __init__(self, K, D, *, M,
                transitions="race",
                transition_kwargs=None,
                observation_kwargs=None,
                **kwargs):

        init_state_distn = InitialStateDistribution(K, D, M=M)
        init_state_distn.log_pi0 = np.log(np.concatenate(([0.999],(0.001/(K-1))*np.ones(K-1))))

        transition_classes = dict(
            race=AccumulationRaceTransitions,
            ddm=DDMTransitions)
        transition_kwargs = transition_kwargs or {}
        transitions = transition_classes[transitions](K, D, M=M, **transition_kwargs)

        observation_kwargs = observation_kwargs or {}
        observation_distn = AccumulationObservations(K, D, M=M, **observation_kwargs)

        super().__init__(K, D, M=M,
                            init_state_distn=init_state_distn,
                            transitions=transitions,
                            observations=observation_distn)

class AccumulationGaussianEmissions(GaussianEmissions):
    def __init__(self, N, K, D, M=0, single_subspace=True):
        super(AccumulationGaussianEmissions, self).__init__(N, K, D, M=M, single_subspace=single_subspace)
        # Make sure the input matrix Fs is set to zero and never updated
        self.Fs *= 0

    # Construct an emissions model
    @property
    def params(self):
        return self._Cs, self.ds, self.inv_etas

    @params.setter
    def params(self, value):
        self._Cs, self.ds, self.inv_etas = value

    def initialize(self, base_model, datas, inputs=None, masks=None, tags=None,
                   num_em_iters=50, num_tr_iters=50):

        print("Initializing...")
        print("First with FA using {} steps of EM.".format(num_em_iters))
        fa, xhats, Cov_xhats, lls = factor_analysis_with_imputation(self.D, datas, masks=masks, num_iters=num_em_iters)

        # define objective
        Td = sum([x.shape[0] for x in xhats])
        def _objective(params, itr):
            new_datas = [np.dot(x,params[0].T)+params[1] for x in xhats]
            obj = base_model.log_likelihood(new_datas,inputs=inputs)
            return -obj / Td

        # initialize R and r
        R = 0.1*np.random.randn(self.D,self.D)
        r = 0.01*np.random.randn(self.D)
        params = [R,r]

        print("Next by transforming latents to match AR-HMM prior using {} steps of max log likelihood.".format(num_tr_iters))
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

        # scale x's to be max at 1.25
        for d in range(self.D):
            x_transformed = [ (np.dot(x,R.T)+r)[:,d] for x in xhats]
            max_x = np.max(x_transformed)
            R[d,:] *= 1.25 / max_x
            r[d] *= 1.25 / max_x

        self.Cs = (fa.W @ np.linalg.inv(R)).reshape([1,self.N,self.D])
        self.ds = fa.mean - fa.W @ np.linalg.inv(R) @ r
        self.inv_etas = np.log(fa.sigmasq).reshape([1,self.N])


class AccumulationPoissonEmissions(PoissonEmissions):
    def __init__(self, N, K, D, M=0, single_subspace=True, link="softplus", bin_size=1.0):
        super(AccumulationPoissonEmissions, self).__init__(N, K, D, M=M, single_subspace=single_subspace, link=link, bin_size=bin_size)
        # Make sure the input matrix Fs is set to zero and never updated
        self.Fs *= 0

    # Construct an emissions model
    @property
    def params(self):
        return self._Cs, self.ds

    @params.setter
    def params(self, value):
        self._Cs, self.ds = value

    def invert(self, data, input=None, mask=None, tag=None, clip=np.array([0.0,1.0])):
#         yhat = self.link(np.clip(data, .1, np.inf))
        yhat = smooth(data,20)
        xhat = self.link(np.clip(yhat, 0.01, np.inf))
        xhat = self._invert(xhat, input=input, mask=mask, tag=tag)
        num_pad = 10
        xhat = smooth(np.concatenate((np.zeros((num_pad,self.D)),xhat)),10)[num_pad:,:]
        xhat = np.clip(xhat, -0.25, 1.5)

        if np.abs(xhat[0]).any()>1.0:
                xhat[0] = 0.2*npr.randn(1,self.D)
        return xhat

    # @ensure_args_are_lists
    def initialize(self, base_model, datas, inputs=None, masks=None, tags=None,
                   emission_optimizer="bfgs", num_optimizer_iters=1000):
        print("Initializing Emissions parameters...")

        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]

        Td = sum([data.shape[0] for data in datas])
        xs = [base_model.sample(T=data.shape[0],input=input)[1] for data, input in zip(datas, inputs)]
        def _objective(params, itr):
            self.params = params
            obj = 0
            obj += self.log_prior()
            for data, input, mask, tag, x in \
                zip(datas, inputs, masks, tags, xs):
                obj += np.sum(self.log_likelihoods(data, input, mask, tag, x))
            return -obj / Td

        # Optimize emissions log-likelihood
        optimizer = dict(bfgs=bfgs, lbfgs=lbfgs)[emission_optimizer]
        self.params = \
            optimizer(_objective,
                      self.params,
                      num_iters=num_optimizer_iters,
                      full_output=False)


class LatentAccumulation(SLDS):
    def __init__(self, N, K, D, *, M,
            transitions="race",
            transition_kwargs=None,
            dynamics_kwargs=None,
            emissions="gaussian",
            emission_kwargs=None,
            single_subspace=True,
            **kwargs):

        init_state_distn = InitialStateDistribution(K, D, M=M)
        init_state_distn.log_pi0 = np.log(np.concatenate(([0.999],(0.001/(K-1))*np.ones(K-1))))

        transition_classes = dict(
            race=AccumulationRaceTransitions,
            ddm=DDMTransitions)
        self.transitions_label = transitions
        self.transition_kwargs = transition_kwargs
        transition_kwargs = transition_kwargs or {}
        transitions = transition_classes[transitions](K, D, M=M, **transition_kwargs)

        self.dynamics_kwargs = dynamics_kwargs
        dynamics_kwargs = dynamics_kwargs or {}
        dynamics = AccumulationObservations(K, D, M=M, **dynamics_kwargs)

        self.emissions_label = emissions
        emission_classes = dict(
            gaussian=AccumulationGaussianEmissions,
            poisson=AccumulationPoissonEmissions)
        emission_kwargs = emission_kwargs or {}
        emissions = emission_classes[emissions](N, K, D, M=M,
            single_subspace=single_subspace, **emission_kwargs)

        super().__init__(N, K=K, D=D, M=M,
                            init_state_distn=init_state_distn,
                            transitions=transitions,
                            dynamics=dynamics,
                            emissions=emissions)

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None,
                   num_optimizer_iters=1000, num_em_iters=25,
                   betas=None, accum_log_sigmasq=None):

        # First initialize the observation model
        self.base_model = Accumulation(self.K, self.D, M=self.M,
                                       transitions=self.transitions_label,
                                       transition_kwargs=self.transition_kwargs,
                                       observation_kwargs=self.dynamics_kwargs)
        # if betas is not None:
        #     self.base_model.betas = betas
        # if log_sigma_scale is not None:
        #     self.base_model.accum_log_sigmasq = accum_log_sigmasq
        self.emissions.initialize(self.base_model,
                                  datas, inputs, masks, tags)
                                  # num_optimizer_iters=num_optimizer_iters)
        # betas = betas + 0.01*npr.randn()
        # log_sigma_scale = np.log(5 + 10*npr.rand())*np.ones(2,)
        # self.dynamics.params = (betas, log_sigma_scale)
        # print("Done.")


        if self.emissions_label=="gaussian":
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
