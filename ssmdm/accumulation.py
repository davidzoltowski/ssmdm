import autograd.numpy as np
import autograd.numpy.random as npr

import ssm
from ssm import hmm, lds
from ssm.hmm import HMM
from ssm.lds import SLDS
from ssm.util import random_rotation, ensure_args_are_lists, softplus
from ssm.observations import Observations, AutoRegressiveDiagonalNoiseObservations
from ssm.transitions import Transitions, RecurrentTransitions, RecurrentOnlyTransitions
from ssm.init_state_distns import InitialStateDistribution
from ssm.emissions import _LinearEmissions, GaussianEmissions, PoissonEmissions
from ssm.preprocessing import factor_analysis_with_imputation, interpolate_data, pca_with_imputation
from ssm.optimizers import adam_step, rmsprop_step, sgd_step, lbfgs, bfgs, convex_combination

from ssmdm.misc import smooth

import copy

from tqdm.auto import trange
from autograd.scipy.misc import logsumexp
from autograd.tracer import getval
from autograd.misc import flatten
from autograd import value_and_grad

class AccumulationRaceTransitions(RecurrentTransitions):
    def __init__(self, K, D, M=0, scale=400):
        assert K == D+1
        assert D >= 1
        super(AccumulationRaceTransitions, self).__init__(K, D, M)

        # "Race" transitions with D+1 states
        # Transition to state d when x_d > 1.0
        # State 0 is the accumulation state
        # scale determines sharpness of the threshold
        # Transitions out of boundary states occur w/ very low probability
        top_row = np.concatenate(([0.0],-scale*np.ones(D)))
        rest_rows = np.hstack((-scale*np.ones((D,1)),-scale*np.ones((D,D)) + np.diag(2.0*scale*np.ones(D))))
        self.log_Ps = np.vstack((top_row,rest_rows))
        self.Ws = np.zeros((K,M))
        self.Rs = np.vstack((np.zeros(D),scale*np.eye(D)))

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

class AccumulationRaceSoftTransitions(RecurrentOnlyTransitions):
    def __init__(self, K, D, M=0, scale=100):
        assert K == D+1
        assert D >= 1
        super(AccumulationRaceSoftTransitions, self).__init__(K, D, M)

        # Like Race Transitions but soft boundaries
        # Transitions depend on previous x only
        # Transition to state d when x_d > 1.0
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

class DDMTransitions(RecurrentTransitions):
    def __init__(self, K, D, M=0, scale=400):
        assert K == 3
        assert D == 1
        # assert M == 1
        super(DDMTransitions, self).__init__(K, D, M)

        # DDM has one accumulation state and boundary states at +/- 1
        self.log_Ps = -scale*np.ones((K,K)) + np.diag(np.concatenate(([scale],2.0*scale*np.ones(K-1))))
        self.Ws = np.zeros((K, M))
        self.Rs = np.array([0, scale, -scale]).reshape((3, 1))

    @property
    def params(self):
        return ()

    @params.setter
    def params(self, value):
        pass
        # scale = value
        # K, M = self.K, self.M
        # self.log_Ps = -scale*np.ones((K,K)) + np.diag(np.concatenate(([scale],2.0*scale*np.ones(K-1))))
        # self.Ws = np.zeros((K, M))
        # self.Rs = np.array([0, scale, -scale]).reshape((3, 1))
        # pass

    def initialize(self, datas, inputs=None, masks=None, tags=None):
        pass

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        pass

class DDMSoftTransitions(RecurrentOnlyTransitions):
    def __init__(self, K, D, M=0, scale=200):
        assert K == 3
        assert D == 1
        super(DDMSoftTransitions, self).__init__(K, D, M)

        # DDM where transitions out of boundary state can occur
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

class DDMCollapsingTransitions(RecurrentTransitions):
    def __init__(self, K, D, M=0, scale=200):
        assert K == 3
        assert D == 1
        # assert M == 1
        super(DDMCollapsingTransitions, self).__init__(K, D, M)

        # Parameters linking past observations to state distribution
        self.log_Ps = -scale*np.ones((K,K)) + np.diag(np.concatenate(([scale],2.0*scale*np.ones(K-1))))
        self.bound_scale = 0.008 # 0.01
        self.Ws = self.bound_scale * scale * np.eye(K,M)
        self.Ws[0][0] = 0.0
        self.Rs = np.array([0, scale, -scale]).reshape((3, 1))

    @property
    def params(self):
        return ()

    @params.setter
    def params(self, value):
        pass
        # scale = value
        # K, M = self.K, self.M
        # self.log_Ps = -scale*np.ones((K,K)) + np.diag(np.concatenate(([scale],2.0*scale*np.ones(K-1))))
        # bound_scale = 0.01
        # self.Ws = bound_scale * scale * np.eye(K,M)
        # self.Ws[0][0] = 0.0
        # self.Rs = np.array([0, scale, -scale]).reshape((3, 1))

    def initialize(self, datas, inputs=None, masks=None, tags=None):
        pass

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        pass


class AccumulationObservations(AutoRegressiveDiagonalNoiseObservations):
    def __init__(self, K, D, M, lags=1, learn_A=True, learn_V=False):
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

        # set input Accumulation params, one for each dimension
        # first D inputs are accumulated in different dimensions
        # rest of M-D inputs are applied to each dimension
        self._betas = 0.1*np.ones(D,)
        self.learn_V = learn_V
        self._V = 0.0*np.ones((D, M-D)) # additional covariates, if they exist
        self.Vs[0] = np.hstack((self._betas*np.eye(D,D), self._V))
        for d in range(1,K):
            self.Vs[d] *= np.zeros((D,M))

        # set noise variances
        self.accum_log_sigmasq = np.log(1e-3)*np.ones(D,)
        mask1 = np.vstack( (np.ones(D,), np.zeros((K-1,D))) )
        mask2 = np.vstack( (np.zeros(D), np.ones((K-1,D))) )
        self.bound_variance = 1e-4
        self._log_sigmasq = self.accum_log_sigmasq * mask1 + np.log(self.bound_variance) * mask2
        self._log_sigmasq_init = (self.accum_log_sigmasq + np.log(2) )* mask1 + np.log(self.bound_variance) * mask2

        # Set the remaining parameters to fixed values
        self.bs = np.zeros((K, D))
        acc_mu_init = np.zeros((1,D))
        self.mu_init = np.vstack((acc_mu_init,np.ones((K-1,D))))

    @property
    def params(self):
        params = self._betas, self.accum_log_sigmasq
        params = params + (self._a_diag,) if self.learn_A else params
        params = params + (self._V,) if self.learn_V else params
        return params

    @params.setter
    def params(self, value):
        self._betas, self.accum_log_sigmasq = value[:2]
        if self.learn_A:
            self._a_diag = value[2]
        if self.learn_V:
            self._V = value[-1]

        K, D, M = self.K, self.D, self.M

        # update V
        mask0 = np.hstack((np.eye(D), np.ones((D,M-D)))) # state K = 0
        mask = np.vstack((mask0[None,:,:], np.zeros((K-1,D,M))))

        # self.Vs = self._betas * mask
        self.Vs = np.hstack((np.diag(self._betas), self._V)) * mask

        # update sigmas
        mask1 = np.vstack( (np.ones(D,), np.zeros((K-1,D))) )
        mask2 = np.vstack( (np.zeros(D), np.ones((K-1,D))) )
        self._log_sigmasq = self.accum_log_sigmasq * mask1 + np.log(self.bound_variance) * mask2
        self._log_sigmasq_init = (self.accum_log_sigmasq + np.log(2) )* mask1 + np.log(self.bound_variance) * mask2

        # update A
        # if self.learn_A:
        mask1 = np.vstack( (np.eye(D)[None,:,:],np.zeros((K-1,D,D))) ) # for accum state
        mask2 = np.vstack( (np.zeros((1,D,D)), np.tile(np.eye(D),(K-1,1,1)) ))
        self._As = self._a_diag*mask1 + mask2

    # @property
    # def betas(self):
    #     return self._betas
    #
    # @betas.setter
    # def betas(self, value):
    #     assert value.shape == (self.D,)
    #     self._betas = value
    #     mask = np.vstack((np.eye(self.D)[None,:,:], np.zeros((self.K-1,self.D,self.D))))
    #     self.Vs = self._betas * mask

    def initialize(self, datas, inputs=None, masks=None, tags=None):
        pass

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        Observations.m_step(self, expectations, datas, inputs, masks, tags, **kwargs)

class AccumulationGLMObservations(AutoRegressiveDiagonalNoiseObservations):
    def __init__(self, K, D, M, lags=1):
        super(AccumulationGLMObservations, self).__init__(K, D, M)

        # diagonal dynamics for each state
        # only learn dynamics for accumulation state
        a_diag=np.ones((D,1))
        self._a_diag = a_diag
        mask1 = np.vstack( (np.eye(D)[None,:,:],np.zeros((K-1,D,D))) ) # for accum state
        mask2 = np.vstack( (np.zeros((1,D,D)), np.tile(np.eye(D),(K-1,1,1)) ))
        self._As = self._a_diag*mask1 + mask2

        # set input Accumulation params, one for each dimension
        # They only differ in their input
        self._V0 = np.zeros((D,M))
        self.Vs[0] = self._V0       # ramp
        for d in range(1,K):
            self.Vs[d] *= np.zeros((D,M))

        # set noise variances
        self.accum_log_sigmasq = np.log(1e-3)*np.ones(D,)
        mask1 = np.vstack( (np.ones(D,), np.zeros((K-1,D))) )
        mask2 = np.vstack( (np.zeros(D), np.ones((K-1,D))) )
        self.bound_variance = 1e-5
        self._log_sigmasq = self.accum_log_sigmasq * mask1 + np.log(self.bound_variance) * mask2

        # Set the remaining parameters to fixed values
        self.bs = np.zeros((K, D))
        self.mu_init = np.zeros((K, D))
        self._log_sigmasq_init = np.log(.001 * np.ones((K, D)))

    @property
    def params(self):
        params = self._V0, self.accum_log_sigmasq, self._a_diag
        return params

    @params.setter
    def params(self, value):
        self._V0, self.accum_log_sigmasq, self._a_diag = value

        K, D, M = self.K, self.D, self.M

        # Update V
        self.Vs = np.vstack((self._V0*np.ones((1,D,M)), np.zeros((K-1,D,M))))

        # update sigma
        mask1 = np.vstack( (np.ones(D,), np.zeros((K-1,D))) )
        mask2 = np.vstack( (np.zeros(D), np.ones((K-1,D))) )
        self._log_sigmasq = self.accum_log_sigmasq * mask1 + np.log(self.bound_variance) * mask2

        # update A
        mask1 = np.vstack( (np.eye(D)[None,:,:],np.zeros((K-1,D,D))) ) # for accum state
        mask2 = np.vstack( (np.zeros((1,D,D)), np.tile(np.eye(D),(K-1,1,1)) ))
        self._As = self._a_diag*mask1 + mask2

    def initialize(self, datas, inputs=None, masks=None, tags=None):
        pass

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        Observations.m_step(self, expectations, datas, inputs, masks, tags, **kwargs)

class Accumulation(HMM):
    def __init__(self, K, D, *, M,
                transitions="race",
                transition_kwargs=None,
                observations="acc",
                observation_kwargs=None,
                **kwargs):

        init_state_distn = AccumulationInitialStateDistribution(K, D, M=M)
        init_state_distn.log_pi0 = np.log(np.concatenate(([0.999],(0.001/(K-1))*np.ones(K-1))))

        transition_classes = dict(
            racesoft=AccumulationRaceSoftTransitions,
            race=AccumulationRaceTransitions,
            ddmsoft=DDMSoftTransitions,
            ddm=DDMTransitions,
            ddmcollapsing=DDMCollapsingTransitions)
        transition_kwargs = transition_kwargs or {}
        transitions = transition_classes[transitions](K, D, M=M, **transition_kwargs)

        observation_classes = dict(
            acc=AccumulationObservations,
            accglm=AccumulationGLMObservations)
        observation_kwargs = observation_kwargs or {}
        observation_distn = observation_classes[observations](K, D, M=M, **observation_kwargs)

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

        if self.D == 1 and base_model.transitions.type_name == "DDMTransitions":

            d_init = np.mean([y[0:3] for y in datas],axis=(0,1))
            u_sum = np.array([np.sum(u) for u in inputs])
            y_end = np.array([y[-3:] for y in datas])
            u_l, u_u = np.percentile(u_sum, [20,80]) # use 20th and 80th percentile input
            y_U = y_end[np.where(u_sum>=u_u)]
            y_L = y_end[np.where(u_sum<=u_l)]
            C_init = (1.0/2.0)*np.mean((np.mean(y_U,axis=0) - np.mean(y_L,axis=0)),axis=0)

            self.Cs = C_init.reshape([1,self.N,self.D])
            self.ds = d_init.reshape([1,self.N])
            self.inv_etas = np.log(fa.sigmasq).reshape([1,self.N])

        else:

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

            # scale x's to be max at 1.1
            for d in range(self.D):
                x_transformed = [ (np.dot(x,R.T)+r)[:,d] for x in xhats]
                max_x = np.max(x_transformed)
                R[d,:] *= 1.1 / max_x
                r[d] *= 1.1 / max_x

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
        if self.bin_size < 1:
            yhat = smooth(data,20)

        xhat = self.link(np.clip(yhat, 0.01, np.inf))
        xhat = self._invert(xhat, input=input, mask=mask, tag=tag)
        xhat = smooth(xhat,10)

        if self.bin_size < 1:
            # xhat = np.clip(xhat, -0.9, 0.9)
            xhat = np.clip(xhat, -1.02, 1.02)
        # for t in range(xhat.shape[0]):
        #     if np.all(xhat[np.max([0,t-2]):t+3]>0.99) and t>2:
        #     # if np.median(xhat[np.max([0,t-2]):t+3])>0.99 and t>0:
        #         xhat[np.minimum(0,t-2):] = 1.01*np.ones(np.shape(xhat[np.minimum(0,t-2):]))
        #         xhat[:np.maximum(0,t-2)] = 0.0*np.ones(np.shape(xhat[:np.maximum(0,t-2)]))
        #         break
        #     if np.all(xhat[np.max([0,t-2]):t+3]<-0.99) and t>2:
        #     # if np.median(xhat[np.max([0,t-2]):t+3])>0.99 and t>0:
        #         xhat[np.minimum(0,t-2):] = -1.01*np.ones(np.shape(xhat[np.minimum(0,t-2):]))
        #         xhat[:np.maximum(0,t-2)] = 0.0*np.ones(np.shape(xhat[:np.maximum(0,t-2)]))
        #         break

        if np.abs(xhat[0]).any()>1.0:
                xhat[0] = 0.05*npr.randn(1,self.D)
        return xhat

    def initialize(self, base_model, datas, inputs=None, masks=None, tags=None,
                   emission_optimizer="bfgs", num_optimizer_iters=1000):
        print("Initializing Emissions parameters...")

        if self.D == 1 and base_model.transitions.type_name == "DDMTransitions":

            d_init = np.mean([y[0:3] for y in datas],axis=(0,1))
            u_sum = np.array([np.sum(u) for u in inputs])
            y_end = np.array([y[-3:] for y in datas])
            u_l, u_u = np.percentile(u_sum, [20,80]) # use 20th and 80th percentile input
            y_U = y_end[np.where(u_sum>=u_u)]
            y_L = y_end[np.where(u_sum<=u_l)]
            C_init = (1.0/2.0)*np.mean((np.mean(y_U,axis=0) - np.mean(y_L,axis=0)),axis=0)
            self.Cs = C_init.reshape([1,self.N,self.D]) / self.bin_size
            self.ds = d_init.reshape([1,self.N]) / self.bin_size

        else:
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

class RampStepPoissonEmissions(PoissonEmissions):
    def __init__(self, N, K, D, M=0, single_subspace=False, link="softplus", bin_size=1.0):
        super(RampStepPoissonEmissions, self).__init__(N, K, D, M=M, single_subspace=single_subspace, link=link, bin_size=bin_size)
        # Make sure the input matrix Fs is set to zero and never updated
        self.Fs *= 0
        self.C = self._Cs[0]
        self._Cs = self.C * np.ones((K,N,D))

    # Construct an emissions model
    @property
    def params(self):
        return self.C, self.ds

    @params.setter
    def params(self, value):
        self.C, self.ds = value
        self._Cs = self.C * np.ones((self.K,self.N,self.D))

    def _invert(self, data, input=None, mask=None, tag=None):
        """
        Approximate invert the linear emission model with the pseudoinverse

        y = Cx + d + noise; C orthogonal.
        xhat = (C^T C)^{-1} C^T (y-d)
        """
        # assert self.single_subspace, "Can only invert with a single emission model"

        C, F, d = self.Cs[0], self.Fs[0], self.ds[0]
        C_pseudoinv = np.linalg.solve(C.T.dot(C), C.T).T

        # Account for the bias
        bias = input.dot(F.T) + d

        if not np.all(mask):
            data = interpolate_data(data, mask)
            # We would like to find the PCA coordinates in the face of missing data
            # To do so, alternate between running PCA and imputing the missing entries
            for itr in range(25):
                mu = (data - bias).dot(C_pseudoinv)
                data[:, ~mask[0]] = (mu.dot(C.T) + bias)[:, ~mask[0]]

        # Project data to get the mean
        return (data - bias).dot(C_pseudoinv)

    def invert(self, data, input=None, mask=None, tag=None, clip=np.array([0.0,1.0])):
        yhat = smooth(data,20)
        xhat = self.link(np.clip(yhat, 0.01, np.inf))
        xhat = self._invert(xhat, input=input, mask=mask, tag=tag)
        num_pad = 10
        xhat = smooth(np.concatenate((np.zeros((num_pad,self.D)),xhat)),10)[num_pad:,:]
        xhat = np.clip(xhat, -1.1, 1.1)
        if np.abs(xhat[0]).any()>1.0:
                xhat[0] = 0.1*npr.randn(1,self.D)
        return xhat

    def initialize(self, base_model, datas, inputs=None, masks=None, tags=None,
                   emission_optimizer="bfgs", num_optimizer_iters=50):
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

        init_state_distn = AccumulationInitialStateDistribution(K, D, M=M)
        init_state_distn.log_pi0 = np.log(np.concatenate(([0.9999],(0.0001/(K-1))*np.ones(K-1))))

        transition_classes = dict(
            racesoft=AccumulationRaceSoftTransitions,
            race=AccumulationRaceTransitions,
            ddmsoft=DDMSoftTransitions,
            ddm=DDMTransitions,
            ddmcollapsing=DDMCollapsingTransitions)
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
            poisson=AccumulationPoissonEmissions,
            rampstep=RampStepPoissonEmissions)
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
        self.base_model.observations.Vs = self.dynamics.Vs
        self.base_model.observations.As = self.dynamics.As
        self.base_model.observations.Sigmas = self.dynamics.Sigmas
        self.emissions.initialize(self.base_model,
                                  datas, inputs, masks, tags)

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


class AccumulationInitialStateDistribution(InitialStateDistribution):
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
        # do not update the parameters
        self.log_pi0 = self.log_pi0
