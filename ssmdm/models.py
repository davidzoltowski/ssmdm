import autograd.numpy as np
import autograd.numpy.random as npr
from ssm.core import BaseHMM, BaseSwitchingLDS
from ssm.init_state_distns import InitialStateDistribution
from ssm.transitions import _Transitions, RecurrentOnlyTransitions
from ssm.observations import _Observations, AutoRegressiveDiagonalNoiseObservations, _AutoRegressiveObservationsBase
from ssm.emissions import _LinearEmissions, GaussianEmissions, PoissonEmissions

# preprocessing
from ssm.preprocessing import factor_analysis_with_imputation
from tqdm.auto import trange

# for initialization
from ssm.optimizers import adam_step, rmsprop_step, sgd_step, convex_combination
from autograd.scipy.misc import logsumexp
from autograd.tracer import getval
from autograd.misc import flatten
from autograd import value_and_grad

"""
Observed DDM
"""
class DDMObservations(AutoRegressiveDiagonalNoiseObservations):
    def __init__(self, K, D=1, M=1, lags=1, beta=1.0, sigmas=1e-3 * np.ones((3, 1))):
        assert K == 3
        assert D == 1
        assert M == 1
        super(DDMObservations, self).__init__(K, D, M)
        
        # The only free parameters of the DDM are the ramp rate...
        self.beta = beta
        
        # and the noise variances, which are initialized in the AR constructor
        self._log_sigmasq = np.log(sigmas)
        
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
        return self.beta, self._log_sigmasq
        
    @params.setter
    def params(self, value):
        self.beta, self._log_sigmasq = value
        mask = np.reshape(np.array([0, 1, 0]), (3, 1, 1))
        self.Vs = mask * self.beta
        
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        pass

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        _Observations.m_step(self, expectations, datas, inputs, masks, tags, **kwargs)
        

# Do the same for the transition model
class DDMTransitions(RecurrentOnlyTransitions):
    def __init__(self, K, D, M=0, scale=500):
        assert K == 3
        assert D == 1
        assert M == 1
        super(DDMTransitions, self).__init__(K, D, M)

        # Parameters linking past observations to state distribution
        self.Ws = np.zeros((3, 1))
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


def DDM(D=1, M=1, beta=1.0, sigmas=np.array([[1e-4], [1e-4], [1e-4]])):
    K, D, M = 3, 1, 1
    
    # Build the initial state distribution, the transitions, and the observations
    init_state_distn = InitialStateDistribution(K, D, M)
    init_state_distn.log_pi0 = np.log([0.01, 0.98, 0.01])
    transition_distn = DDMTransitions(K, D, M)
    observation_distn = DDMObservations(K, D, M, beta=beta, sigmas=sigmas)
    
    return BaseHMM(K, D, M, init_state_distn, transition_distn, observation_distn)

"""
Latent DDM
"""
class DDMGaussianEmissions(GaussianEmissions):
    def __init__(self, N, K, D, M=0, single_subspace=True):
        super(DDMGaussianEmissions, self).__init__(N, K, D, M=M, single_subspace=single_subspace)
        # Make sure the input matrix Fs is set to zero and never updated
        self.Fs *= 0

    # Construct an emissions model
    @property
    def params(self):
        return self._Cs, self.ds, self.inv_etas

    @params.setter
    def params(self, value):
        self._Cs, self.ds, self.inv_etas = value

    def initialize(self, datas, inputs=None, masks=None, tags=None, num_em_iters=50, num_tr_iters=50):
        print("Initializing...")
        print("First with FA using {} steps of EM.".format(num_em_iters))
        fa, xhats, Cov_xhats, lls = factor_analysis_with_imputation(self.D, datas, masks=masks, num_iters=num_em_iters)
        
        # define objective
        def _objective(params, itr):
            Td = sum([x.shape[0] for x in xhats])
            new_datas = [np.dot(x,params[0].T)+params[1] for x in xhats]
            obj = DDM().log_likelihood(new_datas,inputs=inputs)
            return -obj / Td
        
        # initialize R and r
        R = 0.5*np.random.randn(self.D,self.D)
        r = 0.0
        params = [R,r]
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
        
        self.Cs = (fa.W @ np.linalg.inv(R)).reshape([1,self.N,self.D])
        self.ds = fa.mean - np.squeeze(fa.W @ np.linalg.inv(R) * r)
        self.inv_etas = np.log(fa.sigmasq)
        
def LatentDDM(N, beta=1.0, sigmas=np.array([[1e-5], [1e-3], [1e-5]])):
    K, D, M = 3, 1, 1
    
    # Build the initial state distribution, the transitions, and the observations
    init_state_distn = InitialStateDistribution(K, D, M)
    init_state_distn.log_pi0 = np.log([0.01, 0.98, 0.01])
    transition_distn = DDMTransitions(K, D, M)
    dynamics_distn = DDMObservations(K, D, M, beta=beta, sigmas=sigmas)

    emission_distn = DDMGaussianEmissions(N, K, D, M=M, single_subspace=True)

    # Make the SLDS
    return BaseSwitchingLDS(N, K, D, M, init_state_distn, transition_distn, dynamics_distn, emission_distn)

"""
1D Accumulator
"""
class _AccumulatorObservationsBase(_AutoRegressiveObservationsBase):
    """
    This class specifies the mean of general accumulator observations. It has three discrete states and can
    handle arbitrary continuous and input dimensions. It does not specify the covariances.
    """
    def __init__(self, K, D, M, lags=1):
        assert K == 3
        super(_AccumulatorObservationsBase, self).__init__(K, D, M)
        
        # the parameters are the input weights - one input weight for each dimension of the input
        self._betas = np.ones(M)
        
        # Initialize input weights to zero, besides the diagonal in accumulation state
        self.Vs[0] *= np.zeros((D,M))            # left bound
        self.Vs[1] = self._betas*np.eye(D,M)       # ramp
        self.Vs[2] *= np.zeros((D,M))  
        
        # learn diagonal autoregressive dynamics for each state 
        # a_diag are parameters of the diagonal for each state
        self._a_diag = np.ones(K,D,1)
        mask = np.array([np.eye(D), np.eye(D), np.eye(D)])
        self._As = self.a_diag * mask
        
        # fix the remaining values
        self.bs = np.zeros((K, D))
        self.mu_init = np.zeros((K, D))

    @property
    def params(self):
        return self._betas, self._a_diag
        
    @params.setter
    def params(self, value):
        self._betas, self._a_diag = value
        
        # set V
        v_mask = np.array([np.zeros((D,M)), np.eye(D,M), np.zeros((D,M))])
        self.Vs = mask * self._betas
        
        # set A
        a_mask = np.array([np.eye(D), np.eye(D), np.eye(D)])
        self._As = self._a_diag * a_mask

    def initialize(self, datas, inputs=None, masks=None, tags=None):
        pass

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        _Observations.m_step(self, expectations, datas, inputs, masks, tags, **kwargs)
        
class AccumulatorDiagonalInputNoiseObservations(_AccumulatorObservationsBase):
    def __init__(self, K, D, M):
        assert K == 3
        super(AccumulatorDiagonalInputNoiseObservations, self).__init__(K, D, M)
        
        # initialize initial state covariance
        self._log_sigmasq_init = np.zeros((K, D))

        # have two diagonal noise covariances, one without input and one with input
        self._log_sigmasq = np.zeros((2, K, D))
        
    
    # sigma init 
    @property
    def sigmasq_init(self):
        return np.exp(self._log_sigmasq_init)

    @sigmasq_init.setter
    def sigmasq_init(self, value):
        assert value.shape == (self.K, self.D)
        assert np.all(value > 0)
        self._log_sigmasq_init = np.log(value)
        
    @property
    def Sigmas_init(self):
        return np.array([np.diag(np.exp(log_s)) for log_s in self._log_sigmasq_init])
    
    @property
    def sigmasq(self):
        return np.exp(self._log_sigmasq)
    
    # sigma
    @sigmasq.setter
    def sigmasq(self, value):
        assert value.shape == (2, self.K, self.D)
        assert np.all(value > 0)
        self._log_sigmasq = np.log(value)
        
    @property
    def Sigmas(self):
        return np.array([ [np.diag(np.exp(log_s)) for log_s in log_sigs] for log_sigs in log_sigmasq])

    @Sigmas.setter
    def Sigmas(self, value):
        assert value.shape == (2, self.K, self.D, self.D)
        sigmasq = np.array([ [np.diag(S) for S in Sigs] for Sigs in value])
        assert np.all(sigmasq > 0)
        self._log_sigmasq = np.log(sigmasq)
        
    @property
    def params(self):
        return super(AccumulatorDiagonalInputNoiseObservations, self).params + (self._log_sigmasq,)

    @params.setter
    def params(self, value):
        self._log_sigmasq = value[-1]
        super(AccumulatorDiagonalInputNoiseObservations, self.__class__).params.fset(self, value[:-1])
    
    def permute(self, perm):
        super(AccumulatorDiagonalInputNoiseObservations, self).permute(perm)
        self._log_sigmasq_init = self._log_sigmasq_init[perm]
        self._log_sigmasq = self._log_sigmasq[perm]
        
    # def initialize(self, datas, inputs=None, masks=None, tags=None, localize=True):
    # take into account the two different covariances
    
    # def log_likelihoods(self, data, input, mask, tag=None):
    # also compute log-likelihood given sigma, depending on input 

    # def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
    # change sigmas to get the correct sigma, depending on input 
    
    # def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
    
    
class AccumulatorObservations(DDMObservations):
    def __init__(self, K, D, M=1, lags=1, beta=1.0, sigmas=1e-3 * np.ones((3, 1)), As=np.ones((3,1,1))):
        super(AccumulatorObservations, self).__init__(K, D, M, beta=beta, sigmas=sigmas)

        # learn diagonal autoregressive dynamics 
        self._As = As

    @property
    def params(self):
        return self.beta, self._log_sigmasq, self._As

    @params.setter
    def params(self, value):
        self.beta, self._log_sigmasq, self._As = value
        mask = np.reshape(np.array([0, 1, 0]), (3, 1, 1))
        self.Vs = mask * self.beta

    def initialize(self, datas, inputs=None, masks=None, tags=None):
        pass

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        _Observations.m_step(self, expectations, datas, inputs, masks, tags, **kwargs)
    
def Accumulator(beta=1.0, sigmas=np.array([[1e-5], [1e-3], [1e-5]]), As=np.ones((3, 1, 1))):
    K, D, M = 3, 1, 1
    
    # Build the initial state distribution, the transitions, and the observations
    init_state_distn = InitialStateDistribution(K, D, M)
    init_state_distn.log_pi0 = np.log([0.01, 0.98, 0.01])
    transition_distn = DDMTransitions(K, D, M)
    observation_distn = AccumulatorObservations(K, D, M, beta=beta, sigmas=sigmas, As=As)
    
    return BaseHMM(K, D, M, init_state_distn, transition_distn, observation_distn)


"""
2D Accumulator
TODO -> the V matrix should be D by M, or you should assert that D = M. 
"""
class Accumulator2DObservations(AutoRegressiveDiagonalNoiseObservations):
    def __init__(self, K, D=2, M=0, lags=1, betas=np.ones(2,), sigmas=1e-3 * np.ones((3, 2)), a_diag=np.ones((3,2,1))):
        assert K == 3
        assert D == 2
        super(Accumulator2DObservations, self).__init__(K, D, M)
        
        # dynamics matrix for accumulation state
        self._a_diag = a_diag
        mask = np.array([np.eye(D), np.eye(D), np.eye(D)])
        self._As = self._a_diag*mask

        # set input accumulator params, one for each dimension
        self._betas = betas
        
        # They only differ in their input 
        self.Vs[0] *= np.zeros((D,D))            # left bound
        self.Vs[1] = self._betas*np.eye(D)       # ramp
        self.Vs[2] *= np.zeros((D,D))            # right bound
        
        # set noise variances, which are initialized in the AR constructor
        self._log_sigmasq = np.log(sigmas)
        
        # Set the remaining parameters to fixed values
        self.bs = np.zeros((3, D))
        self.mu_init = np.zeros((3, D))
        self._log_sigmasq_init = np.log(.01 * np.ones((3, D)))
        
    @property
    def params(self):
        return self._betas, self._log_sigmasq, self._a_diag
        
    @params.setter
    def params(self, value):
        D = self.D
        self._betas, self._log_sigmasq, self._a_diag = value
        mask = np.array([np.zeros((D,D)), np.eye(D), np.zeros((D,D))])
        self.Vs = mask * self._betas
        a_mask = np.array([np.eye(D), np.eye(D), np.eye(D)])
        self._As = self._a_diag*a_mask       
        
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        pass

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        _Observations.m_step(self, expectations, datas, inputs, masks, tags, **kwargs)
    

# Transition model
class Accumulator2DTransitions(RecurrentOnlyTransitions):
    def __init__(self, K, D, M=0, scale=100):
        assert K == 3
        assert D == 2
        assert M == 2
        super(Accumulator2DTransitions, self).__init__(K, D, M)
        
        self.Ws = np.zeros((K,M))
        self.Rs = np.array([[scale, -scale], [0, 0], [-scale, scale]]).reshape((K,D)) # K by D
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

#def Accumulator2D(D=2, M=2, betas=np.ones(2,), sigmas=np.array([[2e-4,1e-4],[3e-4,5e-4],[1e-4,2e-4]]), a_diag=np.ones((3,2,1))):
def Accumulator2D(D=2, M=2, betas=np.ones(2,), sigmas=1e-3 * np.ones((3, 2)), a_diag=np.ones((3,2,1))):
    K, D, M = 3, 2, 2
    
    # Build the initial state distribution, the transitions, and the observations
    init_state_distn = InitialStateDistribution(K, D, M)
    init_state_distn.log_pi0 = np.log([0.01, 0.98, 0.01])
    transition_distn = Accumulator2DTransitions(K, D, M)
    observation_distn = Accumulator2DObservations(K, D, M, betas=betas, sigmas=sigmas, a_diag=a_diag)
    
    return BaseHMM(K, D, M, init_state_distn, transition_distn, observation_distn)