import autograd.numpy as np
import autograd.numpy.random as npr
from ssm.core import BaseHMM, BaseSwitchingLDS
from ssm.init_state_distns import InitialStateDistribution
from ssm.transitions import _Transitions, RecurrentOnlyTransitions
from ssm.observations import _Observations, AutoRegressiveDiagonalNoiseObservations
from ssm.emissions import _LinearEmissions, GaussianEmissions, PoissonEmissions

"""
Observed DDM
"""
class DDMObservations(AutoRegressiveDiagonalNoiseObservations):
    def __init__(self, K, D, M=1, lags=1, beta=1.0, sigmas=1e-3 * np.ones((3, 1))):
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
        
    def initialize(self, datas, inputs=None, masks=None, tags=None, covariances=None):
        pass

    def m_step(self, expectations, datas, inputs, masks, tags, covariances, **kwargs):
        _Observations.m_step(self, expectations, datas, inputs, masks, tags, covariances, **kwargs)
        

# Do the same for the transition model
class DDMTransitions(RecurrentOnlyTransitions):
    def __init__(self, K, D, M=0, scale=100):
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
    
    def initialize(self, datas, inputs=None, masks=None, tags=None, covariances=None):
        pass
    
    def m_step(self, expectations, datas, inputs, masks, tags, covariances, **kwargs):
        pass


def DDM(beta=1.0, sigmas=np.array([[1e-5], [1e-3], [1e-5]])):
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

    def initialize(self, datas, inputs=None, masks=None, tags=None, covariances=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        pca = self._initialize_with_pca(datas, inputs=inputs, masks=masks, tags=tags, covariances=None)
        self.inv_etas[:,...] = np.log(pca.noise_variance_)
        
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

    def initialize(self, datas, inputs=None, masks=None, tags=None, covariances=None):
        pass

    def m_step(self, expectations, datas, inputs, masks, tags, covariances, **kwargs):
        _Observations.m_step(self, expectations, datas, inputs, masks, tags, covariances, **kwargs)

    
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
        
    def initialize(self, datas, inputs=None, masks=None, tags=None, covariances=None):
        pass

    def m_step(self, expectations, datas, inputs, masks, tags, covariances, **kwargs):
        _Observations.m_step(self, expectations, datas, inputs, masks, tags, covariances, **kwargs)
    

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
    
    def initialize(self, datas, inputs=None, masks=None, tags=None, covariances=None):
        pass
    
    def m_step(self, expectations, datas, inputs, masks, tags, covariances, **kwargs):
        pass
    
def Accumulator2D(D=2, M=2, betas=np.ones(2,), sigmas=np.array([[2e-4,1e-4],[3e-4,5e-4],[1e-4,2e-4]]), a_diag=np.ones((3,2,1))):
    K, D, M = 3, 2, 2
    
    # Build the initial state distribution, the transitions, and the observations
    init_state_distn = InitialStateDistribution(K, D, M)
    init_state_distn.log_pi0 = np.log([0.01, 0.98, 0.01])
    transition_distn = Accumulator2DTransitions(K, D, M)
    observation_distn = Accumulator2DObservations(K, D, M, betas=betas, sigmas=sigmas, a_diag=a_diag)
    
    return BaseHMM(K, D, M, init_state_distn, transition_distn, observation_distn)