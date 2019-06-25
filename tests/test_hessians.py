import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad, hessian
from scipy.linalg import block_diag
import matplotlib.pyplot as plt

def softplus(x):
    return np.log1p(np.exp(x))

# generate fake data
N = 10 # number of observations
D = 2 # number of covariates
T = 100 # number of time points

x = npr.randn(T,D)
C = npr.randn(N,D)
d = npr.randn(N)

lambdas = softplus( np.dot(x, C.T) + d)
y = npr.poisson(lambdas)

# compute Hessian wrt x of log-likelihood using autograd
def obj(x):
    lambdas = softplus( np.dot(x, C.T) + d)
    obj = np.sum(y*np.log(lambdas)) - np.sum(lambdas) # + const
    return obj

g = grad(obj)
hess = hessian(obj)
hessian_autograd = hess(x).reshape((T*D),(T*D))

# compute Hessian wrt x of log-likelihood analytically
# use lambdas from above
expterms = np.exp( -np.dot(x, C.T) - d)
diags = (y / lambdas * (expterms - 1.0 / lambdas) - expterms ) / (1.0+expterms)**2
hessian_diag = np.einsum('tn, ni, nj ->tij', diags, C, C)
hessian_analytical = block_diag(*hessian_diag)

# norm
print("Norm of difference: ", np.linalg.norm(hessian_analytical - hessian_autograd))

#
im_max = 25
plt.ion()
clims = ( np.min((hessian_autograd, hessian_analytical)), np.max((hessian_autograd, hessian_analytical)) )
plt.figure(figsize=[12,4])
plt.subplot(131)
plt.title("autograd")
plt.imshow(hessian_autograd[:im_max,:im_max], aspect="auto")
plt.clim(clims)
plt.colorbar()
plt.subplot(132)
plt.title("analytical")
plt.imshow(hessian_analytical[:im_max,:im_max], aspect="auto")
plt.clim(clims)
plt.colorbar()
plt.tight_layout()
plt.subplot(133)
plt.title("difference")
plt.imshow(hessian_autograd[:im_max,:im_max]-hessian_analytical[:im_max,:im_max], aspect="auto")
# plt.clim(clims)
plt.clim([-1e-4,1e-4])
plt.colorbar()
plt.tight_layout()

# check allclose
assert np.allclose(hessian_autograd, hessian_analytical)
