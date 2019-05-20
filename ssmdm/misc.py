import numpy as np
import numpy.random as npr
import itertools
from scipy.stats import multivariate_normal

def generate_clicks(T=1.0,dt=0.01,rate_r=20,rate_l=20):
    """
    This function generates right and left 'clicks' from two Poisson processes with rates rate_r and rate_l
    over T seconds with bin sizes dt. The outputs are binned clicks into discrete time bins.
    """

    # number of clicks
    num_r = npr.poisson(rate_r*T)
    num_l = npr.poisson(rate_l*T)

    # click times
    click_time_r = np.sort(npr.uniform(low=0.0,high=T,size=[num_r,1]))
    click_time_l = np.sort(npr.uniform(low=0.0,high=T,size=[num_l,1]))

    # binned outputs are arrays with dimensions Tx1
    binned_r = np.histogram(click_time_r,np.arange(0.0,T+dt,dt))[0]
    binned_l = np.histogram(click_time_l,np.arange(0.0,T+dt,dt))[0]

    return binned_r, binned_l

def factor_analysis(D, ys, num_iters=15):
    # D is number of latent dimensions
    # ys is list of data points

    # concatenate ys
    all_y = np.array(list(itertools.chain(*ys)))

    # observation dimensions
    Nobs,N = np.shape(all_y)

    # compute mean across column
    mu_y = np.mean(all_y,axis=0,keepdims=True)

    # subtract mean
    ys_zero = all_y - mu_y

    # initialize C, Psi
    Cfa = np.random.randn(N,D)
    psi = np.eye(N)

    # run EM
    pbar = trange(num_iters)
    lls = []
    for i in pbar:

        # E-step
        lamb = np.linalg.inv(Cfa.T@np.linalg.inv(psi)@Cfa + np.eye(D))
        mu_x = (lamb@Cfa.T@np.linalg.inv(psi)@(ys_zero.T)).T
        mu_xxT = [lamb + np.outer(mu_x[i,:],mu_x[i,:]) for i in range(Nobs)]

        # M-step
        Cfa = ( np.linalg.inv((mu_x.T@mu_x) + Nobs * lamb) @ mu_x.T @ ys_zero ).T
        np.fill_diagonal(psi,  np.diag( (1.0 / Nobs) * (ys_zero.T @ ys_zero - ys_zero.T @ mu_x @ Cfa.T)))

        # add small elements to diagonal of psi for stability (TODO: add condition)
        np.fill_diagonal(psi,  np.diag(psi +1e-7*np.eye(N)))

        # compute log likelihood
        log_py = np.sum(multivariate_normal.logpdf(all_y, mean=mu_y[0,:], cov=(Cfa@Cfa.T + psi)))
        lls += [log_py]

        pbar.set_description("Itr {} LP: {:.1f}".format(i, lls[-1]))
        pbar.update(1)

    # get xhats
    my_xhats = [(lamb@Cfa.T@np.linalg.inv(psi)@(y - mu_y[0,:]).T).T for y in ys]

    return Cfa, my_xhats, lls , psi

def smooth(xs, window_size=5):
    # window size is number of bins on each side*2, +1

    T,N = np.shape(xs)
    x_smooth = np.zeros(np.shape(xs))

    # win is number of bins on each side
    win = int( (window_size - 1) / 2 )

    for t in range(T):
        smooth_window = np.arange(np.maximum(t-win,0),np.minimum(t+win,T-1))
        x_smooth[t,:] = np.mean(xs[smooth_window,:],axis=0)

    return x_smooth
