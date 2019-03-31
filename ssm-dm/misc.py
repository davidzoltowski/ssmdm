import numpy as np
import numpy.random as npr

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