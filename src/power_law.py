import numpy as np

def power_law(exponent, xmax, sample_size, xmin=1, discrete=True):
    """This function simulates a dataset that follows a powerlaw
    distribution with a given exponent and xmax.


    parameters
    ----------

    exponent: (float>1) 
        exponent of the powerlaw distribution equation (y=1/x^exponent).

    xmax: (int or float > xmin)
        the maximum possible x value in the simulated dataset.

    sample_size: (int>5)
        samples size of the simulated dataset.

    xmin: (int or float >=1)
        the minimum possible x value in the simulated dataset (default: 1).

    discrete: (bool) 
        whether the simulated powerlaw contains discrete (True) or continuous (False) values (default: True)
    
    returns
    -------

    1D array of X'es that has a size of sample_size parameter.
    
    """
    if discrete == True:
        #arrange numpy array of number from 1 to xmax+1 in the float format.
    	x = np.arange(xmin, xmax + 1, dtype='float')
    else:
    	x = np.linspace(xmin, xmax, xmax**2)
    #plug each value into powerlaw equation to start generating probability mass function (pmf)
    pmf = 1 / x**exponent
    #divide each pmf value by the sum of all the pmf values. The sum of the resulting
    #pmf array should become 1.
    pmf /= pmf.sum()
    #np.random.choice function generates a random sample of a given sample size
    #from a given 1-D array x, given the probabilities for each value.
    return np.random.choice(x, sample_size, p=pmf)
