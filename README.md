# How to use the code

To run the Fit_Bayes code with given defaults:

    Fit_Bayes(data)

where data is a list or array of numbers from 1 to infinity.


Defaults:

    gamma_range=[1,6] - so far the default prior is flat within the exponent range 1 to 6. Change the range by inputing your own numbers a and b: gamma_range=[a,b].

    xmin=1,xmax=np.infty - will fit the whole distribution. Rectrict the fit by inputing your own xmin and xmax, e.g. xmin=3,xmax=100.

    descrete=True - assumes that the distribution being fitted is discrete. If distribution is continuous change - discrete=False.

    niters=5000 - MCMC will perform 5000 iterations. Change to a desired number of iterations, e.g. niters=1000.

    sigma=2.0 - default variance for selecting step size from gaussian distribution during MCMC. To increase efficiency of MCMC sigma should be very small, e.g. sigma=10**-5.
    
Choose a prior:
Currently there are 3 prior options implemented. 

    1. Powerlaw (default). To change exponent of the powerlaw prior when calling Fit_Bayes(data) 
    add prior=['powerlaw', desired_exponent]. Note that exponent must be >=1. 
    
    2. Exponential. To change prior to exponential with a wanted power: prior=['exponential', desired_power]. 
    Desired power can range from 0.01 to infinity. Note that values from 0.01 to 0.1 work best. 
    
    3. Flat. To select flat prior: prior=['',0].
    
Plotting:

    Object = Fit_Bayes(data)
    
Plot the fit:

    Fit_Bayes.plot_fit(Object, label=None, color=None)

Plot the prior:

    Fit_Bayes.plot_prior(Object, label=None, color=None)
    
Plot the likelihood:

    Fit_Bayes.plot_likelihood(Object, label=None, color=None)
    
Plot the posterior:

    Fit_Bayes.plot_posterior(Object, bins=100, alpha=0.5, color=None, label=None, range=None)
