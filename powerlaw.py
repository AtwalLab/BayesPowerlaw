from scipy.optimize import newton
from scipy.special import zeta
import scipy as sp
import numpy as np
from scipy.stats import uniform
import matplotlib.pyplot as plt



def power_law(exponent, xmax, sample_size, discrete=True):
    """This function simulates a dataset that follows a powerlaw
    distribution with a given exponent and xmin>=1.
    Inputs:
    -exponent = exponent of the powerlaw distribution equation (y=1/x^exponent),
    -max_clone_size = the maximum possible x value in the simulated dataset.
    -sample_size = wanted number of datapoints in the dataset.
    -discrete = if True - discrete power law is simulated. If False - continuous powerlaw.
    Returns:
    -numpy array of x that has a size of sample_size parameter."""
    if discrete==True:
    #arrange numpy array of number from 1 to xmax+1 in the float format.
    	x = np.arange(1, xmax+1, dtype='float')
    else:
    	x = np.linspace(1, xmax, xmax**2)
    #plug each value into powerlaw equation to start generating probability mass function (pmf)
    pmf = 1/x**exponent
    #divide each pmf value by the sum of all the pmf values. The sum of the resulting
    #pmf array should become 1.
    pmf /= pmf.sum()
    #np.random.choice function generates a random sample of a given sample size
    #from a given 1-D array x, given the probabilities for each value.
    return np.random.choice(x, sample_size, p=pmf)


class Fit(object):

    def __init__(self, data, initial_guess=np.linspace(1,5,10), xmin=1, xmax=None, discrete=True):
        self.data=data
        self.xmin=xmin
        if xmax is None:
            self.xmax=max(self.data)+10.0
        else:
            self.xmax=xmax
        if self.xmin>1 or self.xmax!=np.infty:
            self.data=self.data[(self.data>=self.xmin) & (self.data<=self.xmax)]
        self.n=len(self.data)
        self.constant=np.sum(np.log(self.data))/self.n
        self.discrete=discrete
        if self.discrete==True:
            self.initial_guess=initial_guess
            self.best_guess=self.Guess_discrete()
        else:
            self.best_guess=self.Guess_continuous()

    def Z(self,gamma):
        """The normalization function Z."""  
        if np.isfinite(self.xmax):
            s=0
            for i in range(int(self.xmin),int(self.xmax)+1):
                s+=(1.0/(i**gamma))
        else:
            s=zeta(gamma,self.xmin)
        return s

    def F(self,gamma):
        """The optimization function F(gamma). C is the second term in the definition of F(gamma) and is independent of
        gamma. C need to be defined before running the function. Function Z must be run beforehand as well."""
        h = 1e-8
        Z_prime = (self.Z(gamma+h) - self.Z(gamma-h))/(2*h)
        return (Z_prime/self.Z(gamma))+self.constant

    def Guess_discrete(self):
        best_guess=np.zeros(len(self.initial_guess))
        for i in range(len(self.initial_guess)):
            try:
                best_guess[i]=newton(self.F,self.initial_guess[i])
            except RuntimeError: best_guess[i]=0
        best_guess=best_guess[best_guess!=0]
        log_likelihood=np.zeros(len(best_guess))
        for i in range(len(best_guess)):
            log_likelihood[i]=-self.n*np.log(self.Z(best_guess[i]))-best_guess[i]*np.sum(np.log(self.data))
        self.log_likelihood=log_likelihood
        if len(log_likelihood)==0:
            best_guess=0
        else:
            best_guess=best_guess[np.where(log_likelihood == max(log_likelihood))]
        return best_guess

    def Guess_continuous(self):
        return 1+self.n/(np.sum(np.log(self.data))-self.n*np.log(self.xmin))


class Fit_Bayes(object):
    """This function fits the data to powerlaw distribution and outputs the exponent. 
    If the data consists of mixture of 2 or 3 powerlaws, the function will identify the mixture of exponents 
    as well as the weights each exponent carries.

    parameters
    ----------
    data: (list or np.array of numbers) 
        An array of data from the powerlaw that is being fitted here y=x^(-gamma)/Z.
        All values must be integers or floats from 1 to infinity.

    gamma_range: ([float>1,float<10])
        The first value in the list indicates the lowest possible exponent, which will be used to start the algorithm. 
        The second value is the largest possible exponent. The algorithm will reject any exponent values higher than that.
        Default is [1.01,6.0], since exponent of 1 and lower is mathematically invalid, and exponents above 6 are rare.

    xmin: (int or float >=1)
        The lowest value from the data included in the powerlaw fit. 
        Default value is "None", in which case the minimum value observed in the data is used.
    
    xmax: (int or float >= xmin)
        The highest value from the data included in the powerlaw fit. 
        Default value is "None", in which case the maximum value observed in the data is used.
        To set xmax to infinity use xmax=np.infty.
    
    discrete: (bool) 
        Whether or not the powerlaw is discrete or continuous.
        Default True, in which case powerlaw is assumed to be discrete.
        The distinction is important when estimating partition function Z.

    niters: (int) 
        Number of MCMC iterations performed. The default is 10000.
    
    sigma: (float>0)
        Standard deviation of the sampling step size during MCMC for both gamma and weights.
        Samples are taken from a gaussian distribution with a mean of zero.
        Default value is 0.05, which is seen to perform with good efficiency.
    
    burn_in (int):
        Number of MCMC iterations performs during burn in. 
        The results from burn in are discarded from a final posterior distribution.
        The default is 1000.

    prior: (string)
        Prior type for performing Bayesian inference. Possible values:
        'jeffrey' : derived Jeffrey's prior for fitting powerlaw distributions (default).
        'flat'    : flat prior within the specified gamma range.
    
    mixed: (int>0) 
        Number of distinct powerlaws the dataset is thought to consist of. Default is 1.


    attributes
    ----------
    n: 
        Sample size of the data.
        (int>5)

    params:
        Number of parameters fitted (mixed*2-1), that is number of exponents (mixed) + number of weights (mixed-1),
        where the last weight is 1-[weights].
        (int)

    niters:
        Number of iterations in the main MCMC algorithm. 
        Scales with number of parameters (niters x params).
        (int)

    burn:
        Number of iterations in the burn in MCMC algorithm. 
        Scales with number of parameters (burn_in x params).
        (int)

    gammas:
        An array of 10000 gamma values within the given gamma_range for plotting prior.
        (1D np.array)

    weight:
        An array of 100 weight values from 0 to 1 for plotting prior.
        (1D np.array)

    prior_gamma:
        An array of probabilities of each value in gammas for plotting prior.
        (1D np.array)

    prior_weight:
        An array of probabilities of each value in weights for plotting prior.
        (1D np.array)

    """
    def __init__(self, 
                data, 
                gamma_range=[1.01,6.0], 
                xmin=None, 
                xmax=None, 
                discrete=True, 
                niters=10000, 
                sigma=0.05,
                burn_in=1000, 
                prior='flat', 
                mixed=1):

        #convert data to numpy array in case input is a list.
        self.data=np.array(data) 

        #xmin
        if xmin is None:
            self.xmin = min(self.data)
        else:
            self.xmin=xmin

        #xmax
        if xmax is None:
            self.xmax=max(self.data)+1
        else:
            self.xmax=xmax

        #filter data given xmin and xmax
        if self.xmin>1 or self.xmax!=np.infty:
            self.data=self.data[(self.data>=self.xmin) & (self.data<=self.xmax)]
        
        self.n=len(self.data) #length of data array
        self.mixed = np.arange(mixed) # number of powerlaws in data arranged in the array from 1 to mixed
        self.params = mixed*2-1 # total number of parameters fitted (number of exponents + number of weights where the last weight is equal to 1-[weights])
        self.range=gamma_range #exponent range
        self.discrete=discrete #is data discrete or continuous
        self.prior_model=prior #prior used (jeffrey's (default) or flat)
        self.niters = niters * self.params  # number of iterations in MCMC scaled given number of parameters
        self.sigma=sigma #standard deviation of step size in MCMC
        self.burn=burn_in*self.params # number of iterations in MCMC burn in scaled given number of parameters

        # make array of possible gammas, given the gamma_range, and an array of possible weights.
        self.gammas = np.linspace(1.01, self.range[1], 10000)
        self.weight = np.linspace(0.001, 1, 100)

        if self.prior_model == 'jeffrey':
            #make array of prior function given the jeffrey's prior.
            self.prior_gamma = [1 / (i - 1) for i in self.gammas] #(FIX THIS TO CORRECT PRIOR)
        else:
            #make array of prior function given the flat prior.
            self.prior_gamma=(sp.stats.uniform(self.range[0],self.range[1]-self.range[0])).pdf(self.gammas)
        
        #make array of prior function for weights (flat prior).
        self.prior_weight = (sp.stats.uniform(self.range[0], 1 - 0)).pdf(self.weight)


    def Z(self,gamma):
        """
        Partition function Z for discrete and continuous powerlaw distributions.

        parameters
        ----------
        gamma: (float)
            Randomly sampled target exponent.

        returns
        ------
        s:
            Partition value.

        """
        if self.discrete==True: #when powerlaw is discrete
            if np.isfinite(self.xmax): #if xmax is NOT infinity:
                #Calculate zeta from Xmin to Infinity and substract Zeta from Xmax to Infinity
                #To find zeta from Xmin to Xmax.
                s=zeta(gamma, self.xmin)-zeta(gamma, self.xmax)
            else:
                #if xmax is infinity, simply calculate zeta from Xmin till infinity.
                s=zeta(gamma,self.xmin)
        else:
            #calculate normalization function when powerlaw is continuous.
            #s=(xmax^(-gamma+1)/(1-gamma))-(xminx^(-gamma+1)/(1-gamma))
            s = (self.xmax**(-gamma + 1) / (1 - gamma)) - \
                (self.xmin**(-gamma + 1) / (1 - gamma))
        return s

    def Z_prime(self,gamma):

        h = 1e-8
        s = (self.Z(gamma + h) - self.Z(gamma - h)) / (2 * h)
        return s

    def Z_prime2(self, gamma):
        h = 1e-8
        s = (self.Z(gamma + 2*h) - 2*self.Z(gamma) + self.Z(gamma - 2*h)) / (4*(h**2))
        return s

    def Z_jeffrey(self,gamma):
        z=np.sqrt((-1)*(-self.Z_prime2(gamma))/self.Z(gamma)+self.Z_prime(gamma)**2/self.Z(gamma)**2)
        return z

    def log_prior(self, gamma):
        """
        This function calculates prior given target exponent and prior model.

        parameters
        ----------
        gamma: (float)
            Randomly sampled target exponent.

        returns
        -------
        prior_answer: 
            calculated log of prior.

        """
        if self.prior_model == 'jeffrey':
            #Calculate log of jeffrey's prior
            prior_answer=np.log(self.Z_jeffrey(gamma))
        else:
            #Flat prior: prior=1/(b-a)
            prior_answer = np.log(
                (sp.stats.uniform(self.range[0], self.range[1] - self.range[0])).pdf(gamma))
        return prior_answer

    def L (self,gamma_params,weight_params):
        """
        This function calculates the log likelihood given target exponent and weight values.

        parameters
        ----------

        gamma_params: (1D np.array)
            Array of randomly sampled target exponents for each powerlaw in the data.

        weight_params: (1D np.array)
            Array of randomly sampled target weights for each powerlaw in the data.

        returns
        -------

        lik: 
            calculated log likelihood value.

        """

        lik=0
        for i in range(len(self.mixed)):
            l=(weight_params[i]*self.data**(-gamma_params[i]))/self.Z(gamma_params[i])
            lik=lik+l
        return np.sum(np.log(lik))

    def target (self, gamma_params, weight_params):
        """
        

        parameters
        ----------

        gamma_params: (1D np.array)
            Array of randomly sampled target exponents for each powerlaw in the data.

        weight_params: (1D np.array)
            Array of randomly sampled target weights for each powerlaw in the data.

        returns
        -------

        p: 
            calculated target values - log of (likelihood x prior)

        """
        p=0
        if (np.sum(gamma_params < self.range[0]) != 0) or (np.sum(gamma_params > self.range[1]) != 0) or (np.sum(weight_params < 0) != 0):
            p=0
        else:
            p = self.L(gamma_params,weight_params)+self.log_prior(gamma_params[0])
        return p

    def sample_new(self, gamma_params, weight_params,sigma_g,sigma_w):
        """

        parameters
        ----------

        gamma_params: (1D np.array)
            Array of randomly sampled target exponents for each powerlaw in the data.

        weight_params: (1D np.array)
            Array of randomly sampled target weights for each powerlaw in the data.

        sigma_g:

        sigma_w:

        returns
        -------

        gamma_params_p: 
        
        weight_params_p:


        """
        gamma_params_p = np.zeros(len(self.mixed))
        weight_params_p = np.zeros(len(self.mixed))
        for i in range(len(self.mixed)):
            gamma_params_p[i] = gamma_params[i] + sp.stats.norm(0, sigma_g).rvs()
            if len(self.mixed)>1:
                weight_params_p[i]= weight_params[i] + sp.stats.norm(0, sigma_w).rvs()
            else:
                weight_params_p[i] = weight_params[i]
        weight_params_p[-1] = 1 - np.sum(weight_params_p[0:-1])
        return gamma_params_p, weight_params_p

    def random_walk(self,gamma_params,weight_params,sigma):
        """


        parameters
        ----------

        gamma_params: (1D np.array)
            Array of randomly sampled target exponents for each powerlaw in the data.

        weight_params: (1D np.array)
            Array of randomly sampled target weights for each powerlaw in the data.

        sigma: (float>0)

        returns
        -------

        a: 

        gamma_params_p:

        weight_params_p:


        """
        gamma_params_p, weight_params_p = self.sample_new(gamma_params, weight_params, sigma, sigma)
        target_p = self.target(gamma_params_p,weight_params_p)
        target = self.target(gamma_params,weight_params)
        a=-10**8
        if target_p != 0:
            a = min(0, target_p - target)
        return a, gamma_params_p, weight_params_p

    def burn_in(self,gamma_params,weight_params,sigma):
        """


        parameters
        ----------

        gamma_params: (1D np.array)
            Array of randomly sampled target exponents for each powerlaw in the data.

        weight_params: (1D np.array)
            Array of randomly sampled target weights for each powerlaw in the data.

        sigma: (float>0)

        returns
        -------

        gamma_params:

        weight_params:


        """
        a, gamma_params_p, weight_params_p = self.random_walk(gamma_params,weight_params,sigma)
        if a == 0.0:
            gamma_params = gamma_params_p
            weight_params = weight_params_p
        return gamma_params, weight_params

    def monte_carlo(self,gamma_params, weight_params,sigma):
        """


        parameters
        ----------

        gamma_params: (1D np.array)
            Array of randomly sampled target exponents for each powerlaw in the data.

        weight_params: (1D np.array)
            Array of randomly sampled target weights for each powerlaw in the data.

        sigma: (float>0)

        returns
        -------

        gamma_params:

        weight_params:


        """
        a, gamma_params_p, weight_params_p = self.random_walk(gamma_params,weight_params,sigma)
        if a>np.log(np.random.uniform(0.0, 1.0)):
            gamma_params = gamma_params_p
            weight_params = weight_params_p
        return gamma_params, weight_params

    def posterior (self):
        """


        parameters
        ----------

        None.

        returns
        -------

        gamma_params:

        weight_params:


        """
        sigma_burn = 1.0
        gamma_params = np.array([self.range[0]] * (len(self.mixed)))
        weight_params = np.array([1/len(self.mixed)] * (len(self.mixed)))
        #perform a burn in first without recording gamma values
        for i in range(1,self.burn+1):
            gamma_params, weight_params=self.burn_in(gamma_params,weight_params,sigma_burn)
        #now perform the rest of the sampling while recording gamma values
        samples_gamma = np.zeros([len(self.mixed),self.niters+1])
        samples_gamma[:,0]=gamma_params
        samples_weight = np.zeros([len(self.mixed), self.niters + 1])
        samples_weight[:,0] = weight_params
        for i in range(1,self.niters+1):
            gamma_params, weight_params=self.monte_carlo(gamma_params,weight_params,self.sigma)
            samples_gamma[:,i] = gamma_params
            samples_weight[:, i] = weight_params
        return samples_gamma, samples_weight

    def bic (self, samples_gamma, samples_weight):
        """


        parameters
        ----------

        samples_gamma: (2D np.array)

        samples_weight: (2D np.array)

        returns
        -------

        b:
            Beyesian information criteria (BIC) value.


        """
        gamma_params=np.mean(samples_gamma,axis=1)
        weight_params = np.mean(samples_weight, axis=1)
        b=np.log(self.n)*(len(self.mixed)*2-1)-2*(self.L(gamma_params,weight_params))
        return b


    @staticmethod
    def plot_fit(bayes_object, label=None, color=None):
        data = bayes_object.data
        exponent = bayes_object.best_guess
        xmin = bayes_object.xmin
        xmax = bayes_object.xmax
        discrete=bayes_object.discrete
        

        def Z(exponent,xmin,xmax):
            """The normalization function Z.
            Note that default arguments of xmin and xmax make Z equivalent to Riemann zeta function which is already
            implemented in Scipy as zeta(gamma,1)"""
            if np.isfinite(xmax):
                s=0.0
                for i in range(int(xmin),int(xmax+1)):
                    s+=(1.0/(i**exponent))
            else:
                s=zeta(exponent,xmin)
            return s
        
        def powerlawpdf(data, exponent, xmin):
            """The power law probability function
            Input: x - array of clone sizes; gamma - desired exponent.
            Output: array of probabilities of each clone size x."""
            if discrete==True:
                powerlaw=(data**(-exponent))/Z(exponent,xmin,xmax)
            else:
                powerlaw=((data)**(-exponent))*(xmin**(exponent-1)/(exponent-1))
            return powerlaw
        
        if discrete==True:
            unique, counts = np.unique(data, return_counts=True)
            frequency=counts/np.sum(counts)
            xpmax=((Z(exponent,xmin,xmax)/len(data))**(-1/exponent))*2
            xp=np.arange(np.min(data),xpmax)
            plt.loglog(unique,frequency,'o', color=color, markeredgecolor='black', label=label)
            plt.plot(xp,powerlawpdf(xp,exponent,xmin), color=color, linewidth=2)
            plt.legend()
        else:
            xp=np.arange(np.min(data),np.max(data)+1)
            plt.hist(data, bins=50, color=color, label=label)
            plt.plot(xp,powerlawpdf(xp,exponent,xmin), color=color, linewidth=2)
            plt.xscale('log')
            plt.yscale('log')
            plt.legend()
        
        return 
        
        
    @staticmethod
    def plot_prior(bayes_object, color=None, label=None):
        plt.plot(bayes_object.gammas, bayes_object.prior, color=color, label=label)
        return
    
    @staticmethod
    def plot_posterior1(bayes_object, bins=100, alpha=0.5, color=None, label=None, range=None):
        plt.hist(bayes_object.samples, bins, alpha=alpha, color=color, label=label, range=range, normed=True)
        plt.legend()
        return

    


