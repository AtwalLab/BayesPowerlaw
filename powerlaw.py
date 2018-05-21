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
    """
    This function fits the data to powerlaw distribution and outputs the exponent. 
    If the data consists of mixture of 2 or 3 powerlaws, the function will identify the mixture of exponents 
    as well as the weights each exponent carries.

    The object is constructed of:
        *data - an array or list of data points to fit.
        *gamma_range - a range of exponents valid for fitting. The default is from 1 to 6, since exponent below one is
            mathematically invalid, and exponents above 6 are very rare. 
        *xmin and xmax - minimum and maximum data values used for fitting the powerlaw. Default xmin is 1 and default xmax is none.
            When xmax is none, the function calculates xmax by determining largest data value and adding 10.
        *discrete=True means that the data consists of discrete values and will be fitted as such. If descrete=False,
            the data consists of continuous values and will be fitted accordingly. It's important to identify whether your
            data is discrete or continuous before doing the fitting as normalization function differs between the two.
        *niters - a number of MCMC iterations performed. The default is 5000, since this is sufficient to acquire the
            exponent with high confidence and takes around 30 seconds.
        *sigma - standard deviation of the step size during MCMC. The step size is samples from a gaussian distribution with
            the mean of zero and standard deviation sigma.
        *prior - two prior functions are available. Prior='none' (default) produces a flat prior within the specified gamma range.
            prior='jeffrey' produces a derived jeffrey's prior 1/gamma-1.
        *mixed - how many different powerlaws the data set is constructed of (up to 3).
    """
    def __init__(self, data, gamma_range=[1,6], xmin=1, xmax=None, discrete=True, niters=10000, sigma=0.05, burn_in=1000, prior='flat', mixed=1):
        self.data=np.array(data) #convert data to numpy array.
        self.xmin=xmin #xmin
        if xmax is None: #calculate xmax if unspecified
            self.xmax=max(self.data)+1
        else:
            self.xmax=xmax #xmax if specified
        if self.xmin>1 or self.xmax!=np.infty: #filter data given xmin and xmax
            self.data=self.data[(self.data>=self.xmin) & (self.data<=self.xmax)]
        self.n=len(self.data) #length of data array
        self.mixed = np.arange(mixed)  # number of powerlaws in data
        self.params = mixed*2-1
        self.range=gamma_range #exponent range
        self.discrete=discrete #is data discrete or continuous?
        self.prior_model=prior #prior used (flat (default) vs. jeffrey's)
        self.niters=niters #number of iterations in MCMC
        self.sigma=sigma #standard deviation of step size in MCMC
        self.burn=burn_in*self.params

        # make array of possible gammas, given the gamma_range
        self.gammas = np.linspace(1.01, self.range[1], 10000)
        self.weight = np.linspace(0.001, 1, 100)

        if self.prior_model == 'jeffrey':
            #make array of prior function given the jeffrey's prior.
            self.prior_gamma = [1 / (i - 1) for i in self.gammas]
        else:
            #make array of prior function given the flat prior.
            self.prior_gamma=(sp.stats.uniform(self.range[0],self.range[1]-self.range[0])).pdf(self.gammas)
        self.prior_weight = (sp.stats.uniform(self.range[0], 1 - 0)).pdf(self.weight)

        self.samples_gamma, self.samples_weight = self.posterior()
        self.bic=self.bayes_ic()

    def log_prior(self,gamma):
        """
        This function calculates prior given target exponent.
        Input: gamma - a randomly sampled target exponent using MCMC algorithm.
        Output: prior_answer - calculated log prior given the exponent gamma.
        """
        if self.prior_model=='jeffrey':
            #Jeffrey's prior for continuous powerlaw distribution: prior=1/(gamma-1)
            prior_answer=np.log(1/(gamma-1))
        else:
            #Flat prior: prior=1/(b-a)
            prior_answer=np.log((sp.stats.uniform(self.range[0],self.range[1]-self.range[0])).pdf(gamma))
        return prior_answer

    def Z(self,gamma):
        """
        The normalization function Z for discrete and continuous powerlaw distributions.
        Input: gamma - a randomly sampled target exponent using MCMC algorithm.
        Output: s - normalization value.
        """
        if self.discrete==True: #when powerlaw is discrete
            if np.isfinite(self.xmax): #if xmax is NOT infinity:
                #Calculate zeta from Xmin to Infinity and substract Zeta from Xmax to Inifinity
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
    
    def L (self,gamma_params,weight_params):
        """
        Pure powerlaw.
        This function calculates the log likelihood given a sampled gamma in MCMC algorithm.
        Input: gamma - a randomly sampled target exponent using MCMC algorithm.
        Output: lik - log likelihood value.
        """
        lik=0
        for i in range(len(self.mixed)):
            l=(weight_params[i]*self.data**(-gamma_params[i]))/self.Z(gamma_params[i])
            lik=lik+l
        return np.sum(np.log(lik))

    def target (self, gamma_params, weight_params):
        p=0
        if (np.sum(gamma_params <= self.range[0]) != 0) or (np.sum(gamma_params > self.range[1]) != 0) or (np.sum(weight_params < 0) != 0):
            p=0
        else:
            p = self.L(gamma_params,weight_params)+self.log_prior(gamma_params[0])
        return p

    def sample_new(self, gamma_params, weight_params,sigma_g,sigma_w):
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
        gamma_params_p, weight_params_p = self.sample_new(gamma_params, weight_params, sigma, sigma)
        target_p = self.target(gamma_params_p,weight_params_p)
        target = self.target(gamma_params,weight_params)
        a=-10**8
        if target_p != 0:
            a = min(0, target_p - target)
        return a, gamma_params_p, weight_params_p

    def burn_in(self,gamma_params,weight_params,sigma):
        a, gamma_params_p, weight_params_p = self.random_walk(gamma_params,weight_params,sigma)
        if a == 0.0:
            gamma_params = gamma_params_p
            weight_params = weight_params_p
        return gamma_params, weight_params

    def monte_carlo(self,gamma_params, weight_params,sigma):
        a, gamma_params_p, weight_params_p = self.random_walk(gamma_params,weight_params,sigma)
        if a>np.log(np.random.uniform(0.0, 1.0)):
            gamma_params = gamma_params_p
            weight_params = weight_params_p
        return gamma_params, weight_params

    def posterior (self):
        sigma_burn = 1.0
        gamma_params = np.array([1.01] * (len(self.mixed)))
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

    def bayes_ic (self):
        gamma_params=np.mean(self.samples_gamma,axis=1)
        weight_params = np.mean(self.samples_weight, axis=1)
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
    
    # @staticmethod
    # def plot_likelihood(bayes_object, color=None, label=None):
    #     plt.plot(bayes_object.gammas,bayes_object.log_likelihood, color=color, label=label)
    #     plt.legend()
    #     return
    
    @staticmethod
    def plot_posterior1(bayes_object, bins=100, alpha=0.5, color=None, label=None, range=None):
        plt.hist(bayes_object.samples, bins, alpha=alpha, color=color, label=label, range=range, normed=True)
        plt.legend()
        return

    @staticmethod
    def plot_posterior2(bayes_object, bins=100, alpha=0.5, color=None, label=None, range=None):
        plt.hist(bayes_object.samples_1, bins, alpha=alpha,
                 color='red', label='gamma 1', range=range, normed=True)
        plt.hist(bayes_object.samples_2, bins, alpha=alpha,
                 color='blue', label='gamma 2', range=range, normed=True)
        plt.hist(bayes_object.samples_w, bins, alpha=alpha,
                 color='black', label='weight 1', range=range, normed=True)
        plt.legend()
        return

    @staticmethod
    def plot_posterior3(bayes_object, bins=100, alpha=0.5, color=None, label=None, range=None):
        plt.hist(bayes_object.samples_1, bins, alpha=alpha,
                 color='red', label='gamma 1', range=range, normed=True)
        plt.hist(bayes_object.samples_2, bins, alpha=alpha,
                 color='blue', label='gamma 2', range=range, normed=True)
        plt.hist(bayes_object.samples_3, bins, alpha=alpha,
                 color='green', label='gamma 3', range=range, normed=True)
        plt.hist(bayes_object.samples_w1, bins, alpha=alpha,
                 color='black', label='weight 1', range=range, normed=True)
        plt.hist(bayes_object.samples_w2, bins, alpha=alpha,
                 color='gray', label='weight 2', range=range, normed=True)
        plt.legend()
        return


