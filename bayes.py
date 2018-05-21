from scipy.optimize import newton
from scipy.special import zeta
import scipy as sp
import numpy as np
from scipy.stats import uniform
import matplotlib.pyplot as plt


class fitBayes(object):
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

    def __init__(self, data, gamma_range=[1, 6], xmin=1, xmax=None, discrete=True, niters=10000, sigma=0.05, burn_in=1000, prior='flat', mixed=1):
        self.data = np.array(data)  # convert data to numpy array.
        self.xmin = xmin  # xmin
        if xmax is None:  # calculate xmax if unspecified
            self.xmax = max(self.data) + 10.0
        else:
            self.xmax = xmax  # xmax if specified
        if self.xmin > 1 or self.xmax != np.infty:  # filter data given xmin and xmax
            self.data = self.data[(self.data >= self.xmin)
                                  & (self.data <= self.xmax)]
        self.n = len(self.data)  # length of data array
        self.mixed = np.arange(mixed)  # number of powerlaws in data
        self.params = mixed * 2 - 1
        self.gamma_range = gamma_range  # exponent range
        self.discrete = discrete  # is data discrete or continuous?
        self.prior_model = prior  # prior used (flat (default) vs. jeffrey's)
        self.niters = niters  # number of iterations in MCMC
        self.sigma = sigma  # standard deviation of step size in MCMC
        self.sigma_burn = 1.0
        self.burn_in = burn_in * self.params

        self.gamma_params = np.array([1.01] * (len(self.mixed)))
        self.weight_params = np.array([1 / len(self.mixed)] * (len(self.mixed)))

        self.samples_gamma = np.zeros([len(self.mixed), self.niters + 1])
        self.samples_gamma[:, 0] = self.gamma_params
        self.samples_weight = np.zeros([len(self.mixed), self.niters + 1])
        self.samples_weight[:, 0] = self.weight_params

        self.kwargs = self.__dict__

        self.posterior()

    def posterior(self):
        #perform a burn in first without recording gamma values
        for i in range(1, self.burn_in + 1):
            self.gamma_params, self.weight_params = burnIn(self.gamma_params,self.weight_params,**self.kwargs).perform(**self.kwargs)
        #now perform the rest of the sampling while recording gamma values
        for i in range(1, self.niters + 1):
            self.gamma_params, self.weight_params = monteCarlo(**self.kwargs).perform(**self.kwargs)
            self.samples_gamma[:, i] = self.gamma_params
            self.samples_weight[:, i] = self.weight_params
        return self.samples_gamma, self.samples_weight


class sampleNew:
    def __init__(self,**kwargs):
        self.gamma_params = kwargs['gamma_params']
        self.weight_params = kwargs['weight_params']
        self.sigma= kwargs['sigma']

    def gamma(self):
        for i in range(len(self.gamma_params)):
            self.gamma_params[i]=self.gamma_params[i] + sp.stats.norm(0, self.sigma).rvs()
        return self.gamma_params

    def weight(self):
        for i in range(len(self.weight_params)):
            self.weight_params[i]=self.weight_params[i] + sp.stats.norm(0, self.sigma).rvs()
        return self.weight_params


class randomWalk:
    def __init__(self,**kwargs):
        self.gamma_params= kwargs['gamma_params']
        self.weight_params = kwargs['weight_params']
        self.sigma = kwargs['sigma']
        self.mixed = kwargs['mixed']

        self.gamma_params_p = np.zeros(len(self.mixed))
        self.weight_params_p = np.zeros(len(self.mixed))
        self.a = -10**8

    def perform(self,**kwargs):
        for i in range(len(self.mixed)):
                self.gamma_params_p[i] = sampleNew(**kwargs).gamma()
                if len(self.mixed) > 1:
                    self.weight_params_p[i] = sampleNew(**kwargs).weight()
                    self.weight_params_p[-1] = 1 - np.sum(self.weight_params_p[0:-1])
                else:
                    self.weight_params_p[i] = self.weight_params[i]

        target_p = Target(self.gamma_params_p,self.weight_params_p, **kwargs).calculate(**kwargs)
        target = Target(self.gamma_params,self.weight_params, **kwargs).calculate(**kwargs)

        if target_p!=0:
            self.a = min(0, target_p - target)

        return self.a, self.gamma_params_p, self.weight_params_p
    

class burnIn:
    def __init__(self, gamma_params, weight_params,**kwargs):
        self.gamma_params= gamma_params
        self.weight_params= weight_params
        self.sigma = kwargs['sigma']
        self.mixed = kwargs['mixed']


    def perform(self,**kwargs):
        a, gamma_params_p, weight_params_p=randomWalk(**kwargs).perform(**kwargs)
        if a == 0.0:
            self.gamma_params = gamma_params_p
            self.weight_params = weight_params_p
        return self.gamma_params, self.weight_params
        

class monteCarlo:
    def __init__(self,**kwargs):
        self.gamma_params=kwargs['gamma_params']
        self.weight_params = kwargs['weight_params']
        self.mixed=kwargs['mixed']

    def perform(self,**kwargs):
        a, gamma_params_p, weight_params_p = randomWalk(**kwargs).perform(**kwargs)
        if a > np.log(np.random.uniform(0.0, 1.0)):
            self.gamma_params = gamma_params_p
            self.weight_params = weight_params_p
        return self.gamma_params, self.weight_params


class Target:
    def __init__(self, gammas, weights, **kwargs):
        self.r=kwargs['gamma_range']

        self.gammas=gammas
        self.weights=weights
        self.p = 0

    def calculate(self,**kwargs):
        proceed = ((np.sum(self.gammas <= self.r[0]) == 0) + (np.sum(self.gammas > self.r[1]) == 0) + (np.sum(self.weights < 0) == 0) == 3)
        if proceed:
            self.p = Likelihood(self.gammas, self.weights, **kwargs).calculate(**kwargs) + log_prior(**kwargs).calculate(self.gammas[0])
        return self.p


class Likelihood:
    def __init__(self, gammas, weights, **kwargs):
        self.data=kwargs['data']
        self.mixed=kwargs['mixed']
        
        self.gammas=gammas
        self.weights=weights
        self.likelihood=0

    def calculate(self,**kwargs):
            """
            Pure powerlaw.
            This function calculates the log likelihood given a sampled gamma in MCMC algorithm.
            Input: gamma - a randomly sampled target exponent using MCMC algorithm.
            Output: lik - log likelihood value.
            """
            for i in range(len(self.mixed)):
                l = (self.weights[i] * self.data **
                    (-self.gammas[i])) / Z(**kwargs).calculate(self.gammas[i])
                self.likelihood = self.likelihood + l
            return self.likelihood

class Z:
    """
    The normalization function Z for discrete and continuous powerlaw distributions.
    Input: gamma - a randomly sampled target exponent using MCMC algorithm.
    Output: s - normalization value.
    """
    def __init__(self, **kwargs):
        self.discrete=kwargs['discrete']
        self.xmax = kwargs['xmax']
        self.xmin=kwargs['xmin']

    def calculate(self,gamma):
        if self.discrete == True:  # when powerlaw is discrete
            if np.isfinite(self.xmax):  # if xmax is NOT infinity:
                #Calculate zeta from Xmin to Infinity and substract Zeta from Xmax to Inifinity
                #To find zeta from Xmin to Xmax.
                s = zeta(gamma, self.xmin) - zeta(gamma, self.xmax)
            else:
                #if xmax is infinity, simply calculate zeta from Xmin till infinity.
                s = zeta(gamma, self.xmin)
        else:
            #calculate normalization function when powerlaw is continuous.
            #s=(xmax^(-gamma+1)/(1-gamma))-(xminx^(-gamma+1)/(1-gamma))
            s = (self.xmax**(-gamma + 1) / (1 - gamma)) - \
                (self.xmin**(-gamma + 1) / (1 - gamma))
        return s


class log_prior:
    """
    This function calculates prior given target exponent.
    Input: gamma - a randomly sampled target exponent using MCMC algorithm.
    Output: prior_answer - calculated log prior given the exponent gamma.
    """
    def __init__(self,**kwargs):
        self.range=kwargs['gamma_range']
        self.prior_model=kwargs['prior']

    def calculate(self, gamma):
        if self.prior_model == 'jeffrey':
            #Jeffrey's prior for continuous powerlaw distribution: prior=1/(gamma-1)
            prior_answer = np.log(1 / (gamma - 1))
        else:
            #Flat prior: prior=1/(b-a)
            prior_answer = np.log(
                (sp.stats.uniform(self.range[0], self.range[1] - self.range[0])).pdf(gamma))
        return prior_answer
