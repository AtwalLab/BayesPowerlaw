import matplotlib
matplotlib.use('Agg')

from sys import *
import sys

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
            self.xmax=max(self.data)+50
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

    def Z(self,gamma,xmin=1,xmax=np.infty):
        """The normalization function Z.
        Note that default arguments of xmin and xmax make Z equivalent to Riemann zeta function which is already
        implemented in Scipy as zeta(gamma,1)"""
        if np.isfinite(xmax):
            s=0
            for i in xrange(xmin,xmax+1):
                s+=(1/(i**gamma))
        else:
            s=zeta(gamma,xmin)
        return s

    def F(self,gamma):
        """The optimization function F(gamma). C is the second term in the definition of F(gamma) and is independent of
        gamma. C need to be defined before running the function. Function Z must be run beforehand as well."""
        h = 1e-8
        Z_prime = (self.Z(gamma+h,self.xmin,self.xmax) - self.Z(gamma-h,self.xmin,self.xmax))/(2*h)
        return (Z_prime/self.Z(gamma,self.xmin,self.xmax))+self.constant

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
    def __init__(self, data, gamma_range=[1,6], xmin=1, xmax=None, discrete=True, niters=5000, sigma=0.01, prior=['',0]):
        self.data=data
        self.xmin=xmin
        if xmax is None:
            self.xmax=max(self.data)+50
        else:
            self.xmax=xmax
        if self.xmin>1 or self.xmax!=np.infty:
            self.data=self.data[(self.data>=self.xmin) & (self.data<=self.xmax)]
        self.n=len(self.data)
        self.constant=np.sum(np.log(self.data))/self.n
        self.range=gamma_range
        self.discrete=discrete
        self.prior_model=prior[0]
        self.niters=niters
        self.sigma=sigma
        self.prior_exp=prior[1]
        self.gammas=np.linspace(1.01,self.range[1], 10000)
        if self.discrete==True:
            self.log_likelihood=np.array([-self.n*np.log(self.Z(j))-j*np.sum(np.log(self.data)) for j in self.gammas])
        else:
            self.log_likelihood=np.array([-j*np.sum(np.log(self.data))-self.n*np.log(self.xmin)+self.n*j*np.log(self.xmin)+self.n*np.log(j-1) for j in self.gammas])
        if self.prior_model=='powerlaw':
            self.prior=[(self.prior_exp-1)/(i**self.prior_exp) for i in self.gammas]
        if self.prior_model=='exponential':
            self.prior=[(1/self.prior_exp)*np.exp(-i/self.prior_exp) for i in self.gammas]
        else:    
            self.prior=(sp.stats.uniform(self.range[0],self.range[1]-self.range[0])).pdf(self.gammas)
        self.samples=self.posterior()
        self.best_guess=np.mean(self.samples)

    def log_prior(self,gamma):
        if self.prior_model=='powerlaw':
            prior_answer=np.log((self.prior_exp-1)/(gamma**self.prior_exp))
        if self.prior_model=='exponential':
            prior_answer=np.log((1/self.prior_exp)*np.exp(-gamma/self.prior_exp))
        else:
            prior_answer=np.log((sp.stats.uniform(self.range[0],self.range[1]-self.range[0])).pdf(gamma))
        return prior_answer

    def Z(self,gamma):
        """The normalization function Z."""  
        if np.isfinite(self.xmax):
            s=0
            for i in range(int(self.xmin),int(self.xmax)+1):
                s+=(1.0/(i**gamma))
        else:
            s=zeta(gamma,self.xmin)
        return s
    

    def L(self,gamma):
        if self.discrete==True:
            lik = (-self.n*np.log(self.Z(gamma))-gamma*np.sum(np.log(self.data)))
        else:
            lik = (-gamma * np.sum(np.log(self.data)) - self.n * np.log(self.xmin) + self.n * gamma * np.log(self.xmin) + self.n * np.log(gamma - 1))
        return lik


    def target (self, gamma):
        if gamma <= self.range[0] or gamma > self.range[1]:
            p = 0
        else:
            p = self.L(gamma)+self.log_prior(gamma)
        return p
    

    def monte_carlo(self,gamma,sigma,accept,a_array,burn_in=False):
        gamma_p = gamma + sp.stats.norm(0, sigma).rvs()
        if self.target(gamma_p)==0:
            a=-1
        else: 
            a = min(0,self.target(gamma_p)-self.target(gamma))
            a_array=np.append(a_array,a)
        if burn_in==True:
            if a==0.0:
                gamma=gamma_p
        else:
            
            if a>np.log(np.random.uniform(0.0, 1.0)):
                gamma=gamma_p
                accept=accept+1
        return gamma, accept, a_array


    def posterior (self):
        a_array=np.array([])
        sigma_burn = 1.0
        gamma=1.01
        accept=0
        burn_in=1000
        acceptance=np.array([])
        sigma_record=np.array([])
        #perform a burn in first without recording gamma values
        for i in range(1,burn_in+1):
            gamma, accept, a_array=self.monte_carlo(gamma, sigma_burn, accept,a_array, burn_in=True)
        #now perform the rest of the sampling while recording gamma values
        samples = np.zeros(self.niters+1)
        samples[0]=gamma
        for i in range(1,self.niters+1):
            gamma, accept, a_array=self.monte_carlo(gamma, self.sigma, accept, a_array)
            samples[i] = gamma
        self.a=a_array
        return samples

   
    @staticmethod
    def plot_fit(bayes_object, label=None, color=None):
        data = bayes_object.data
        exponent = bayes_object.best_guess
        xmin = bayes_object.xmin
        xmax = bayes_object.xmin
        discrete=bayes_object.discrete
        
        def Z(exponent,xmin,xmax):
            """The normalization function Z.
            Note that default arguments of xmin and xmax make Z equivalent to Riemann zeta function which is already
            implemented in Scipy as zeta(gamma,1)"""
            if np.isfinite(xmax):
                s=0
                for i in range(xmin,xmax):
                    s+=(1/(i**exponent))
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
                powerlaw=((data/xmin)**(-exponent))*((exponent-1)/xmin)
            return powerlaw
        
        if discrete==True:
            unique, counts = np.unique(data, return_counts=True)
            frequency=counts/np.sum(counts)
            xp=np.arange(np.min(data),np.max(data)+1)
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
    def plot_likelihood(bayes_object, color=None, label=None):
        plt.plot(bayes_object.gammas,bayes_object.log_likelihood, color=color, label=label)
        plt.legend()
        return
    
    @staticmethod
    def plot_posterior(bayes_object, bins=100, alpha=0.5, color=None, label=None, range=None):
        plt.hist(bayes_object.samples, bins, alpha=alpha, color=color, label=label, range=range, normed=True)
        plt.legend()
        return
    


# exponent=3.0
# xmax=100
# sample_size=1000

# data=power_law(exponent, xmax, sample_size, discrete=False)

# test=Fit_Bayes(data)

# print (test.best_guess)

exponent=np.array([1.01, 1.1, 1.3, 1.5, 1.7, 1.9, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
xmax = 10000
sample_size = int(argv[1])


ML_mean = np.zeros((len(exponent), 50))
Bayes_mean = np.zeros((len(exponent), 50))
for i in range(len(exponent)):
    for j in range(50):
        data = power_law(exponent[i], xmax, sample_size)
        ML = Fit(data)
        Bayes = Fit_Bayes(data)
        ML_mean[i, j] = np.mean(ML.best_guess)
        Bayes_mean[i, j] = np.mean(Bayes.samples)

ml_mean=np.mean(ML_mean, axis=1)
ml_std=np.std(ML_mean,axis=1)
bayes_mean=np.mean(Bayes_mean, axis=1)
bayes_std=np.std(Bayes_mean, axis=1)

plt.figure(figsize=(20, 18))
plt.scatter(exponent, ml_mean, color='red', label='ML')
plt.errorbar(exponent, ml_mean, yerr=ml_std, ls='none', color='red', elinewidth=1, capsize=4)
plt.scatter(exponent, bayes_mean, color='blue', label='Bayes')
plt.errorbar(exponent, bayes_mean, yerr=bayes_std, ls='none', color='blue', elinewidth=1, capsize=4)
plt.plot(exponent, exponent, color='black', label='Correct')
plt.legend(fontsize=15)
plt.ylabel('Fitted Exponent', fontsize=15)
plt.xlabel('Real Exponent', fontsize=15)

plt.savefig('test2_fits_{}.png'.format(sample_size))


# simulation_exponents=np.array([1.01, 1.1, 1.3, 1.5, 1.7, 1.9, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
# sample_size=int(argv[1])
# xmin=1
# xmax=1000

# Bayes_fits=np.zeros(len(simulation_exponents))
# Bayes_fits_std=np.zeros(len(simulation_exponents))

# for k in range(5):
#     for i in range(len(simulation_exponents)):
#         bayes_tests=np.zeros(20)
#         for j in range(20):
#             data=power_law(simulation_exponents[i], xmax, sample_size)
#             bayes = Fit_Bayes(data, prior=['',0.0])
#             bayes_tests[j]=np.mean(bayes.samples)
#         Bayes_fits[i]=np.mean(bayes_tests)
#         Bayes_fits_std[i]=np.std(bayes_tests)
#     np.savetxt('Fits_'+str(int(argv[1]))+'_'+str(k)+'.txt', (Bayes_fits), delimiter=',')
#     np.savetxt('Fits_std'+str(int(argv[1]))+'_'+str(k)+'.txt', (Bayes_fits_std), delimiter=',')