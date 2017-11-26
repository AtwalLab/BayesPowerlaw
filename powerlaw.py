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

	def __init__(self, data, initial_guess=np.linspace(1,5,10), xmin=1, xmax=np.infty, discrete=True):
		self.data=data
		self.n=len(data)
		self.constant=np.sum(np.log(data))/self.n
		self.xmin=xmin
		self.xmax=xmax
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
	def __init__(self, data, gamma_range=[1,5], xmin=1, xmax=np.infty, discrete=True, niters=5000):
		self.data=data
		self.n=len(data)
		self.constant=np.sum(np.log(data))/self.n
		self.range=gamma_range
		self.xmin=xmin
		self.xmax=xmax
		self.discrete=discrete
		self.niters=niters
		self.gammas=np.linspace(1.01,gamma_range[1]*2, 10000)
		if self.discrete==True:
			self.log_likelihood=np.array([-self.n*np.log(self.Z(j))-j*np.sum(np.log(self.data)) for j in self.gammas])
			self.likelihood = np.exp(self.log_likelihood - max(self.log_likelihood))
		self.prior=(sp.stats.uniform(self.range[0],self.range[1]-self.range[0])).pdf(self.gammas)
		self.samples=self.posterior()

	def pri(self,gamma):
		return (sp.stats.uniform(self.range[0],self.range[1]-self.range[0])).pdf(gamma)

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

	def l(self,gamma):
		if self.discrete==True:
			lik = np.exp((-self.n*np.log(self.Z(gamma))-gamma*np.sum(np.log(self.data)))-max(self.log_likelihood))
		else:
			lik= np.prod((self.data**-gamma)*(1-gamma))
		return lik


	def target (self, gamma):
		if gamma <= 1 or gamma > 5:
			p = 0
		else:
			p = self.l(gamma)*self.pri(gamma)
		return p

	
	def posterior (self):
		sigma = 1.0
		naccept=0
		gamma=1.01
		burn_in=100
		#perform a burn in first without recording gamma values
		for i in range(burn_in+1):
			gamma_p=gamma+sp.stats.norm(0,sigma).rvs()
			a=(self.target(gamma_p)/self.target(gamma))*(gamma_p/gamma)
			if a>=1:
				gamma = gamma_p	
		#now perform the rest of the sampling while recording gamma values
		samples = np.zeros(self.niters+1)
		samples[0]=gamma
		sigma=0.8
		for i in range(1,self.niters+1):
			gamma_p=gamma+sp.stats.norm(0,sigma).rvs()
			a = (self.target(gamma_p) / self.target(gamma)) * (gamma_p / gamma)
			if a >= 1:
				gamma = gamma_p
			samples[i]=gamma
			
			if i%200==0:
				mean=np.mean(samples[i - 200:i])
				std=np.std(samples[i - 200:i])
				if np.mean(samples[i - 100:i]) < (mean + std) and np.mean(samples[i - 100:i]) > (mean - std):
					sigma+=0.2

			self.sigma=sigma

		return samples


exponent = np.linspace(1.02, 4.9, 25)

xmax = int(argv[1])
sample_size = int(argv[2])
iterations=np.array([1000,5000])


for n in range(len(iterations)):
	ML_mean = np.zeros((len(exponent), 50))
	Bayes_mean = np.zeros((len(exponent), 50))
	for i in range(len(exponent)):
		for j in range(50):
			data = power_law(exponent[i], xmax, sample_size)
			ML = Fit(data)
			Bayes = Fit_Bayes(data, niters=iterations[n])
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

	plt.savefig('xmax{}_N{}_its{}.png'.format(xmax, sample_size, iterations[n]))
