from __future__ import division
from scipy.optimize import newton
from scipy.special import zeta
import numpy as np


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

	def __init__(self, data, initial_guess=None, xmin=1, xmax=np.infty, discrete=True):
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
			log_likelihood[i]=-self.n*np.log(self.Z(best_guess[i]))-best_guess[i]*np.sum(np.log(data))
		self.log_likelihood=log_likelihood
		return best_guess[np.where(log_likelihood==max(log_likelihood))]

	def Guess_continuous(self):
		return 1+self.n/(np.sum(np.log(self.data))-self.n*np.log(self.xmin))
