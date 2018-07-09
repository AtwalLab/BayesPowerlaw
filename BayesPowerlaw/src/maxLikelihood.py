from scipy.optimize import newton
from scipy.special import zeta
import numpy as np


class maxLikelihood(object):
    """This function fits the data to powerlaw distribution and outputs the exponent
    using the maximum likelihood and Newton-Raphson approach.

    parameters
    ----------
    data: (list or np.array of numbers)
        An array of data from the powerlaw that is being fitted here y=x^(-gamma)/Z.
        All values must be integers or floats from 1 to infinity.

    initial_guess: (list [lower range>=1, upper range, number of initial guesses])
        a list for generating a 1D array containing a variation of initial guesses for 
        Newton-Raphson algorithm.

    min: (int or float >=1)
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

    attributes
    ----------
    
    n: 
        Sample size of the data.
        (int>5)

    constant:
        a constant calculated to be added to 1st order Z differential prior to
        preforming Newton-Raphson algorithm.
        (float)

    initial_guess:
        an array of initial guesses used in Newton-Raphson algorithm.
        (1D array)

    """

    def __init__(self, 
                data, 
                initial_guess=[1,6,10], 
                xmin=None, 
                xmax=None, 
                discrete=True):

        self.data=np.array(data)

        #xmin
        if xmin is None:
            self.xmin = min(self.data)
        else:
            self.xmin = xmin

        if xmax is None:
            self.xmax=max(self.data)+10.0
        else:
            self.xmax=xmax

        if self.xmin>1 or self.xmax!=np.infty:
            self.data=self.data[(self.data>=self.xmin) & (self.data<=self.xmax)]
        self.n=len(self.data)
        self.constant=np.sum(np.log(self.data))/self.n
        self.discrete=discrete
        self.initial_guess = np.linspace(
            initial_guess[0], initial_guess[1], initial_guess[2])

    def Z(self, gamma):
        """
        Partition function Z for discrete and continuous powerlaw distributions.

        parameters
        ----------
        gamma: (float)
            exponent guess.

        returns
        ------
        s:
            Partition value.

        """
        
        if self.discrete == True:  # when powerlaw is discrete
            if np.isfinite(self.xmax):  # if xmax is NOT infinity:
                #Calculate zeta from Xmin to Infinity and substract Zeta from Xmax to Infinity
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

    def F(self,gamma):
        """The optimization function. 
        
        parameters
        ----------

        gamma: (float)
            exponent guess.

        returns
        -------

            First order Z differential plus constant attribute.

        """
        h = 1e-8
        Z_prime = (self.Z(gamma+h) - self.Z(gamma-h))/(2*h)
        return (Z_prime/self.Z(gamma))+self.constant

    def Guess(self):
        """
        Master function that performs Newton-Raphson algorithm and determins best
        exponent guess via maximum likelihood.

        parameters
        ----------

        None.

        returns
        -------

        best_guess: (1D array)
            an array containing best exponent guesses given the maximum likelihood.
            Length of array will be more than 1 if same likelihood value is associated
            with more than one exponent guess. 
        """
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
