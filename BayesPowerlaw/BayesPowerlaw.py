from scipy.special import zeta
from scipy.optimize import newton
import scipy as sp
import numpy as np
from scipy.stats import uniform
import matplotlib.pyplot as plt
import warnings


class bayes(object):
    """This function fits the data to powerlaw distribution and outputs the exponent
    using Bayesian inference (markov chain monte carlo metropolis-hastings algorithm). 
    If the data consists of mixture more than 1 powerlaws, if specified the function will identify the mixture of exponents 
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
    
    sigma: ([float>0,float>0])
        Standard deviation of the sampling step size during MCMC for both gamma (first value) and weight (second value).
        Samples are taken from a gaussian distribution with a mean of zero.
        Default value is 0.05 for both, which is seen to perform with good efficiency.

    sigma_burn: ([float>0,float>0])
        Standard deviation of the sampling step size during MCMC burn in for both gamma (first value) and weight (second value).
        Samples are taken from a gaussian distribution with a mean of zero.
        Default value is 1.0 for both, which quickly travels to a correct value range.
    
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

    fit: (bool)
        Whether or not to perform fitting while creating the object.
        Not necessary if only desired to plot a power law with now fit.
        Default True.

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

    sigma_g: 
        Standard deviation of the sampling step size during MCMC for gamma.
        (float>0)

    sigma_w: 
        Standard deviation of the sampling step size during MCMC for weight.
        (float>0)

    sigma_burn_g: 
        Standard deviation of the sampling step size during MCMC burn in for gamma.
        (float>0)

    sigma_burn_w: 
        Standard deviation of the sampling step size during MCMC burn in for weight.
        (float>0)

    prior_gamma:
        An array of probabilities of each value in gammas for plotting prior.
        (1D np.array)

    prior_weight:
        An array of probabilities of each value in weights for plotting prior.
        (1D np.array)

    gamma_posterior:
        A 2D array of accepted exponents in each iteration after burn in.
        Rows - each powerlaw in the mixture.
        Columns - accepted exponent each iteration.
        (2D np.array)

    weight_posterior:
        A 2D array of accepted weights in each iteration after burn in.
        Rows - each powerlaw in the mixture.
        Columns - accepted weight each iteration.
        (2D np.array)
    """

    def __init__(self,
                 data,
                 gamma_range=[1.01, 7.99],
                 xmin=None,
                 xmax=None,
                 discrete=True,
                 niters=10000,
                 sigma=[0.05, 0.05],
                 sigma_burn=[1.0, 1.0],
                 burn_in=1000,
                 prior='jeffrey',
                 mixed=1,
                 fit=True):

        #convert data to numpy array in case input is a list.
        self.data = np.array(list(data))
        assert len(self.data)>0, "your data input is empty"
        assert type(self.data[0])!=np.str_, "data input must only contain positive integers or floats, not strings"
        assert np.sum(self.data<=0)==0, "data input values must be larger than 0"
        if len(np.unique(self.data))==len(self.data) and discrete==True:
            warnings.warn('ATTENTION, it appears you are fitting continuous data with discrete option. Consider using "discrete=False"', Warning)


        #xmin
        if xmin is None:
            self.xmin = min(self.data)
        else:
            self.xmin = xmin
            assert type(self.xmin)==int or type(self.xmin)==float,"xmin must be a number or a float"
            assert self.xmin>0, "xmin must be positive"

        #xmax
        if xmax is None:
            self.xmax = max(self.data) + 10.0
        else:
            self.xmax = xmax
            assert type(self.xmax)==int or type(self.xmax)==float,"xmax must be a number or a float"
            assert self.xmax>self.xmin, "xmax must be larger than xmin"

        #filter data given xmin and xmax
        if self.xmin > 1 or self.xmax != np.infty:
            self.data = self.data[(self.data >= self.xmin)
                                  & (self.data <= self.xmax)]

        self.n = len(self.data)  # length of data array
        # number of powerlaws in data arranged in the array from 1 to mixed
        assert type(mixed)==int and mixed>=1,"mixed parameter must be a positive integer"
        self.mixed = np.arange(mixed)
        # total number of parameters fitted (number of exponents + number of weights where the last weight is equal to 1-[weights])
        self.params = mixed * 2 - 1
        self.range = gamma_range  # exponent range
        assert len(list(self.range))==2, "gamma range input must contain two values"
        assert self.range[0]>1, "lower bound of exponent range must be larger than 1"
        if self.range[1]>=8:
            warnings.warn('ATTENTION, avoid using the upper bound of exponent range of 8 and more. Will cause Runtime Warning', Warning)
        self.discrete = discrete  # is data discrete or continuous
        assert type(self.discrete)==bool, "discrete must be boolean type. Choose false if data is continuous"
        self.prior_model = prior  # prior used (jeffrey's (default) or flat)
        assert self.prior_model=="jeffrey" or self.prior_model=="flat", "prior must be equal to either 'jeffrey' or 'flat'"
        # number of iterations in MCMC scaled given number of parameters
        self.niters = niters * self.params
        # standard deviation of gamma step size in MCMC
        self.sigma_g = sigma[0]
        # standard deviation of weight step size in MCMC
        self.sigma_w = sigma[1]
        # standard deviation of gamma step size in burn in
        self.sigma_burn_g = sigma_burn[0]
        # standard deviation of weight step size in burn in
        self.sigma_burn_w = sigma_burn[1]
        # number of iterations in MCMC burn in scaled given number of parameters
        self.burn = burn_in * self.params

        # make array of possible gammas, given the gamma_range, and an array of possible weights.
        self.gammas = np.linspace(1.01, self.range[1], 10000)
        self.weight = np.linspace(0.001, 1, 100)

        if self.prior_model == 'jeffrey':
            #make array of prior function given the jeffrey's prior.
            self.prior_gamma = [self.Z_jeffrey(i) for i in self.gammas]
        else:
            #make array of prior function given the flat prior.
            self.prior_gamma = (sp.stats.uniform(
                self.range[0], self.range[1] - self.range[0])).pdf(self.gammas)

        #make array of prior function for weights (flat prior).
        self.prior_weight = (sp.stats.uniform(0, 1)).pdf(self.weight)

        if fit:
            self.gamma_posterior, self.weight_posterior = self.posterior()

    def Z(self, gamma):
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

    def Z_prime(self, gamma):
        """
        This function calculates first differential of partition function Z.

        parameters
        ----------
        gamma: (float)
            Randomly sampled target exponent.

        returns
        ------
        First differential of the Z function.

        """

        h = 1e-8
        return (self.Z(gamma + h) - self.Z(gamma - h)) / (2 * h)

    def Z_prime2(self, gamma):
        """
        This function calculates second differential of partition function Z.

        parameters
        ----------
        gamma: (float)
            Randomly sampled target exponent.

        returns
        ------
        Second differential of the Z function.

        """

        h = 1e-4
        return (self.Z(gamma + h) - 2 * self.Z(gamma) + self.Z(gamma - h)) / (h**2)

    def Z_jeffrey(self, gamma):
        """
        This function calculates Jeffrey's prior for a given exponent.

        parameters
        ----------
        gamma: (float)
            Randomly sampled target exponent.

        returns
        ------
        Calculated Jeffrey's prior.

        """
        jeffrey=np.sqrt((self.Z_prime2(gamma)) / self.Z(gamma) - self.Z_prime(gamma)**2 / self.Z(gamma)**2)

        return jeffrey

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
            prior_answer = np.log(self.Z_jeffrey(gamma))
        else:
            #Flat prior: prior=1/(b-a)
            prior_answer = np.log(
                (sp.stats.uniform(self.range[0], self.range[1] - self.range[0])).pdf(gamma))
        return prior_answer

    def weight_prior(self, weight):
        """
        This function calculates prior given target weight and flat prior.

        parameters
        ----------
        gamma: (float)
            Randomly sampled target weight.

        returns
        -------
        prior_answer: 
            calculated log of prior.

        """
        prior_answer = np.log(
            (sp.stats.uniform(0, 1)).pdf(weight))
        return prior_answer

    def L(self, gamma_params, weight_params):
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
        Calculated log likelihood value.

        """

        lik = 0
        for i in range(len(self.mixed)):
            l = (weight_params[i] * self.data **
                 (-gamma_params[i])) / self.Z(gamma_params[i])
            lik = lik + l
        return np.sum(np.log(lik))

    def target(self, gamma_params, weight_params):
        """
        This function calculates target values for comparing existing exponents and weight to
        newly samples exponents and weights. Target for given parameters is equal to
        log of (likelihood x prior).

        parameters
        ----------

        gamma_params: (1D np.array)
            Array of randomly sampled target exponents for each powerlaw in the data.

        weight_params: (1D np.array)
            Array of randomly sampled target weights for each powerlaw in the data.

        returns
        -------

        p: 
            calculated target value.

        """
        p = 0
        if (np.sum(gamma_params < self.range[0]) != 0) or (np.sum(gamma_params > self.range[1]) != 0) or (np.sum(weight_params < 0) != 0):
            p = 0
        else:
            prior=0
            for i in range(len(gamma_params)):
                prior = prior + \
                    self.log_prior(gamma_params[i]) + \
                    self.weight_prior(weight_params[i])
            p = self.L(gamma_params, weight_params) + prior
        return p

    def sample_new(self, gamma_params, weight_params, sigma_g, sigma_w):
        """
        This function performs random sampling of gammas and weights given the initial
        gamma and weight values.

        parameters
        ----------

        gamma_params: (1D np.array)
            Array of randomly sampled target exponents for each powerlaw in the data.

        weight_params: (1D np.array)
            Array of randomly sampled target weights for each powerlaw in the data.

        sigma_g: (float>0)
            Standard deviation of the sampling step size during MCMC for gamma.

        sigma_w: (float>0)
            Standard deviation of the sampling step size during MCMC for weights.

        returns
        -------

        gamma_params_p: 
            a 1D array of newly sampled exponent values.
        
        weight_params_p:
            a 1D array of newly sampled weight values.

        """
        gamma_params_p = np.zeros(len(self.mixed))
        weight_params_p = np.zeros(len(self.mixed))
        for i in range(len(self.mixed)):
            gamma_params_p[i] = gamma_params[i] + \
                sp.stats.norm(0, sigma_g).rvs()
            if len(self.mixed) > 1:
                weight_params_p[i] = weight_params[i] + \
                    sp.stats.norm(0, sigma_w).rvs()
            else:
                weight_params_p[i] = weight_params[i]
        weight_params_p[-1] = 1 - np.sum(weight_params_p[0:-1])
        return gamma_params_p, weight_params_p

    def random_walk(self, gamma_params, weight_params, sigma_g, sigma_w):
        """
        This function samples new target exponent and weight values using sample_new function
        and calculates acceptance values to compare the initial parameter values to the 
        newly sampled ones.

        parameters
        ----------

        gamma_params: (1D np.array)
            Array of randomly sampled target exponents for each powerlaw in the data.

        weight_params: (1D np.array)
            Array of randomly sampled target weights for each powerlaw in the data.

        sigma_g: (float>0)
            Standard deviation of gamma step size during random sampling.

        sigma_w: (float>0)
            Standard deviation of weight step size during random sampling.

        returns
        -------

        a: 
            acceptance value to compare which parameter values are more likely to result in
            a more accurate fit.

        gamma_params_p:
            1D array of randomly sampled exponents.

        weight_params_p:
            1D array of randomly sampled weights.

        """
        gamma_params_p, weight_params_p = self.sample_new(
            gamma_params, weight_params, sigma_g, sigma_w)
        target_p = self.target(gamma_params_p, weight_params_p)
        target = self.target(gamma_params, weight_params)
        a = -10**8
        if target_p != 0:
            a = min(0, target_p - target)
        return a, gamma_params_p, weight_params_p

    def burn_in(self, gamma_params, weight_params):
        """
        This function preforms the burn in part of the MCMC algorithm that will get the 
        initial values to the roughly correct range for fitting. The parameters accepted
        during burn in are not included in the final samples array.

        parameters
        ----------

        gamma_params: (1D np.array)
            Array of randomly sampled target exponents for each powerlaw in the data.

        weight_params: (1D np.array)
            Array of randomly sampled target weights for each powerlaw in the data.


        returns
        -------

        gamma_params:
            1D array of accepted exponent values.

        weight_params:
            1D array of accepted weight values.

        """
        a, gamma_params_p, weight_params_p = self.random_walk(
            gamma_params, weight_params, self.sigma_burn_g, self.sigma_burn_w)
        if a == 0.0:
            gamma_params = gamma_params_p
            weight_params = weight_params_p
        return gamma_params, weight_params

    def monte_carlo(self, gamma_params, weight_params):
        """
        This function preforms the full MCMC algorithm. The parameters accepted in this part
        of MCMC will be saved in the final samples array.

        parameters
        ----------

        gamma_params: (1D np.array)
            Array of randomly sampled target exponents for each powerlaw in the data.

        weight_params: (1D np.array)
            Array of randomly sampled target weights for each powerlaw in the data.


        returns
        -------

        gamma_params:
            1D array of accepted exponent values.

        weight_params:
            1D array of accepted weight values.

        """
        a, gamma_params_p, weight_params_p = self.random_walk(
            gamma_params, weight_params, self.sigma_g, self.sigma_w)
        if a > np.log(np.random.uniform(0.0, 1.0)):
            gamma_params = gamma_params_p
            weight_params = weight_params_p
        return gamma_params, weight_params

    def posterior(self):
        """
        A master function that executes burn in and monte carlo algorithms while
        storing the accepted parameter values into the final samples array.

        parameters
        ----------

        None.

        returns
        -------

        samples_gamma:
            A 2D array of accepted exponents in each iteration after burn in.
            Rows - each powerlaw in the mixture.
            Columns - accepted exponent each iteration.

        samples_weight:
            A 2D array of accepted weights in each iteration after burn in.
            Rows - each powerlaw in the mixture.
            Columns - accepted weight each iteration.

        """
        gamma_params = np.array([self.range[0]] * (len(self.mixed)))
        weight_params = np.array([1 / len(self.mixed)] * (len(self.mixed)))
        #perform a burn in first without recording gamma values
        for i in range(1, self.burn + 1):
            gamma_params, weight_params = self.burn_in(
                gamma_params, weight_params)
        #now perform the rest of the sampling while recording gamma values
        samples_gamma = np.zeros([len(self.mixed), self.niters + 1])
        samples_gamma[:, 0] = gamma_params
        samples_weight = np.zeros([len(self.mixed), self.niters + 1])
        samples_weight[:, 0] = weight_params
        for i in range(1, self.niters + 1):
            gamma_params, weight_params = self.monte_carlo(
                gamma_params, weight_params)
            samples_gamma[:, i] = gamma_params
            samples_weight[:, i] = weight_params
        return samples_gamma, samples_weight

    def bic(self):
        """
        This function calculates the Bayesian Information Criteria for
        determining the number of parameters most optimal for fitting the dataset.

        parameters
        ----------

        samples_gamma: (2D np.array)
            accepted exponents in each iteration after burn in.
            Rows - each powerlaw in the mixture.
            Columns - accepted exponent each iteration.

        samples_weight: (2D np.array)
            accepted weights in each iteration after burn in.
            Rows - each powerlaw in the mixture.
            Columns - accepted weight each iteration.

        returns
        -------

        b:
            Bayesian Information Criteria (BIC) value.

        """
        gamma_params = sp.stats.mode(self.gamma_posterior, axis=1)[0]
        weight_params = sp.stats.mode(self.weight_posterior, axis=1)[0]
        b = np.log(self.n) * (len(self.mixed) * 2 - 1) - 2 * (self.L(gamma_params, weight_params))
        return b

    def powerlawpdf(self, final_gamma, xmin=None):
        """
        The power law probability function for generating the best fit curve.

        parameters
        ----------
        final_gamma: (float)
            Final exponent used to generate the best fit curve. For best results
            use the mean of posterior samples.

        returns
        -------
        xp: (1D array)
            array of X's arranged from xmin to xmax of the data.

        yp: (1D array)
            array of probabilities for each X given the final exponent.
        
        """
        if xmin == None:
            xmin = self.xmin
        if self.discrete:
            xp = np.arange(xmin, self.xmax)
        else:
            xp = np.linspace(xmin, self.xmax, 100)
        yp = (xp**(-final_gamma)) / self.Z(final_gamma)

        return xp, yp

    def plot_fit(self,
                 gamma_mean,
                 data_label=None,
                 data_color=None,
                 edge_color=None,
                 fit_color=None,
                 scatter_size=10,
                 line_width=1,
                 fit=True,
                 log=True,
                 xmin=None):
        """
        Function for plotting the date as a power law distribution on a log log scale
        along with the best fit.

        parameters
        ----------

        gamma_mean: (float)
            Final exponent used to generate the best fit curve. For best results
            use the mean of posterior samples.

        data_label: (str)
            curve label.

        data color: (str)
            color of the data scatter plot.

        edge_color: (str)
            color of the scatter marker edge.

        fit_color: (str)
            color of the best fit curve.
        
        scatter_size: (int or float)
            scatter marker size.

        line_width: (int or float)
            width of the best fit curve.

        fit: (bool)
            Whether to plot the best fit curve or not (default True).

        log: (bool)
            Whether to plot in the log scale or not (default True).

        returns
        -------

        None.

        """
        if self.discrete:
            unique, counts = np.unique(self.data, return_counts=True)
            frequency = counts / np.sum(counts)
        else:
            yx = plt.hist(self.data, bins=1000, normed=True)
            plt.clf()
            counts_pre = (yx[0])
            unique_pre = ((yx[1])[0:-1])
            unique = unique_pre[counts_pre != 0]
            frequency = counts_pre[counts_pre != 0]
            # frequency = counts / np.sum(counts)
        X, Y = self.powerlawpdf(gamma_mean, xmin)
        if log:
            plt.xscale('log')
            plt.yscale('log')
        plt.scatter(unique, frequency, s=scatter_size,
                    color=data_color, edgecolor=edge_color,label=data_label)
        if fit:
            plt.plot(X, Y, color=fit_color, linewidth=line_width)
        return

    def plot_prior(self, color=None, label=None):
        """
        Function for plotting prior for gammas.

        parameters
        ----------

        color: (str)
            color of the curve
        
        label: (str)
            label of the curve

        returns
        -------

        None.

        """
        plt.plot(self.gammas, self.prior_gamma, color=color, label=label)
        return

    def plot_posterior(self, samples, bins=100, alpha=None, color=None, label=None, range=None, normed=True):
        """
        Function for plotting posterior histogram.

        parameters
        ----------
        samples: (1D array)
            an array of accepted exponents during the MCMC algorithm.
        
        bins: (int)
            number of bins used for the histogram (default: 100).
        
        alpha: (0<=float<=1)
            transparency of the histogram.

        color: (str)
            color of the histogram.
        
        label: (str)
            label of the histogram.
        
        range: (tuple)
            range of histogram's X axis.
        
        normed: (bool)
            whether the histogram is normalized or not (default: True)

        returns
        -------

        None.

        """

        plt.hist(samples, bins, alpha=alpha, color=color,
                 label=label, range=range, normed=normed)
        return


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
                 initial_guess=[1, 6, 10],
                 xmin=None,
                 xmax=None,
                 discrete=True):

        self.data = np.array(data)
        assert len(self.data)>0, "your data input is empty"
        assert type(self.data[0])!=np.str_, "data input must only contain positive integers or floats, not strings"
        assert np.sum(self.data<=0)==0, "data input values must be larger than 0"
        if len(np.unique(self.data))==len(self.data) and discrete==True:
            warnings.warn('ATTENTION, it appears you are fitting continuous data with discrete option. Consider using "discrete=False"', Warning)


        #xmin
        if xmin is None:
            self.xmin = min(self.data)
        else:
            self.xmin = xmin
            assert type(self.xmax)==int or type(self.xmax)==float,"xmax must be a number or a float"
            assert self.xmax>self.xmin, "xmax must be larger than xmin"

        if xmax is None:
            self.xmax = max(self.data) + 10.0
        else:
            self.xmax = xmax
            assert type(self.xmax)==int or type(self.xmax)==float,"xmax must be a number or a float"
            assert self.xmax>self.xmin, "xmax must be larger than xmin"

        if self.xmin > 1 or self.xmax != np.infty:
            self.data = self.data[(self.data >= self.xmin)
                                  & (self.data <= self.xmax)]
        self.n = len(self.data)
        self.constant = np.sum(np.log(self.data)) / self.n
        self.discrete = discrete
        assert type(self.discrete)==bool, "discrete must be boolean type. Choose false if data is continuous"
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

    def F(self, gamma):
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
        Z_prime = (self.Z(gamma + h) - self.Z(gamma - h)) / (2 * h)
        return (Z_prime / self.Z(gamma)) + self.constant

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
        best_guess = np.zeros(len(self.initial_guess))
        for i in range(len(self.initial_guess)):
            try:
                best_guess[i] = newton(self.F, self.initial_guess[i])
            except RuntimeError:
                best_guess[i] = 0
        best_guess = best_guess[best_guess != 0]
        log_likelihood = np.zeros(len(best_guess))
        for i in range(len(best_guess)):
            log_likelihood[i] = -self.n * \
                np.log(self.Z(best_guess[i])) - \
                best_guess[i] * np.sum(np.log(self.data))
        self.log_likelihood = log_likelihood
        if len(log_likelihood) == 0:
            best_guess = 0
        else:
            best_guess = best_guess[np.where(
                log_likelihood == max(log_likelihood))]
        return best_guess



def power_law(exponents, weights, xmax, sample_size, xmin=1, discrete=True):
    """This function simulates a dataset that follows a powerlaw
    distribution with a given exponent and xmax.


    parameters
    ----------

    exponents: (array of floats>1) 
        array of exponents of the mixed powerlaw distribution equation. Array length is equal to the simulated mixture size.
        e.g. for single power law, array will contain only one value, mixture of 2 - two values, etc.

    weights: (array of 0<floats<=1) 
        array of weights of the mixed powerlaw distribution equation. Array length is equal to the simulated mixture size.
        e.g. for single power law, array will contain only one value that will be equal to 1, mixture of 2 - two values, etc.
        The sum of array must always be equal to 1.

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
    pmf=np.zeros(len(x))
    for i in range(len(exponents)):
        f = 1 / x**exponents[i]
        f /= f.sum()
        pmf = pmf + weights[i]*f
    #np.random.choice function generates a random sample of a given sample size
    #from a given 1-D array x, given the probabilities for each value.
    return np.random.choice(x, sample_size, p=pmf)


def demo():
    """
    Performs a demonstration of BayesPowerlaw.
    Parameters
    ----------
    
    None.

    Return
    ------

    None.
    """

    import os
    example_dir = os.path.dirname(__file__)
    example = 'examples/scripts/tweets.py'
    file_name = '%s/%s' % (example_dir, example)
    with open(file_name, 'r') as f:
        content = f.read()
        line = '-------------------------------------------------------------'
        print('Running %s:\n%s\n%s\n%s' %
            (file_name, line, content, line))
    exec(open(file_name).read())
