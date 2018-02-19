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
import powerlaw as pl


exponent = np.array([1.01, 1.1, 1.4, 1.7, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
xmax = 10000
sample_size = int(argv[1])


ML_mean = np.zeros((len(exponent), 50))
Bayes_mean = np.zeros((len(exponent), 50))
for i in range(len(exponent)):
    for j in range(50):
        data = pl.power_law(exponent[i], xmax, sample_size)
        ML = pl.Fit(data)
        Bayes = pl.Fit_Bayes(data)
        ML_mean[i, j] = np.mean(ML.best_guess)
        Bayes_mean[i, j] = np.mean(Bayes.samples)

ml_mean = np.mean(ML_mean, axis=1)
ml_std = np.std(ML_mean, axis=1)
bayes_mean = np.mean(Bayes_mean, axis=1)
bayes_std = np.std(Bayes_mean, axis=1)

plt.figure()
plt.scatter(exponent, ml_mean, color='red', label='ML')
plt.errorbar(exponent, ml_mean, yerr=ml_std, ls='none',
             color='red', elinewidth=1, capsize=4)
plt.scatter(exponent, bayes_mean, color='blue', label='Bayes')
plt.errorbar(exponent, bayes_mean, yerr=bayes_std, ls='none',
             color='blue', elinewidth=1, capsize=4)
plt.plot(exponent, exponent, color='black', label='Correct')
plt.legend(fontsize=15)
plt.ylabel('Fitted Exponent', fontsize=15)
plt.xlabel('Simulated Exponent', fontsize=15)
plt.tight_layout()

plt.savefig('MLvsBayes_fits_{}.png'.format(sample_size))
