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

exponents = ([1.01,2.0,3.0,4.0,5.0])
xmax = 10000
sample_size = 100000

data1 = pl.power_law(exponents[0], xmax, sample_size)
data2 = pl.power_law(exponents[1], xmax, sample_size)
data3 = pl.power_law(exponents[2], xmax, sample_size)
data4 = pl.power_law(exponents[3], xmax, sample_size)
data5 = pl.power_law(exponents[4], xmax, sample_size)

fractions=([10, 25, 50, 75, 100, 250, 500, 750, 1000, 5000, 10000, 50000, 100000])


Bayes_fits1 = np.zeros(len(fractions))
Bayes_std1 = np.zeros(len(fractions))
for i in range(len(fractions)):
    data=np.random.choice(data1, fractions[i])
    Bayes = pl.Fit_Bayes(data)
    Bayes_fits1[i] = np.mean(Bayes.samples)
    Bayes_std1[i] = np.std(Bayes.samples)

Bayes_fits2 = np.zeros(len(fractions))
Bayes_std2 = np.zeros(len(fractions))
for i in range(len(fractions)):
    data = np.random.choice(data2, fractions[i])
    Bayes = pl.Fit_Bayes(data)
    Bayes_fits2[i] = np.mean(Bayes.samples)
    Bayes_std2[i] = np.std(Bayes.samples)

Bayes_fits3 = np.zeros(len(fractions))
Bayes_std3 = np.zeros(len(fractions))
for i in range(len(fractions)):
    data = np.random.choice(data3, fractions[i])
    Bayes = pl.Fit_Bayes(data)
    Bayes_fits3[i] = np.mean(Bayes.samples)
    Bayes_std3[i] = np.std(Bayes.samples)

Bayes_fits4 = np.zeros(len(fractions))
Bayes_std4 = np.zeros(len(fractions))
for i in range(len(fractions)):
    data = np.random.choice(data4, fractions[i])
    Bayes = pl.Fit_Bayes(data)
    Bayes_fits4[i] = np.mean(Bayes.samples)
    Bayes_std4[i] = np.std(Bayes.samples)

Bayes_fits5 = np.zeros(len(fractions))
Bayes_std5 = np.zeros(len(fractions))
for i in range(len(fractions)):
    data = np.random.choice(data5, fractions[i])
    Bayes = pl.Fit_Bayes(data)
    Bayes_fits5[i] = np.mean(Bayes.samples)
    Bayes_std5[i] = np.std(Bayes.samples)


plt.figure()
plt.plot(fractions, Bayes_fits1, '-o', color='blue',
         label='1', linewidth=1.0)
plt.errorbar(fractions, Bayes_fits1, yerr=Bayes_std1,
             ls='none', color='black', elinewidth=1, capsize=4)
plt.axhline(y=exponents[0], color='black', linewidth=0.5)

plt.plot(fractions, Bayes_fits2, '-o', color='red',
         label='2', linewidth=1.0)
plt.errorbar(fractions, Bayes_fits2, yerr=Bayes_std2,
             ls='none', color='black', elinewidth=1, capsize=4)
plt.axhline(y=exponents[1], color='black', linewidth=0.5)

plt.plot(fractions, Bayes_fits3, '-o', color='green',
         label='3', linewidth=1.0)
plt.errorbar(fractions, Bayes_fits3, yerr=Bayes_std3,
             ls='none', color='black', elinewidth=1, capsize=4)
plt.axhline(y=exponents[2], color='black', linewidth=0.5)


plt.plot(fractions, Bayes_fits4, '-o', color='purple',
         label='4', linewidth=1.0)
plt.errorbar(fractions, Bayes_fits4, yerr=Bayes_std4,
             ls='none', color='black', elinewidth=1, capsize=4)
plt.axhline(y=exponents[3], color='black', linewidth=0.5)


plt.plot(fractions, Bayes_fits5, '-o', color='orange',
         label='5', linewidth=1.0)
plt.errorbar(fractions, Bayes_fits5, yerr=Bayes_std5,
             ls='none', color='black', elinewidth=1, capsize=4)
plt.axhline(y=exponents[4], color='black', linewidth=0.5)


plt.legend(fontsize=15)
plt.ylabel('Fitted Exponent', fontsize=15)
plt.xlabel('Sample fraction', fontsize=15)
plt.tight_layout()

plt.savefig('Subsample_fits.png')
plt.savefig('Subsample_fits.svg')

