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

exponents = ([1.01,1.0,2.0,3.0,4.0,5.0])
xmax = 10000
sample_size = 100000

data1 = pl.power_law(exponents[0], xmax, sample_size)
data2 = pl.power_law(exponents[1], xmax, sample_size)
data3 = pl.power_law(exponents[2], xmax, sample_size)
data4 = pl.power_law(exponents[3], xmax, sample_size)
data5 = pl.power_law(exponents[4], xmax, sample_size)

fractions=([10, 25, 50, 75, 100, 250, 500, 750, 1000, 5000, 10000, 50000, 100000])

ML_fits1=np.zeros(len(fractions))
Bayes_fits1 = np.zeros(len(fractions))
for i in range(len(fractions)):
    data=np.random.choice(data1, fractions[i])
    ML = pl.Fit(data)
    Bayes = pl.Fit_Bayes(data)
    ML_fits1[i] = np.mean(ML.best_guess)
    Bayes_fits1[i] = np.mean(Bayes.samples)

ML_fits2 = np.zeros(len(fractions))
Bayes_fits2 = np.zeros(len(fractions))
for i in range(len(fractions)):
    data = np.random.choice(data2, fractions[i])
    ML = pl.Fit(data)
    Bayes = pl.Fit_Bayes(data)
    ML_fits2[i] = np.mean(ML.best_guess)
    Bayes_fits2[i] = np.mean(Bayes.samples)

ML_fits3 = np.zeros(len(fractions))
Bayes_fits3 = np.zeros(len(fractions))
for i in range(len(fractions)):
    data = np.random.choice(data3, fractions[i])
    ML = pl.Fit(data)
    Bayes = pl.Fit_Bayes(data)
    ML_fits3[i] = np.mean(ML.best_guess)
    Bayes_fits3[i] = np.mean(Bayes.samples)

ML_fits4 = np.zeros(len(fractions))
Bayes_fits4 = np.zeros(len(fractions))
for i in range(len(fractions)):
    data = np.random.choice(data4, fractions[i])
    ML = pl.Fit(data)
    Bayes = pl.Fit_Bayes(data)
    ML_fits4[i] = np.mean(ML.best_guess)
    Bayes_fits4[i] = np.mean(Bayes.samples)

ML_fits5 = np.zeros(len(fractions))
Bayes_fits5 = np.zeros(len(fractions))
for i in range(len(fractions)):
    data = np.random.choice(data5, fractions[i])
    ML = pl.Fit(data)
    Bayes = pl.Fit_Bayes(data)
    ML_fits5[i] = np.mean(ML.best_guess)
    Bayes_fits5[i] = np.mean(Bayes.samples)

plt.figure()
plt.plot(fractions, ML_fits1, '--s', color='blue', label='ML-1', linewidth=1.0)
plt.plot(fractions, Bayes_fits1, '-o', color='blue',
         label='Bayes-1', linewidth=1.0)
plt.axhline(y=exponents[0], color='black', linewidth=0.5)

plt.plot(fractions, ML_fits2, '--s', color='red', label='ML-2', linewidth=1.0)
plt.plot(fractions, Bayes_fits2, '-o', color='red',
         label='Bayes-2', linewidth=1.0)
plt.axhline(y=exponents[1], color='black', linewidth=0.5)

plt.plot(fractions, ML_fits3, '--s', color='green',
         label='ML-3', linewidth=1.0)
plt.plot(fractions, Bayes_fits3, '-o', color='green',
         label='Bayes-3', linewidth=1.0)
plt.axhline(y=exponents[2], color='black', linewidth=0.5)

plt.plot(fractions, ML_fits4, '--s', color='purple',
         label='ML-4', linewidth=1.0)
plt.plot(fractions, Bayes_fits4, '-o', color='purple',
         label='Bayes-4', linewidth=1.0)
plt.axhline(y=exponents[3], color='black', linewidth=0.5)

plt.plot(fractions, ML_fits5, '--s', color='orange',
         label='ML-5', linewidth=1.0)
plt.plot(fractions, Bayes_fits5, '-o', color='orange',
         label='Bayes-5', linewidth=1.0)
plt.axhline(y=exponents[4], color='black', linewidth=0.5)


plt.legend(fontsize=15)
plt.ylabel('Fitted Exponent', fontsize=15)
plt.xlabel('Sample fraction', fontsize=15)

plt.savefig('Subsample_fits.png')
plt.savefig('Subsample_fits.svg')

