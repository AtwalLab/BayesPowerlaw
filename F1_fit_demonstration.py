from scipy.optimize import newton
from scipy.special import zeta
import scipy as sp
import numpy as np
from scipy.stats import uniform
import matplotlib.pyplot as plt
import powerlaw as pl

#generate data
exponents=([1.01, 2.0, 3.0, 4.0, 5.0])
xmax = 1000
sample_size = 1000

data1 = pl.power_law(exponents[0], xmax, sample_size)
data2 = pl.power_law(exponents[1], xmax, sample_size)
data3 = pl.power_law(exponents[2], xmax, sample_size)
data4 = pl.power_law(exponents[3], xmax, sample_size)
data5 = pl.power_law(exponents[4], xmax, sample_size)

test_bayes1 = pl.Fit_Bayes(data1)
test_bayes2 = pl.Fit_Bayes(data2)
test_bayes3 = pl.Fit_Bayes(data3)
test_bayes4 = pl.Fit_Bayes(data4)
test_bayes5 = pl.Fit_Bayes(data5)


fig = plt.figure()
ax = fig.add_subplot(111)

plt.axvline(x=exponents[0], color='black', linewidth=0.5)
pl.Fit_Bayes.plot_posterior(test_bayes1, 50, label='1.01', color='blue')
plt.axvline(x=exponents[1], color='black', linewidth=0.5)
pl.Fit_Bayes.plot_posterior(test_bayes2, 50, label='2.0', color='red')
plt.axvline(x=exponents[2], color='black', linewidth=0.5)
pl.Fit_Bayes.plot_posterior(test_bayes3, 50, label='3.0', color='green')
plt.axvline(x=exponents[3], color='black', linewidth=0.5)
pl.Fit_Bayes.plot_posterior(test_bayes4, 50, label='4.0', color='purple')
plt.axvline(x=exponents[4], color='black', linewidth=0.5)
pl.Fit_Bayes.plot_posterior(test_bayes5, 50, label='5.0', color='orange')


ax.set_xlim(0.0, 6.0)
plt.xlabel('exponent', fontsize=15)
plt.ylabel('posterior', fontsize=15)
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig('posterior_gamma1to5.png')
plt.savefig('posterior_gamma1to5.svg')


fig = plt.figure()
ax = fig.add_subplot(111)
pl.Fit_Bayes.plot_fit(test_bayes1, color='blue')
plt.xlabel('x', fontsize=15)
plt.ylabel('frequency', fontsize=15)
plt.tight_layout()
plt.savefig('gamma1_fit.png')
plt.savefig('gamma1_fit.svg')

fig = plt.figure()
ax = fig.add_subplot(111)
pl.Fit_Bayes.plot_fit(test_bayes2, color='red')
plt.xlabel('x', fontsize=15)
plt.ylabel('frequency', fontsize=15)
plt.tight_layout()
plt.savefig('gamma2_fit.png')
plt.savefig('gamma2_fit.svg')

fig = plt.figure()
ax = fig.add_subplot(111)
pl.Fit_Bayes.plot_fit(test_bayes3, color='green')
plt.xlabel('x', fontsize=15)
plt.ylabel('frequency', fontsize=15)
plt.tight_layout()
plt.savefig('gamma3_fit.png')
plt.savefig('gamma3_fit.svg')

fig = plt.figure()
ax = fig.add_subplot(111)
pl.Fit_Bayes.plot_fit(test_bayes4, color='orange')
plt.xlabel('x', fontsize=15)
plt.ylabel('frequency', fontsize=15)
plt.tight_layout()
plt.savefig('gamma4_fit.png')
plt.savefig('gamma4_fit.svg')

fig = plt.figure()
ax = fig.add_subplot(111)
pl.Fit_Bayes.plot_fit(test_bayes5, color='orange')
plt.xlabel('x', fontsize=15)
plt.ylabel('frequency', fontsize=15)
plt.tight_layout()
plt.savefig('gamma5_fit.png')
plt.savefig('gamma5_fit.svg')



