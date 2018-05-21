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
import pandas as pd
import powerlaw as pl


df = pd.read_csv(argv[1], names=['item'])
data = np.array(df.item.value_counts())

mixed = [1, 2, 3, 4]

exponents1 = np.zeros([mixed[0], 5])
exponents2 = np.zeros([mixed[1], 5])
exponents3 = np.zeros([mixed[2], 5])
exponents4 = np.zeros([mixed[3], 5])
exponents = ([exponents1, exponents2, exponents3, exponents4])
bic = np.zeros(len(mixed))
for m in mixed:
    bic_store = np.zeros(5)
    for it in range(5):
        obj = pl.Fit_Bayes(data, mixed=m, niters=100000)
        exponents[m - 1][:, it] = np.mean(obj.samples_gamma, axis=1)
        bic_store[it] = obj.bic
    bic[m - 1] = np.mean(bic_store)

exp1 = exponents1.flatten()
exp2 = exponents2.flatten()
exp3 = exponents3.flatten()
exp4 = exponents4.flatten()

plt.figure()
plt.scatter(np.array([1] * len(exp1)), exp1,
            marker='o', color='red', edgecolor='black')
plt.scatter(np.array([2] * len(exp2)), exp2,
            marker='o', color='red', edgecolor='black')
plt.scatter(np.array([3] * len(exp3)), exp3,
            marker='o', color='red', edgecolor='black')
plt.scatter(np.array([4] * len(exp4)), exp4,
            marker='o', color='red', edgecolor='black')
plt.xticks(np.arange(0, 5, step=1.0))
plt.tight_layout()

plt.savefig(argv[2]+'_fits.png')
plt.savefig(argv[2] + '_fits.svg')
np.savetxt(argv[2]+'_BIC.out', bic, delimiter=',')
