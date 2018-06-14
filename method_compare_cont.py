import matplotlib
matplotlib.use('Agg')

from sys import *
import sys

from scipy.optimize import newton
from scipy.special import zeta
import scipy as sp
import numpy as np
from scipy.stats import uniform
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import powerlaw as pl


exponent = np.array([1.01, 1.1, 1.4, 1.7, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
xmax = 1000
sample_size = int(argv[1])


sim_number=50
ML_mean = np.zeros((len(exponent), sim_number))
Bayes_mean_jeffrey = np.zeros((len(exponent), sim_number))
Bayes_mean_flat = np.zeros((len(exponent), sim_number))
Linear_reg = np.zeros((len(exponent), sim_number))
for i in range(len(exponent)):
    for j in range(sim_number):
        #simulate data
        data = pl.power_law(exponent[i], xmax, sample_size, discrete=False)

        #Maximum likelihood fit.
        ML_fit = pl.Fit(data, discrete=False)
        ML_mean[i, j] = np.mean(ML_fit.Guess())

        #Bayes fits jeffrey and flat
        Bayes_jeffrey = pl.Fit_Bayes(data, discrete=False)
        gamma_j,weight_j=Bayes_jeffrey.posterior()
        Bayes_mean_jeffrey[i, j] = np.mean(gamma_j[0])

        Bayes_flat = pl.Fit_Bayes(data, discrete=False,prior='flat')
        gamma_f, weight_f = Bayes_flat.posterior()
        Bayes_mean_flat[i, j] = np.mean(gamma_f[0])

        #Linear regression
        yx = plt.hist(data, bins=1000)
        y = (yx[0])
        x = ((yx[1])[0:-1])
        x_f = x[y != 0]
        y_f = y[y != 0]

        if len(x_f)>10:
            X = np.log(x_f)[0:10]
            Y = np.log(y_f)[0:10]
        else:
            X = np.log(x_f)
            Y = np.log(y_f)
        Linear_reg[i, j] = -stats.linregress(X,Y)[0]


df=pd.DataFrame(index=range(len(exponent)))

df['exponent']=exponent
df['ml_mean'] = np.mean(ML_mean, axis=1)
df['ml_std'] = np.std(ML_mean, axis=1)
df['bayes_jeffrey_mean'] = np.mean(Bayes_mean_jeffrey, axis=1)
df['bayes_jeffrey_std'] = np.std(Bayes_mean_jeffrey, axis=1)
df['bayes_flat_mean'] = np.mean(Bayes_mean_flat, axis=1)
df['bayes_flat_std'] = np.std(Bayes_mean_flat, axis=1)
df['linear_mean']=np.mean(Linear_reg,axis=1)
df['linear_std'] = np.std(Linear_reg, axis=1)

df.to_csv('method_compare_'+sample_size+'_False.txt', sep='\t')

