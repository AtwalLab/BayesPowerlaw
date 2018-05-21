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

exponents = ([1.01, 2.0, 3.0, 4.0])
mixed = ([1, 2, 3, 4])
sample_size1 = int(argv[1])
sample_size2 = int(argv[2])
xmax = 1000


fig1 = plt.figure(figsize=(7, 10))
fig2 = plt.figure(figsize=(7, 10))
fig3 = plt.figure(figsize=(7, 10))
fig4 = plt.figure(figsize=(7, 10))
figures=[fig1,fig2,fig3,fig4]
BIC = np.zeros([4, 4, 4])
fig_num=0
for i in range(len(exponents)):
    for j in range(len(exponents)):
        exp1 = exponents[i]
        exp2 = exponents[j]
        data1_x1 = pl.power_law(exp1, xmax, sample_size1)
        data1_x2 = pl.power_law(exp2, xmax, sample_size2)
        data = np.concatenate((data1_x1, data1_x2), axis=0)
        fig_num=fig_num+1
        for m in range(len(mixed)):
            fits1 = np.array([])
            fits1_std = np.array([])
            bic1 = np.array([])
            for x in range(10):
                Fit1 = pl.Fit_Bayes(data, mixed=mixed[m], niters=100000)
                fits1 = np.append(fits1, np.mean(Fit1.samples_gamma, axis=1).flatten())
                fits1_std = np.append(fits1_std, np.std(Fit1.samples_gamma, axis=1).flatten())
                bic1 = np.append(bic1, Fit1.bic)
            
            ax = figures[m].add_subplot(4, 4, fig_num)
            ax.scatter(np.array([1] * len(fits1)), fits1, marker='o',
                        color='red', edgecolor='black')
            ax.errorbar(np.array([1] * len(fits1)), fits1, yerr=fits1_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)

            plt.xticks(np.arange(0, 2, step=1.0))
            plt.yticks(np.arange(0, 5, step=1.0))


        BIC[m,i,j]=np.mean(bic1)


matplotlib.rcParams.update({'font.size': 6})
plt.tight_layout()



fig1.savefig('mixture2_mix1.png')
fig1.savefig('mixture2_mix1.svg')
fig2.savefig('mixture2_mix2.png')
fig2.savefig('mixture2_mix2.svg')
fig3.savefig('mixture2_mix3.png')
fig3.savefig('mixture2_mix3.svg')
fig4.savefig('mixture2_mix4.png')
fig4.savefig('mixture2_mix4.svg')
np.savetxt('BIC_mixture2.out', BIC, delimiter=',')
