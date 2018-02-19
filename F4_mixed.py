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

exponents = ([1.01, 2.0, 3.0, 4.0, 5.0])
sample_ratios=np.arange(1,11)

sample_size1 = int(argv[1])

xmax1 = 1000
xmax2 = 1000 * int(argv[2])

xmin_tofit=([1,2,3,4,5,6])

for ratio in sample_ratios:
    sample_size2 = sample_size1*ratio
    for exp1 in range(len(exponents)):
        for exp2 in range(len(exponents)):
            data1_x1 = pl.power_law(exponents[exp1], xmax1, sample_size1)
            data1_x2 = pl.power_law(exponents[exp2], xmax2, sample_size2)

            data2_x1 = pl.power_law(exponents[exp1], xmax2, sample_size1)
            data2_x2 = pl.power_law(exponents[exp2], xmax1, sample_size2)

            mixed1 = np.concatenate((data1_x1, data1_x2), axis=0)
            mixed2 = np.concatenate((data2_x1, data2_x2), axis=0)

            xmin_fits1=np.zeros(len(xmin_tofit))
            xmin_fits2 = np.zeros(len(xmin_tofit))
            xmin_std1 = np.zeros(len(xmin_tofit))
            xmin_std2 = np.zeros(len(xmin_tofit))
            for x in range(len(xmin_tofit)):
                Fit1 = pl.Fit_Bayes(mixed1, niters=10000, xmin=xmin_tofit[x])
                Fit2 = pl.Fit_Bayes(mixed2, niters=10000, xmin=xmin_tofit[x])
                xmin_fits1[x] = np.mean(Fit1.samples)
                xmin_fits2[x] = np.mean(Fit2.samples)
                xmin_std1[x] = np.std(Fit1.samples)
                xmin_std2[x] = np.std(Fit2.samples)

            plt.figure()
            plt.plot(np.array(range(1, 7)), xmin_fits1, 'red')
            plt.scatter(np.array(range(1, 7)), xmin_fits1, marker='o',
                        color='red', edgecolor='black', s=100, label='xmax1={} xmax2={}'.format(xmax1,xmax2))
            plt.errorbar(np.array(range(1, 7)), xmin_fits1, yerr=xmin_std1,
                        ls='none', color='red', elinewidth=1, capsize=4)

            plt.figure()
            plt.plot(np.array(range(1, 7)), xmin_fits2, 'black')
            plt.scatter(np.array(range(1, 7)), xmin_fits2, marker='o',
                        color='red', edgecolor='black', s=100, label='xmax1={} xmax2={}'.format(xmax2, xmax1))
            plt.errorbar(np.array(range(1, 7)), xmin_fits2, yerr=xmin_std2,
                         ls='none', color='red', elinewidth=1, capsize=4)


            plt.xlabel('Xmin', fontsize=15)
            plt.ylabel('Exponent', fontsize=15)
            plt.ylim(0, 6.5)
            plt.xlim(0.5, 6.5)
            plt.legend(fontsize=15)
            plt.tight_layout()

            plt.savefig('mixed_{}-{}_fits_{}-{}.png'.format(
                int(exponents[exp1]), int(exponents[exp2]), int(sample_size1), int(sample_size2)))
            plt.savefig('mixed_{}-{}_fits_{}-{}.png'.format(
                int(exponents[exp1]), int(exponents[exp2]), int(sample_size1), int(sample_size2)))

            



