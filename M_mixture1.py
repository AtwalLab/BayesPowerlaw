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
xmax = 1000


f, ([ax1, ax2, ax3, ax4]) = plt.subplots(
    4, 1, sharex=True, sharey=True, figsize=(5, 15))


exp1 = exponents[0]
data1_x1 = pl.power_law(exp1, xmax, sample_size1)

exp2 = exponents[1]
data2_x1 = pl.power_law(exp2, xmax, sample_size1)

exp3 = exponents[2]
data3_x1 = pl.power_law(exp3, xmax, sample_size1)

exp4 = exponents[3]
data4_x1 = pl.power_law(exp4, xmax, sample_size1)

BIC_table=pd.DataFrame(index=['mixed=1','mixed=2','mixed=3','mixed=4'],columns=['1','2','3','4'])

#mixed=1
#ax1
fits1 = np.array([])
fits1_std = np.array([])

fits2 = np.array([])
fits2_std = np.array([])

fits3 = np.array([])
fits3_std = np.array([])

fits4 = np.array([])
fits4_std = np.array([])

bic1 = np.array([])
bic2 = np.array([])
bic3 = np.array([])
bic4 = np.array([])

for x in range(10):
    Fit1 = pl.Fit_Bayes(data1_x1, mixed=1)
    Fit2 = pl.Fit_Bayes(data2_x1, mixed=1)
    Fit3 = pl.Fit_Bayes(data3_x1, mixed=1)
    Fit4 = pl.Fit_Bayes(data4_x1, mixed=1)

    fits1 = np.append(fits1,np.mean(Fit1.samples_gamma,axis=1).flatten())
    fits1_std = np.append(fits1_std,np.std(Fit1.samples_gamma, axis=1).flatten())

    fits2 = np.append(fits2, np.mean(Fit1.samples_gamma, axis=1).flatten())
    fits2_std = np.append(fits2_std, np.std(
        Fit1.samples_gamma, axis=1).flatten())

    fits3 = np.append(fits3, np.mean(Fit1.samples_gamma, axis=1).flatten())
    fits3_std = np.append(fits3_std, np.std(
        Fit1.samples_gamma, axis=1).flatten())

    fits4 = np.append(fits4, np.mean(Fit1.samples_gamma, axis=1).flatten())
    fits4_std = np.append(fits4_std, np.std(
        Fit1.samples_gamma, axis=1).flatten())

    bic1 = np.append(bic1, Fit1.bic)
    bic2 = np.append(bic2, Fit2.bic)
    bic3 = np.append(bic3, Fit3.bic)
    bic4 = np.append(bic4, Fit4.bic)

BIC_table.iloc[0, :] = np.array(
    [np.mean(bic1), np.mean(bic2), np.mean(bic3), np.mean(bic4)])

ax1.scatter(np.array([exp1]*len(fits1)), fits1, marker='o',
            color='red', edgecolor='black')
ax1.errorbar(np.array([exp1] * len(fits1)), fits1, yerr=fits1_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)

ax1.scatter(np.array([exp2] * len(fits2)), fits2, marker='o',
            color='red', edgecolor='black')
ax1.errorbar(np.array([exp2] * len(fits2)), fits2, yerr=fits2_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)

ax1.scatter(np.array([exp3] * len(fits3)), fits3, marker='o',
            color='red', edgecolor='black')
ax1.errorbar(np.array([exp3] * len(fits3)), fits3, yerr=fits3_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)

ax1.scatter(np.array([exp4] * len(fits4)), fits4, marker='o',
            color='red', edgecolor='black')
ax1.errorbar(np.array([exp4] * len(fits4)), fits4, yerr=fits4_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)



#mixed=2
#ax2
fits1 = np.array([])
fits1_std = np.array([])

fits2 = np.array([])
fits2_std = np.array([])

fits3 = np.array([])
fits3_std = np.array([])

fits4 = np.array([])
fits4_std = np.array([])

bic1 = np.array([])
bic2 = np.array([])
bic3 = np.array([])
bic4 = np.array([])

for x in range(10):
    Fit1 = pl.Fit_Bayes(data1_x1, mixed=2)
    Fit2 = pl.Fit_Bayes(data2_x1, mixed=2)
    Fit3 = pl.Fit_Bayes(data3_x1, mixed=2)
    Fit4 = pl.Fit_Bayes(data4_x1, mixed=2)

    fits1 = np.append(fits1, np.mean(Fit1.samples_gamma, axis=1).flatten())
    fits1_std = np.append(fits1_std, np.std(
        Fit1.samples_gamma, axis=1).flatten())

    fits2 = np.append(fits2, np.mean(Fit1.samples_gamma, axis=1).flatten())
    fits2_std = np.append(fits2_std, np.std(
        Fit1.samples_gamma, axis=1).flatten())

    fits3 = np.append(fits3, np.mean(Fit1.samples_gamma, axis=1).flatten())
    fits3_std = np.append(fits3_std, np.std(
        Fit1.samples_gamma, axis=1).flatten())

    fits4 = np.append(fits4, np.mean(Fit1.samples_gamma, axis=1).flatten())
    fits4_std = np.append(fits4_std, np.std(
        Fit1.samples_gamma, axis=1).flatten())

    bic1 = np.append(bic1, Fit1.bic)
    bic2 = np.append(bic2, Fit2.bic)
    bic3 = np.append(bic3, Fit3.bic)
    bic4 = np.append(bic4, Fit4.bic)

BIC_table.iloc[1, :] = np.array(
    [np.mean(bic1), np.mean(bic2), np.mean(bic3), np.mean(bic4)])

ax2.scatter(np.array([exp1] * len(fits1)), fits1, marker='o',
            color='red', edgecolor='black')
ax2.errorbar(np.array([exp1] * len(fits1)), fits1, yerr=fits1_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)

ax2.scatter(np.array([exp2] * len(fits2)), fits2, marker='o',
            color='red', edgecolor='black')
ax2.errorbar(np.array([exp2] * len(fits2)), fits2, yerr=fits2_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)

ax2.scatter(np.array([exp3] * len(fits3)), fits3, marker='o',
            color='red', edgecolor='black')
ax2.errorbar(np.array([exp3] * len(fits3)), fits3, yerr=fits3_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)

ax2.scatter(np.array([exp4] * len(fits4)), fits4, marker='o',
            color='red', edgecolor='black')
ax2.errorbar(np.array([exp4] * len(fits4)), fits4, yerr=fits4_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)



#mixed=3
#ax3
fits1 = np.array([])
fits1_std = np.array([])

fits2 = np.array([])
fits2_std = np.array([])

fits3 = np.array([])
fits3_std = np.array([])

fits4 = np.array([])
fits4_std = np.array([])

bic1 = np.array([])
bic2 = np.array([])
bic3 = np.array([])
bic4 = np.array([])

for x in range(10):
    Fit1 = pl.Fit_Bayes(data1_x1, mixed=3)
    Fit2 = pl.Fit_Bayes(data2_x1, mixed=3)
    Fit3 = pl.Fit_Bayes(data3_x1, mixed=3)
    Fit4 = pl.Fit_Bayes(data4_x1, mixed=3)

    fits1 = np.append(fits1, np.mean(Fit1.samples_gamma, axis=1).flatten())
    fits1_std = np.append(fits1_std, np.std(
        Fit1.samples_gamma, axis=1).flatten())

    fits2 = np.append(fits2, np.mean(Fit1.samples_gamma, axis=1).flatten())
    fits2_std = np.append(fits2_std, np.std(
        Fit1.samples_gamma, axis=1).flatten())

    fits3 = np.append(fits3, np.mean(Fit1.samples_gamma, axis=1).flatten())
    fits3_std = np.append(fits3_std, np.std(
        Fit1.samples_gamma, axis=1).flatten())

    fits4 = np.append(fits4, np.mean(Fit1.samples_gamma, axis=1).flatten())
    fits4_std = np.append(fits4_std, np.std(
        Fit1.samples_gamma, axis=1).flatten())

    bic1 = np.append(bic1, Fit1.bic)
    bic2 = np.append(bic2, Fit2.bic)
    bic3 = np.append(bic3, Fit3.bic)
    bic4 = np.append(bic4, Fit4.bic)

BIC_table.iloc[2, :] = np.array(
    [np.mean(bic1), np.mean(bic2), np.mean(bic3), np.mean(bic4)])

ax3.scatter(np.array([exp1] * len(fits1)), fits1, marker='o',
            color='red', edgecolor='black')
ax3.errorbar(np.array([exp1] * len(fits1)), fits1, yerr=fits1_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)

ax3.scatter(np.array([exp2] * len(fits2)), fits2, marker='o',
            color='red', edgecolor='black')
ax3.errorbar(np.array([exp2] * len(fits2)), fits2, yerr=fits2_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)

ax3.scatter(np.array([exp3] * len(fits3)), fits3, marker='o',
            color='red', edgecolor='black')
ax3.errorbar(np.array([exp3] * len(fits3)), fits3, yerr=fits3_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)

ax3.scatter(np.array([exp4] * len(fits4)), fits4, marker='o',
            color='red', edgecolor='black')
ax3.errorbar(np.array([exp4] * len(fits4)), fits4, yerr=fits4_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)


#mixed=4
#ax4

fits1 = np.array([])
fits1_std = np.array([])

fits2 = np.array([])
fits2_std = np.array([])

fits3 = np.array([])
fits3_std = np.array([])

fits4 = np.array([])
fits4_std = np.array([])

bic1 = np.array([])
bic2 = np.array([])
bic3 = np.array([])
bic4 = np.array([])


for x in range(10):
    Fit1 = pl.Fit_Bayes(data1_x1, mixed=4)
    Fit2 = pl.Fit_Bayes(data2_x1, mixed=4)
    Fit3 = pl.Fit_Bayes(data3_x1, mixed=4)
    Fit4 = pl.Fit_Bayes(data4_x1, mixed=4)

    fits1 = np.append(fits1, np.mean(Fit1.samples_gamma, axis=1).flatten())
    fits1_std = np.append(fits1_std, np.std(
        Fit1.samples_gamma, axis=1).flatten())

    fits2 = np.append(fits2, np.mean(Fit1.samples_gamma, axis=1).flatten())
    fits2_std = np.append(fits2_std, np.std(
        Fit1.samples_gamma, axis=1).flatten())

    fits3 = np.append(fits3, np.mean(Fit1.samples_gamma, axis=1).flatten())
    fits3_std = np.append(fits3_std, np.std(
        Fit1.samples_gamma, axis=1).flatten())

    fits4 = np.append(fits4, np.mean(Fit1.samples_gamma, axis=1).flatten())
    fits4_std = np.append(fits4_std, np.std(
        Fit1.samples_gamma, axis=1).flatten())

    bic1 = np.append(bic1, Fit1.bic)
    bic2 = np.append(bic2, Fit2.bic)
    bic3 = np.append(bic3, Fit3.bic)
    bic4 = np.append(bic4, Fit4.bic)

BIC_table.iloc[3, :] = np.array(
    [np.mean(bic1), np.mean(bic2), np.mean(bic3), np.mean(bic4)])

ax4.scatter(np.array([exp1] * len(fits1)), fits1, marker='o',
            color='red', edgecolor='black')
ax4.errorbar(np.array([exp1] * len(fits1)), fits1, yerr=fits1_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)

ax4.scatter(np.array([exp2] * len(fits2)), fits2, marker='o',
            color='red', edgecolor='black')
ax4.errorbar(np.array([exp2] * len(fits2)), fits2, yerr=fits2_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)

ax4.scatter(np.array([exp3] * len(fits3)), fits3, marker='o',
            color='red', edgecolor='black')
ax4.errorbar(np.array([exp3] * len(fits3)), fits3, yerr=fits3_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)

ax4.scatter(np.array([exp4] * len(fits4)), fits4, marker='o',
            color='red', edgecolor='black')
ax4.errorbar(np.array([exp4] * len(fits4)), fits4, yerr=fits4_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)


matplotlib.rcParams.update({'font.size': 6})


plt.xticks(np.arange(0, 5, step=1.0))
plt.yticks(np.arange(0, 5, step=1.0))
plt.tight_layout()

BIC = np.array(BIC_table)

plt.savefig('mixture1.png')
plt.savefig('mixture1.svg')
np.savetxt('BIC_mixture1.out', BIC, delimiter=',')
