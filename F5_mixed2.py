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

exponents = ([1.01, 2.0, 3.0, 4.0])
sample_size1 = int(argv[1])
sample_size2 = int(argv[2])
xmax = 1000
w = sample_size1/(sample_size1+sample_size2)


#ax1 1-1
f, ([ax1, ax2, ax3, ax4], [ax5, ax6, ax7, ax8], [ax9, ax10, ax11, ax12], [
    ax13, ax14, ax15, ax16]) = plt.subplots(4, 4, sharex=True, sharey=True)

exp1 = exponents[0]
exp2 = exponents[0]
data1_x1 = pl.power_law(exp1, xmax, sample_size1)
data1_x2 = pl.power_law(exp2, xmax, sample_size2)
mixed1 = np.concatenate((data1_x1, data1_x2), axis=0)

fits1=np.zeros(10)
fits2 = np.zeros(10)
weight = np.zeros(10)

fits1_std = np.zeros(10)
fits2_std = np.zeros(10)
weight_std = np.zeros(10)
for x in range(10):
    Fit1 = pl.Fit_Bayes(mixed1, niters=10000, mixed=2)
    fits1[x] = np.mean(Fit1.samples_1)
    fits2[x] = np.mean(Fit1.samples_2)
    weight[x] = np.mean(Fit1.samples_w)

    fits1_std[x] = np.std(Fit1.samples_1)
    fits2_std[x] = np.std(Fit1.samples_2)
    weight_std[x] = np.std(Fit1.samples_w)

ax1.axhline(y=exp1, color='black', linewidth=0.5)
ax1.axhline(y=exp2, color='black', linewidth=0.5)
ax1.axhline(y=w, color='black', linewidth=0.5)

ax1.scatter(np.array(range(1, 11)), fits1, marker='o',
            color='red', edgecolor='black')
ax1.errorbar(np.array(range(1, 11)), fits1, yerr=fits1_std,
            ls='none', color='black', elinewidth=0.5, capsize=2)

ax1.scatter(np.array(range(1, 11)), fits2, marker='o',
            color='grey', edgecolor='black')
ax1.errorbar(np.array(range(1, 11)), fits2, yerr=fits2_std,
            ls='none', color='black', elinewidth=0.5, capsize=2)

ax1.scatter(np.array(range(1, 11)), weight, marker='o',
            color='grey', edgecolor='black')
ax1.errorbar(np.array(range(1, 11)), weight, yerr=weight_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)

ax1.xaxis.set_label_position("top")
ax1.set_xlabel('1.01', fontsize=15)

#ax2 2-1
exp1 = exponents[1]
exp2 = exponents[0]
data1_x1 = pl.power_law(exp1, xmax, sample_size1)
data1_x2 = pl.power_law(exp2, xmax, sample_size2)
mixed1 = np.concatenate((data1_x1, data1_x2), axis=0)

fits1 = np.zeros(10)
fits2 = np.zeros(10)
weight = np.zeros(10)

fits1_std = np.zeros(10)
fits2_std = np.zeros(10)
weight_std = np.zeros(10)
for x in range(10):
    Fit1 = pl.Fit_Bayes(mixed1, niters=10000, mixed=2)
    fits1[x] = np.mean(Fit1.samples_1)
    fits2[x] = np.mean(Fit1.samples_2)
    weight[x] = np.mean(Fit1.samples_w)

    fits1_std[x] = np.std(Fit1.samples_1)
    fits2_std[x] = np.std(Fit1.samples_2)
    weight_std[x] = np.std(Fit1.samples_w)

ax2.axhline(y=exp1, color='black', linewidth=0.5)
ax2.axhline(y=exp2, color='black', linewidth=0.5)
ax2.axhline(y=w, color='black', linewidth=0.5)

ax2.scatter(np.array(range(1, 11)), fits1, marker='o',
            color='red', edgecolor='black')
ax2.errorbar(np.array(range(1, 11)), fits1, yerr=fits1_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)

ax2.scatter(np.array(range(1, 11)), fits2, marker='o',
            color='grey', edgecolor='black')
ax2.errorbar(np.array(range(1, 11)), fits2, yerr=fits2_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)

ax2.scatter(np.array(range(1, 11)), weight, marker='o',
            color='grey', edgecolor='black')
ax2.errorbar(np.array(range(1, 11)), weight, yerr=weight_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)
ax2.xaxis.set_label_position("top")
ax2.set_xlabel('2.0', fontsize=15)

#ax3 3-1
exp1 = exponents[2]
exp2 = exponents[0]
data1_x1 = pl.power_law(exp1, xmax, sample_size1)
data1_x2 = pl.power_law(exp2, xmax, sample_size2)
mixed1 = np.concatenate((data1_x1, data1_x2), axis=0)

fits1 = np.zeros(10)
fits2 = np.zeros(10)
weight = np.zeros(10)

fits1_std = np.zeros(10)
fits2_std = np.zeros(10)
weight_std = np.zeros(10)
for x in range(10):
    Fit1 = pl.Fit_Bayes(mixed1, niters=10000, mixed=2)
    fits1[x] = np.mean(Fit1.samples_1)
    fits2[x] = np.mean(Fit1.samples_2)
    weight[x] = np.mean(Fit1.samples_w)

    fits1_std[x] = np.std(Fit1.samples_1)
    fits2_std[x] = np.std(Fit1.samples_2)
    weight_std[x] = np.std(Fit1.samples_w)

ax3.axhline(y=exp1, color='black', linewidth=0.5)
ax3.axhline(y=exp2, color='black', linewidth=0.5)
ax3.axhline(y=w, color='black', linewidth=0.5)

ax3.scatter(np.array(range(1, 11)), fits1, marker='o',
            color='red', edgecolor='black')
ax3.errorbar(np.array(range(1, 11)), fits1, yerr=fits1_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)

ax3.scatter(np.array(range(1, 11)), fits2, marker='o',
            color='grey', edgecolor='black')
ax3.errorbar(np.array(range(1, 11)), fits2, yerr=fits2_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)

ax3.scatter(np.array(range(1, 11)), weight, marker='o',
            color='grey', edgecolor='black')
ax3.errorbar(np.array(range(1, 11)), weight, yerr=weight_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)
ax3.xaxis.set_label_position("top")
ax3.set_xlabel('3.0', fontsize=15)

#ax4 5-1

exp1 = exponents[3]
exp2 = exponents[0]
data1_x1 = pl.power_law(exp1, xmax, sample_size1)
data1_x2 = pl.power_law(exp2, xmax, sample_size2)
mixed1 = np.concatenate((data1_x1, data1_x2), axis=0)

fits1 = np.zeros(10)
fits2 = np.zeros(10)
weight = np.zeros(10)

fits1_std = np.zeros(10)
fits2_std = np.zeros(10)
weight_std = np.zeros(10)
for x in range(10):
    Fit1 = pl.Fit_Bayes(mixed1, niters=10000, mixed=2)
    fits1[x] = np.mean(Fit1.samples_1)
    fits2[x] = np.mean(Fit1.samples_2)
    weight[x] = np.mean(Fit1.samples_w)

    fits1_std[x] = np.std(Fit1.samples_1)
    fits2_std[x] = np.std(Fit1.samples_2)
    weight_std[x] = np.std(Fit1.samples_w)

ax4.axhline(y=exp1, color='black', linewidth=0.5)
ax4.axhline(y=exp2, color='black', linewidth=0.5)
ax4.axhline(y=w, color='black', linewidth=0.5)

ax4.scatter(np.array(range(1, 11)), fits1, marker='o',
            color='red', edgecolor='black')
ax4.errorbar(np.array(range(1, 11)), fits1, yerr=fits1_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)

ax4.scatter(np.array(range(1, 11)), fits2, marker='o',
            color='grey', edgecolor='black')
ax4.errorbar(np.array(range(1, 11)), fits2, yerr=fits2_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)

ax4.scatter(np.array(range(1, 11)), weight, marker='o',
            color='grey', edgecolor='black')
ax4.errorbar(np.array(range(1, 11)), weight, yerr=weight_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)
ax4.xaxis.set_label_position("top")
ax4.set_xlabel('4.0', fontsize=15)
ax4.yaxis.set_label_position("right")
ax4.set_ylabel('1.01', fontsize=15, rotation=270, labelpad=15)

#ax5 1-2

exp1 = exponents[0]
exp2 = exponents[1]
data1_x1 = pl.power_law(exp1, xmax, sample_size1)
data1_x2 = pl.power_law(exp2, xmax, sample_size2)
mixed1 = np.concatenate((data1_x1, data1_x2), axis=0)

fits1 = np.zeros(10)
fits2 = np.zeros(10)
weight = np.zeros(10)

fits1_std = np.zeros(10)
fits2_std = np.zeros(10)
weight_std = np.zeros(10)
for x in range(10):
    Fit1 = pl.Fit_Bayes(mixed1, niters=10000, mixed=2)
    fits1[x] = np.mean(Fit1.samples_1)
    fits2[x] = np.mean(Fit1.samples_2)
    weight[x] = np.mean(Fit1.samples_w)

    fits1_std[x] = np.std(Fit1.samples_1)
    fits2_std[x] = np.std(Fit1.samples_2)
    weight_std[x] = np.std(Fit1.samples_w)

ax5.axhline(y=exp1, color='black', linewidth=0.5)
ax5.axhline(y=exp2, color='black', linewidth=0.5)
ax5.axhline(y=w, color='black', linewidth=0.5)

ax5.scatter(np.array(range(1, 11)), fits1, marker='o',
            color='red', edgecolor='black')
ax5.errorbar(np.array(range(1, 11)), fits1, yerr=fits1_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)

ax5.scatter(np.array(range(1, 11)), fits2, marker='o',
            color='grey', edgecolor='black')
ax5.errorbar(np.array(range(1, 11)), fits2, yerr=fits2_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)

ax5.scatter(np.array(range(1, 11)), weight, marker='o',
            color='grey', edgecolor='black')
ax5.errorbar(np.array(range(1, 11)), weight, yerr=weight_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)

#ax6 2-2
exp1 = exponents[1]
exp2 = exponents[1]
data1_x1 = pl.power_law(exp1, xmax, sample_size1)
data1_x2 = pl.power_law(exp2, xmax, sample_size2)
mixed1 = np.concatenate((data1_x1, data1_x2), axis=0)

fits1 = np.zeros(10)
fits2 = np.zeros(10)
weight = np.zeros(10)

fits1_std = np.zeros(10)
fits2_std = np.zeros(10)
weight_std = np.zeros(10)
for x in range(10):
    Fit1 = pl.Fit_Bayes(mixed1, niters=10000, mixed=2)
    fits1[x] = np.mean(Fit1.samples_1)
    fits2[x] = np.mean(Fit1.samples_2)
    weight[x] = np.mean(Fit1.samples_w)

    fits1_std[x] = np.std(Fit1.samples_1)
    fits2_std[x] = np.std(Fit1.samples_2)
    weight_std[x] = np.std(Fit1.samples_w)

ax6.axhline(y=exp1, color='black', linewidth=0.5)
ax6.axhline(y=exp2, color='black', linewidth=0.5)
ax6.axhline(y=w, color='black', linewidth=0.5)

ax6.scatter(np.array(range(1, 11)), fits1, marker='o',
            color='red', edgecolor='black')
ax6.errorbar(np.array(range(1, 11)), fits1, yerr=fits1_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)

ax6.scatter(np.array(range(1, 11)), fits2, marker='o',
            color='grey', edgecolor='black')
ax6.errorbar(np.array(range(1, 11)), fits2, yerr=fits2_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)

ax6.scatter(np.array(range(1, 11)), weight, marker='o',
            color='grey', edgecolor='black')
ax6.errorbar(np.array(range(1, 11)), weight, yerr=weight_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)

#ax7 3-2
exp1 = exponents[2]
exp2 = exponents[1]
data1_x1 = pl.power_law(exp1, xmax, sample_size1)
data1_x2 = pl.power_law(exp2, xmax, sample_size2)
mixed1 = np.concatenate((data1_x1, data1_x2), axis=0)

fits1 = np.zeros(10)
fits2 = np.zeros(10)
weight = np.zeros(10)

fits1_std = np.zeros(10)
fits2_std = np.zeros(10)
weight_std = np.zeros(10)
for x in range(10):
    Fit1 = pl.Fit_Bayes(mixed1, niters=10000, mixed=2)
    fits1[x] = np.mean(Fit1.samples_1)
    fits2[x] = np.mean(Fit1.samples_2)
    weight[x] = np.mean(Fit1.samples_w)

    fits1_std[x] = np.std(Fit1.samples_1)
    fits2_std[x] = np.std(Fit1.samples_2)
    weight_std[x] = np.std(Fit1.samples_w)

ax7.axhline(y=exp1, color='black', linewidth=0.5)
ax7.axhline(y=exp2, color='black', linewidth=0.5)
ax7.axhline(y=w, color='black', linewidth=0.5)

ax7.scatter(np.array(range(1, 11)), fits1, marker='o',
            color='red', edgecolor='black')
ax7.errorbar(np.array(range(1, 11)), fits1, yerr=fits1_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)

ax7.scatter(np.array(range(1, 11)), fits2, marker='o',
            color='grey', edgecolor='black')
ax7.errorbar(np.array(range(1, 11)), fits2, yerr=fits2_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)

ax7.scatter(np.array(range(1, 11)), weight, marker='o',
            color='grey', edgecolor='black')
ax7.errorbar(np.array(range(1, 11)), weight, yerr=weight_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)

#ax8 5-2

exp1 = exponents[3]
exp2 = exponents[1]
data1_x1 = pl.power_law(exp1, xmax, sample_size1)
data1_x2 = pl.power_law(exp2, xmax, sample_size2)
mixed1 = np.concatenate((data1_x1, data1_x2), axis=0)

fits1 = np.zeros(10)
fits2 = np.zeros(10)
weight = np.zeros(10)

fits1_std = np.zeros(10)
fits2_std = np.zeros(10)
weight_std = np.zeros(10)
for x in range(10):
    Fit1 = pl.Fit_Bayes(mixed1, niters=10000, mixed=2)
    fits1[x] = np.mean(Fit1.samples_1)
    fits2[x] = np.mean(Fit1.samples_2)
    weight[x] = np.mean(Fit1.samples_w)

    fits1_std[x] = np.std(Fit1.samples_1)
    fits2_std[x] = np.std(Fit1.samples_2)
    weight_std[x] = np.std(Fit1.samples_w)

ax8.axhline(y=exp1, color='black', linewidth=0.5)
ax8.axhline(y=exp2, color='black', linewidth=0.5)
ax8.axhline(y=w, color='black', linewidth=0.5)

ax8.scatter(np.array(range(1, 11)), fits1, marker='o',
            color='red', edgecolor='black')
ax8.errorbar(np.array(range(1, 11)), fits1, yerr=fits1_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)

ax8.scatter(np.array(range(1, 11)), fits2, marker='o',
            color='grey', edgecolor='black')
ax8.errorbar(np.array(range(1, 11)), fits2, yerr=fits2_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)

ax8.scatter(np.array(range(1, 11)), weight, marker='o',
            color='grey', edgecolor='black')
ax8.errorbar(np.array(range(1, 11)), weight, yerr=weight_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)

ax8.yaxis.set_label_position("right")
ax8.set_ylabel('2.0', fontsize=15, rotation=270, labelpad=15)

#ax9 1-3
exp1 = exponents[0]
exp2 = exponents[2]
data1_x1 = pl.power_law(exp1, xmax, sample_size1)
data1_x2 = pl.power_law(exp2, xmax, sample_size2)
mixed1 = np.concatenate((data1_x1, data1_x2), axis=0)

fits1 = np.zeros(10)
fits2 = np.zeros(10)
weight = np.zeros(10)

fits1_std = np.zeros(10)
fits2_std = np.zeros(10)
weight_std = np.zeros(10)
for x in range(10):
    Fit1 = pl.Fit_Bayes(mixed1, niters=10000, mixed=2)
    fits1[x] = np.mean(Fit1.samples_1)
    fits2[x] = np.mean(Fit1.samples_2)
    weight[x] = np.mean(Fit1.samples_w)

    fits1_std[x] = np.std(Fit1.samples_1)
    fits2_std[x] = np.std(Fit1.samples_2)
    weight_std[x] = np.std(Fit1.samples_w)

ax9.axhline(y=exp1, color='black', linewidth=0.5)
ax9.axhline(y=exp2, color='black', linewidth=0.5)
ax9.axhline(y=w, color='black', linewidth=0.5)

ax9.scatter(np.array(range(1, 11)), fits1, marker='o',
            color='red', edgecolor='black')
ax9.errorbar(np.array(range(1, 11)), fits1, yerr=fits1_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)

ax9.scatter(np.array(range(1, 11)), fits2, marker='o',
            color='grey', edgecolor='black')
ax9.errorbar(np.array(range(1, 11)), fits2, yerr=fits2_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)

ax9.scatter(np.array(range(1, 11)), weight, marker='o',
            color='grey', edgecolor='black')
ax9.errorbar(np.array(range(1, 11)), weight, yerr=weight_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)

#ax10 2-3
exp1 = exponents[1]
exp2 = exponents[2]
data1_x1 = pl.power_law(exp1, xmax, sample_size1)
data1_x2 = pl.power_law(exp2, xmax, sample_size2)
mixed1 = np.concatenate((data1_x1, data1_x2), axis=0)

fits1 = np.zeros(10)
fits2 = np.zeros(10)
weight = np.zeros(10)

fits1_std = np.zeros(10)
fits2_std = np.zeros(10)
weight_std = np.zeros(10)
for x in range(10):
    Fit1 = pl.Fit_Bayes(mixed1, niters=10000, mixed=2)
    fits1[x] = np.mean(Fit1.samples_1)
    fits2[x] = np.mean(Fit1.samples_2)
    weight[x] = np.mean(Fit1.samples_w)

    fits1_std[x] = np.std(Fit1.samples_1)
    fits2_std[x] = np.std(Fit1.samples_2)
    weight_std[x] = np.std(Fit1.samples_w)

ax10.axhline(y=exp1, color='black', linewidth=0.5)
ax10.axhline(y=exp2, color='black', linewidth=0.5)
ax10.axhline(y=w, color='black', linewidth=0.5)

ax10.scatter(np.array(range(1, 11)), fits1, marker='o',
            color='red', edgecolor='black')
ax10.errorbar(np.array(range(1, 11)), fits1, yerr=fits1_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)

ax10.scatter(np.array(range(1, 11)), fits2, marker='o',
            color='grey', edgecolor='black')
ax10.errorbar(np.array(range(1, 11)), fits2, yerr=fits2_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)

ax10.scatter(np.array(range(1, 11)), weight, marker='o',
            color='grey', edgecolor='black')
ax10.errorbar(np.array(range(1, 11)), weight, yerr=weight_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)

#ax11 3-3
exp1 = exponents[2]
exp2 = exponents[2]
data1_x1 = pl.power_law(exp1, xmax, sample_size1)
data1_x2 = pl.power_law(exp2, xmax, sample_size2)
mixed1 = np.concatenate((data1_x1, data1_x2), axis=0)

fits1 = np.zeros(10)
fits2 = np.zeros(10)
weight = np.zeros(10)

fits1_std = np.zeros(10)
fits2_std = np.zeros(10)
weight_std = np.zeros(10)
for x in range(10):
    Fit1 = pl.Fit_Bayes(mixed1, niters=10000, mixed=2)
    fits1[x] = np.mean(Fit1.samples_1)
    fits2[x] = np.mean(Fit1.samples_2)
    weight[x] = np.mean(Fit1.samples_w)

    fits1_std[x] = np.std(Fit1.samples_1)
    fits2_std[x] = np.std(Fit1.samples_2)
    weight_std[x] = np.std(Fit1.samples_w)

ax11.axhline(y=exp1, color='black', linewidth=0.5)
ax11.axhline(y=exp2, color='black', linewidth=0.5)
ax11.axhline(y=w, color='black', linewidth=0.5)

ax11.scatter(np.array(range(1, 11)), fits1, marker='o',
            color='red', edgecolor='black')
ax11.errorbar(np.array(range(1, 11)), fits1, yerr=fits1_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)

ax11.scatter(np.array(range(1, 11)), fits2, marker='o',
            color='grey', edgecolor='black')
ax11.errorbar(np.array(range(1, 11)), fits2, yerr=fits2_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)

ax11.scatter(np.array(range(1, 11)), weight, marker='o',
            color='grey', edgecolor='black')
ax11.errorbar(np.array(range(1, 11)), weight, yerr=weight_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)

#ax12 5-3
exp1 = exponents[3]
exp2 = exponents[2]
data1_x1 = pl.power_law(exp1, xmax, sample_size1)
data1_x2 = pl.power_law(exp2, xmax, sample_size2)
mixed1 = np.concatenate((data1_x1, data1_x2), axis=0)

fits1 = np.zeros(10)
fits2 = np.zeros(10)
weight = np.zeros(10)

fits1_std = np.zeros(10)
fits2_std = np.zeros(10)
weight_std = np.zeros(10)
for x in range(10):
    Fit1 = pl.Fit_Bayes(mixed1, niters=10000, mixed=2)
    fits1[x] = np.mean(Fit1.samples_1)
    fits2[x] = np.mean(Fit1.samples_2)
    weight[x] = np.mean(Fit1.samples_w)

    fits1_std[x] = np.std(Fit1.samples_1)
    fits2_std[x] = np.std(Fit1.samples_2)
    weight_std[x] = np.std(Fit1.samples_w)

ax12.axhline(y=exp1, color='black', linewidth=0.5)
ax12.axhline(y=exp2, color='black', linewidth=0.5)
ax12.axhline(y=w, color='black', linewidth=0.5)

ax12.scatter(np.array(range(1, 11)), fits1, marker='o',
            color='red', edgecolor='black')
ax12.errorbar(np.array(range(1, 11)), fits1, yerr=fits1_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)

ax12.scatter(np.array(range(1, 11)), fits2, marker='o',
            color='grey', edgecolor='black')
ax12.errorbar(np.array(range(1, 11)), fits2, yerr=fits2_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)

ax12.scatter(np.array(range(1, 11)), weight, marker='o',
            color='grey', edgecolor='black')
ax12.errorbar(np.array(range(1, 11)), weight, yerr=weight_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)

ax12.yaxis.set_label_position("right")
ax12.set_ylabel('3.0', fontsize=15, rotation=270, labelpad=15)

#ax13 1-5
exp1 = exponents[0]
exp2 = exponents[3]
data1_x1 = pl.power_law(exp1, xmax, sample_size1)
data1_x2 = pl.power_law(exp2, xmax, sample_size2)
mixed1 = np.concatenate((data1_x1, data1_x2), axis=0)

fits1 = np.zeros(10)
fits2 = np.zeros(10)
weight = np.zeros(10)

fits1_std = np.zeros(10)
fits2_std = np.zeros(10)
weight_std = np.zeros(10)
for x in range(10):
    Fit1 = pl.Fit_Bayes(mixed1, niters=10000, mixed=2)
    fits1[x] = np.mean(Fit1.samples_1)
    fits2[x] = np.mean(Fit1.samples_2)
    weight[x] = np.mean(Fit1.samples_w)

    fits1_std[x] = np.std(Fit1.samples_1)
    fits2_std[x] = np.std(Fit1.samples_2)
    weight_std[x] = np.std(Fit1.samples_w)

ax13.axhline(y=exp1, color='black', linewidth=0.5)
ax13.axhline(y=exp2, color='black', linewidth=0.5)
ax13.axhline(y=w, color='black', linewidth=0.5)

ax13.scatter(np.array(range(1, 11)), fits1, marker='o',
            color='red', edgecolor='black')
ax13.errorbar(np.array(range(1, 11)), fits1, yerr=fits1_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)

ax13.scatter(np.array(range(1, 11)), fits2, marker='o',
            color='grey', edgecolor='black')
ax13.errorbar(np.array(range(1, 11)), fits2, yerr=fits2_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)

ax13.scatter(np.array(range(1, 11)), weight, marker='o',
            color='grey', edgecolor='black')
ax13.errorbar(np.array(range(1, 11)), weight, yerr=weight_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)

#ax14 2-5
exp1 = exponents[1]
exp2 = exponents[3]
data1_x1 = pl.power_law(exp1, xmax, sample_size1)
data1_x2 = pl.power_law(exp2, xmax, sample_size2)
mixed1 = np.concatenate((data1_x1, data1_x2), axis=0)

fits1 = np.zeros(10)
fits2 = np.zeros(10)
weight = np.zeros(10)

fits1_std = np.zeros(10)
fits2_std = np.zeros(10)
weight_std = np.zeros(10)
for x in range(10):
    Fit1 = pl.Fit_Bayes(mixed1, niters=10000, mixed=2)
    fits1[x] = np.mean(Fit1.samples_1)
    fits2[x] = np.mean(Fit1.samples_2)
    weight[x] = np.mean(Fit1.samples_w)

    fits1_std[x] = np.std(Fit1.samples_1)
    fits2_std[x] = np.std(Fit1.samples_2)
    weight_std[x] = np.std(Fit1.samples_w)

ax14.axhline(y=exp1, color='black', linewidth=0.5)
ax14.axhline(y=exp2, color='black', linewidth=0.5)
ax14.axhline(y=w, color='black', linewidth=0.5)

ax14.scatter(np.array(range(1, 11)), fits1, marker='o',
            color='red', edgecolor='black')
ax14.errorbar(np.array(range(1, 11)), fits1, yerr=fits1_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)

ax14.scatter(np.array(range(1, 11)), fits2, marker='o',
            color='grey', edgecolor='black')
ax14.errorbar(np.array(range(1, 11)), fits2, yerr=fits2_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)

ax14.scatter(np.array(range(1, 11)), weight, marker='o',
            color='grey', edgecolor='black')
ax14.errorbar(np.array(range(1, 11)), weight, yerr=weight_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)

#ax15 3-5
exp1 = exponents[2]
exp2 = exponents[3]
data1_x1 = pl.power_law(exp1, xmax, sample_size1)
data1_x2 = pl.power_law(exp2, xmax, sample_size2)
mixed1 = np.concatenate((data1_x1, data1_x2), axis=0)

fits1 = np.zeros(10)
fits2 = np.zeros(10)
weight = np.zeros(10)

fits1_std = np.zeros(10)
fits2_std = np.zeros(10)
weight_std = np.zeros(10)
for x in range(10):
    Fit1 = pl.Fit_Bayes(mixed1, niters=10000, mixed=2)
    fits1[x] = np.mean(Fit1.samples_1)
    fits2[x] = np.mean(Fit1.samples_2)
    weight[x] = np.mean(Fit1.samples_w)

    fits1_std[x] = np.std(Fit1.samples_1)
    fits2_std[x] = np.std(Fit1.samples_2)
    weight_std[x] = np.std(Fit1.samples_w)

ax15.axhline(y=exp1, color='black', linewidth=0.5)
ax15.axhline(y=exp2, color='black', linewidth=0.5)
ax15.axhline(y=w, color='black', linewidth=0.5)

ax15.scatter(np.array(range(1, 11)), fits1, marker='o',
            color='red', edgecolor='black')
ax15.errorbar(np.array(range(1, 11)), fits1, yerr=fits1_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)

ax15.scatter(np.array(range(1, 11)), fits2, marker='o',
            color='grey', edgecolor='black')
ax15.errorbar(np.array(range(1, 11)), fits2, yerr=fits2_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)

ax15.scatter(np.array(range(1, 11)), weight, marker='o',
            color='grey', edgecolor='black')
ax15.errorbar(np.array(range(1, 11)), weight, yerr=weight_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)

#ax16 5-5
exp1 = exponents[3]
exp2 = exponents[3]
data1_x1 = pl.power_law(exp1, xmax, sample_size1)
data1_x2 = pl.power_law(exp2, xmax, sample_size2)
mixed1 = np.concatenate((data1_x1, data1_x2), axis=0)

fits1 = np.zeros(10)
fits2 = np.zeros(10)
weight = np.zeros(10)

fits1_std = np.zeros(10)
fits2_std = np.zeros(10)
weight_std = np.zeros(10)
for x in range(10):
    Fit1 = pl.Fit_Bayes(mixed1, niters=10000, mixed=2)
    fits1[x] = np.mean(Fit1.samples_1)
    fits2[x] = np.mean(Fit1.samples_2)
    weight[x] = np.mean(Fit1.samples_w)

    fits1_std[x] = np.std(Fit1.samples_1)
    fits2_std[x] = np.std(Fit1.samples_2)
    weight_std[x] = np.std(Fit1.samples_w)

ax16.axhline(y=exp1, color='black', linewidth=0.5)
ax16.axhline(y=exp2, color='black', linewidth=0.5)
ax16.axhline(y=w, color='black', linewidth=0.5)

ax16.scatter(np.array(range(1, 11)), fits1, marker='o',
            color='red', edgecolor='black')
ax16.errorbar(np.array(range(1, 11)), fits1, yerr=fits1_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)

ax16.scatter(np.array(range(1, 11)), fits2, marker='o',
            color='grey', edgecolor='black')
ax16.errorbar(np.array(range(1, 11)), fits2, yerr=fits2_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)

ax16.scatter(np.array(range(1, 11)), weight, marker='o',
            color='grey', edgecolor='black')
ax16.errorbar(np.array(range(1, 11)), weight, yerr=weight_std,
             ls='none', color='black', elinewidth=0.5, capsize=2)

ax16.yaxis.set_label_position("right")
ax16.set_ylabel('4.0', fontsize=15, rotation=270, labelpad=15)


matplotlib.rcParams.update({'font.size': 6})

f.text(0.495, -0.05, 'Xmin', ha='center', fontsize=15)
f.text(-0.03, 0.5, 'Fitted exponent', va='center',
    rotation='vertical', fontsize=15)

ax1.set_ylim([0, 6])
ax1.set_xlim([0.5, 11.5])
plt.tight_layout()

plt.savefig('mixed_n{}-{}.png'.format(int(sample_size1), int(sample_size2)))
plt.savefig('mixed_n{}-{}.svg'.format(int(sample_size1), int(sample_size2)))
