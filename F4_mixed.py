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
xmax_r = int(argv[2])
sample_ratios = np.arange(1, 11)
xmin_tofit = ([1, 2, 3, 4, 5, 6])
xmax1 = 1000
xmax2 = 1000 * xmax_r

for i in sample_ratios:
    sample_size2=sample_size1*i
    #ax1 1-1
    f, ([ax1, ax2, ax3, ax4], [ax5, ax6, ax7, ax8], [ax9, ax10, ax11, ax12], [
        ax13, ax14, ax15, ax16]) = plt.subplots(4, 4, sharex=True, sharey=True)

    exp1 = exponents[0]
    exp2 = exponents[0]

    data1_x1 = pl.power_law(exp1, xmax1, sample_size1)
    data1_x2 = pl.power_law(exp2, xmax2, sample_size2)
    data2_x1 = pl.power_law(exp1, xmax2, sample_size1)
    data2_x2 = pl.power_law(exp2, xmax1, sample_size2)
    mixed1 = np.concatenate((data1_x1, data1_x2), axis=0)
    mixed2 = np.concatenate((data2_x1, data2_x2), axis=0)
    xmin_fits1 = np.zeros(len(xmin_tofit))
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

    ax1.axhline(y=exp1, color='black', linewidth=0.5)
    ax1.axhline(y=exp2, color='black', linewidth=0.5)
    ax1.plot(np.array(range(1, 7)), xmin_fits1, 'red')
    ax1.scatter(np.array(range(1, 7)), xmin_fits1, marker='o',
                color='red', edgecolor='black')
    ax1.errorbar(np.array(range(1, 7)), xmin_fits1, yerr=xmin_std1,
                ls='none', color='black', elinewidth=0.5, capsize=2)

    ax1.plot(np.array(range(1, 7)), xmin_fits2, 'blue')
    ax1.scatter(np.array(range(1, 7)), xmin_fits2, marker='o',
                color='grey', edgecolor='black')
    ax1.errorbar(np.array(range(1, 7)), xmin_fits2, yerr=xmin_std2,
                ls='none', color='black', elinewidth=0.5, capsize=2)
    ax1.xaxis.set_label_position("top")
    ax1.set_xlabel('1.01', fontsize=15)

    #ax2 2-1
    exp1 = exponents[1]
    exp2 = exponents[0]
    data1_x1 = pl.power_law(exp1, xmax1, sample_size1)
    data1_x2 = pl.power_law(exp2, xmax2, sample_size2)
    data2_x1 = pl.power_law(exp1, xmax2, sample_size1)
    data2_x2 = pl.power_law(exp2, xmax1, sample_size2)
    mixed1 = np.concatenate((data1_x1, data1_x2), axis=0)
    mixed2 = np.concatenate((data2_x1, data2_x2), axis=0)
    xmin_fits1 = np.zeros(len(xmin_tofit))
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

    ax2.axhline(y=exp1, color='black', linewidth=0.5)
    ax2.axhline(y=exp2, color='black', linewidth=0.5)
    ax2.plot(np.array(range(1, 7)), xmin_fits1, 'red')
    ax2.scatter(np.array(range(1, 7)), xmin_fits1, marker='o',
                color='red', edgecolor='black')
    ax2.errorbar(np.array(range(1, 7)), xmin_fits1, yerr=xmin_std1,
                ls='none', color='black', elinewidth=0.5, capsize=2)

    ax2.plot(np.array(range(1, 7)), xmin_fits2, 'blue')
    ax2.scatter(np.array(range(1, 7)), xmin_fits2, marker='o',
                color='grey', edgecolor='black')
    ax2.errorbar(np.array(range(1, 7)), xmin_fits2, yerr=xmin_std2,
                ls='none', color='black', elinewidth=0.5, capsize=2)
    ax2.xaxis.set_label_position("top")
    ax2.set_xlabel('2.0', fontsize=15)

    #ax3 3-1
    exp1 = exponents[2]
    exp2 = exponents[0]
    data1_x1 = pl.power_law(exp1, xmax1, sample_size1)
    data1_x2 = pl.power_law(exp2, xmax2, sample_size2)
    data2_x1 = pl.power_law(exp1, xmax2, sample_size1)
    data2_x2 = pl.power_law(exp2, xmax1, sample_size2)
    mixed1 = np.concatenate((data1_x1, data1_x2), axis=0)
    mixed2 = np.concatenate((data2_x1, data2_x2), axis=0)
    xmin_fits1 = np.zeros(len(xmin_tofit))
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

    ax3.axhline(y=exp1, color='black', linewidth=0.5)
    ax3.axhline(y=exp2, color='black', linewidth=0.5)
    ax3.plot(np.array(range(1, 7)), xmin_fits1, 'red')
    ax3.scatter(np.array(range(1, 7)), xmin_fits1, marker='o',
                color='red', edgecolor='black')
    ax3.errorbar(np.array(range(1, 7)), xmin_fits1, yerr=xmin_std1,
                ls='none', color='black', elinewidth=0.5, capsize=2)

    ax3.plot(np.array(range(1, 7)), xmin_fits2, 'blue')
    ax3.scatter(np.array(range(1, 7)), xmin_fits2, marker='o',
                color='grey', edgecolor='black')
    ax3.errorbar(np.array(range(1, 7)), xmin_fits2, yerr=xmin_std2,
                ls='none', color='black', elinewidth=0.5, capsize=2)
    ax3.xaxis.set_label_position("top")
    ax3.set_xlabel('3.0', fontsize=15)

    #ax4 5-1

    exp1 = exponents[3]
    exp2 = exponents[0]
    data1_x1 = pl.power_law(exp1, xmax1, sample_size1)
    data1_x2 = pl.power_law(exp2, xmax2, sample_size2)
    data2_x1 = pl.power_law(exp1, xmax2, sample_size1)
    data2_x2 = pl.power_law(exp2, xmax1, sample_size2)
    mixed1 = np.concatenate((data1_x1, data1_x2), axis=0)
    mixed2 = np.concatenate((data2_x1, data2_x2), axis=0)
    xmin_fits1 = np.zeros(len(xmin_tofit))
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

    ax4.axhline(y=exp1, color='black', linewidth=0.5)
    ax4.axhline(y=exp2, color='black', linewidth=0.5)
    ax4.plot(np.array(range(1, 7)), xmin_fits1, 'red')
    ax4.scatter(np.array(range(1, 7)), xmin_fits1, marker='o',
                color='red', edgecolor='black')
    ax4.errorbar(np.array(range(1, 7)), xmin_fits1, yerr=xmin_std1,
                ls='none', color='black', elinewidth=0.5, capsize=2)

    ax4.plot(np.array(range(1, 7)), xmin_fits2, 'blue')
    ax4.scatter(np.array(range(1, 7)), xmin_fits2, marker='o',
                color='grey', edgecolor='black')
    ax4.errorbar(np.array(range(1, 7)), xmin_fits2, yerr=xmin_std2,
                ls='none', color='black', elinewidth=0.5, capsize=2)
    ax4.xaxis.set_label_position("top")
    ax4.set_xlabel('4.0', fontsize=15)
    ax4.yaxis.set_label_position("right")
    ax4.set_ylabel('1.01', fontsize=15, rotation=270, labelpad=15)

    #ax5 1-2

    exp1 = exponents[0]
    exp2 = exponents[1]
    data1_x1 = pl.power_law(exp1, xmax1, sample_size1)
    data1_x2 = pl.power_law(exp2, xmax2, sample_size2)
    data2_x1 = pl.power_law(exp1, xmax2, sample_size1)
    data2_x2 = pl.power_law(exp2, xmax1, sample_size2)
    mixed1 = np.concatenate((data1_x1, data1_x2), axis=0)
    mixed2 = np.concatenate((data2_x1, data2_x2), axis=0)
    xmin_fits1 = np.zeros(len(xmin_tofit))
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

    ax5.axhline(y=exp1, color='black', linewidth=0.5)
    ax5.axhline(y=exp2, color='black', linewidth=0.5)
    ax5.plot(np.array(range(1, 7)), xmin_fits1, 'red')
    ax5.scatter(np.array(range(1, 7)), xmin_fits1, marker='o',
                color='red', edgecolor='black')
    ax5.errorbar(np.array(range(1, 7)), xmin_fits1, yerr=xmin_std1,
                ls='none', color='black', elinewidth=0.5, capsize=2)

    ax5.plot(np.array(range(1, 7)), xmin_fits2, 'blue')
    ax5.scatter(np.array(range(1, 7)), xmin_fits2, marker='o',
                color='grey', edgecolor='black')
    ax5.errorbar(np.array(range(1, 7)), xmin_fits2, yerr=xmin_std2,
                ls='none', color='black', elinewidth=0.5, capsize=2)

    #ax6 2-2
    exp1 = exponents[1]
    exp2 = exponents[1]
    data1_x1 = pl.power_law(exp1, xmax1, sample_size1)
    data1_x2 = pl.power_law(exp2, xmax2, sample_size2)
    data2_x1 = pl.power_law(exp1, xmax2, sample_size1)
    data2_x2 = pl.power_law(exp2, xmax1, sample_size2)
    mixed1 = np.concatenate((data1_x1, data1_x2), axis=0)
    mixed2 = np.concatenate((data2_x1, data2_x2), axis=0)
    xmin_fits1 = np.zeros(len(xmin_tofit))
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

    ax6.axhline(y=exp1, color='black', linewidth=0.5)
    ax6.axhline(y=exp2, color='black', linewidth=0.5)
    ax6.plot(np.array(range(1, 7)), xmin_fits1, 'red')
    ax6.scatter(np.array(range(1, 7)), xmin_fits1, marker='o',
                color='red', edgecolor='black')
    ax6.errorbar(np.array(range(1, 7)), xmin_fits1, yerr=xmin_std1,
                ls='none', color='black', elinewidth=0.5, capsize=2)

    ax6.plot(np.array(range(1, 7)), xmin_fits2, 'blue')
    ax6.scatter(np.array(range(1, 7)), xmin_fits2, marker='o',
                color='grey', edgecolor='black')
    ax6.errorbar(np.array(range(1, 7)), xmin_fits2, yerr=xmin_std2,
                ls='none', color='black', elinewidth=0.5, capsize=2)

    #ax7 3-2
    exp1 = exponents[2]
    exp2 = exponents[1]
    data1_x1 = pl.power_law(exp1, xmax1, sample_size1)
    data1_x2 = pl.power_law(exp2, xmax2, sample_size2)
    data2_x1 = pl.power_law(exp1, xmax2, sample_size1)
    data2_x2 = pl.power_law(exp2, xmax1, sample_size2)
    mixed1 = np.concatenate((data1_x1, data1_x2), axis=0)
    mixed2 = np.concatenate((data2_x1, data2_x2), axis=0)
    xmin_fits1 = np.zeros(len(xmin_tofit))
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

    ax7.axhline(y=exp1, color='black', linewidth=0.5)
    ax7.axhline(y=exp2, color='black', linewidth=0.5)
    ax7.plot(np.array(range(1, 7)), xmin_fits1, 'red')
    ax7.scatter(np.array(range(1, 7)), xmin_fits1, marker='o',
                color='red', edgecolor='black')
    ax7.errorbar(np.array(range(1, 7)), xmin_fits1, yerr=xmin_std1,
                ls='none', color='black', elinewidth=0.5, capsize=2)

    ax7.plot(np.array(range(1, 7)), xmin_fits2, 'blue')
    ax7.scatter(np.array(range(1, 7)), xmin_fits2, marker='o',
                color='grey', edgecolor='black')
    ax7.errorbar(np.array(range(1, 7)), xmin_fits2, yerr=xmin_std2,
                ls='none', color='black', elinewidth=0.5, capsize=2)

    #ax8 5-2

    exp1 = exponents[3]
    exp2 = exponents[1]
    data1_x1 = pl.power_law(exp1, xmax1, sample_size1)
    data1_x2 = pl.power_law(exp2, xmax2, sample_size2)
    data2_x1 = pl.power_law(exp1, xmax2, sample_size1)
    data2_x2 = pl.power_law(exp2, xmax1, sample_size2)
    mixed1 = np.concatenate((data1_x1, data1_x2), axis=0)
    mixed2 = np.concatenate((data2_x1, data2_x2), axis=0)
    xmin_fits1 = np.zeros(len(xmin_tofit))
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

    ax8.axhline(y=exp1, color='black', linewidth=0.5)
    ax8.axhline(y=exp2, color='black', linewidth=0.5)
    ax8.plot(np.array(range(1, 7)), xmin_fits1, 'red')
    ax8.scatter(np.array(range(1, 7)), xmin_fits1, marker='o',
                color='red', edgecolor='black')
    ax8.errorbar(np.array(range(1, 7)), xmin_fits1, yerr=xmin_std1,
                ls='none', color='black', elinewidth=0.5, capsize=2)

    ax8.plot(np.array(range(1, 7)), xmin_fits2, 'blue')
    ax8.scatter(np.array(range(1, 7)), xmin_fits2, marker='o',
                color='grey', edgecolor='black')
    ax8.errorbar(np.array(range(1, 7)), xmin_fits2, yerr=xmin_std2,
                ls='none', color='black', elinewidth=0.5, capsize=2)

    ax8.yaxis.set_label_position("right")
    ax8.set_ylabel('2.0', fontsize=15, rotation=270, labelpad=15)

    #ax9 1-3
    exp1 = exponents[0]
    exp2 = exponents[2]
    data1_x1 = pl.power_law(exp1, xmax1, sample_size1)
    data1_x2 = pl.power_law(exp2, xmax2, sample_size2)
    data2_x1 = pl.power_law(exp1, xmax2, sample_size1)
    data2_x2 = pl.power_law(exp2, xmax1, sample_size2)
    mixed1 = np.concatenate((data1_x1, data1_x2), axis=0)
    mixed2 = np.concatenate((data2_x1, data2_x2), axis=0)
    xmin_fits1 = np.zeros(len(xmin_tofit))
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

    ax9.axhline(y=exp1, color='black', linewidth=0.5)
    ax9.axhline(y=exp2, color='black', linewidth=0.5)
    ax9.plot(np.array(range(1, 7)), xmin_fits1, 'red')
    ax9.scatter(np.array(range(1, 7)), xmin_fits1, marker='o',
            color='red', edgecolor='black')
    ax9.errorbar(np.array(range(1, 7)), xmin_fits1, yerr=xmin_std1,
                ls='none', color='black', elinewidth=0.5, capsize=2)

    ax9.plot(np.array(range(1, 7)), xmin_fits2, 'blue')
    ax9.scatter(np.array(range(1, 7)), xmin_fits2, marker='o',
                color='grey', edgecolor='black')
    ax9.errorbar(np.array(range(1, 7)), xmin_fits2, yerr=xmin_std2,
                ls='none', color='black', elinewidth=0.5, capsize=2)

    #ax10 2-3
    exp1 = exponents[1]
    exp2 = exponents[2]
    data1_x1 = pl.power_law(exp1, xmax1, sample_size1)
    data1_x2 = pl.power_law(exp2, xmax2, sample_size2)
    data2_x1 = pl.power_law(exp1, xmax2, sample_size1)
    data2_x2 = pl.power_law(exp2, xmax1, sample_size2)
    mixed1 = np.concatenate((data1_x1, data1_x2), axis=0)
    mixed2 = np.concatenate((data2_x1, data2_x2), axis=0)
    xmin_fits1 = np.zeros(len(xmin_tofit))
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

    ax10.axhline(y=exp1, color='black', linewidth=0.5)
    ax10.axhline(y=exp2, color='black', linewidth=0.5)
    ax10.plot(np.array(range(1, 7)), xmin_fits1, 'red')
    ax10.scatter(np.array(range(1, 7)), xmin_fits1, marker='o',
                color='red', edgecolor='black')
    ax10.errorbar(np.array(range(1, 7)), xmin_fits1, yerr=xmin_std1,
                ls='none', color='black', elinewidth=0.5, capsize=2)

    ax10.plot(np.array(range(1, 7)), xmin_fits2, 'blue')
    ax10.scatter(np.array(range(1, 7)), xmin_fits2, marker='o',
                color='grey', edgecolor='black')
    ax10.errorbar(np.array(range(1, 7)), xmin_fits2, yerr=xmin_std2,
                ls='none', color='black', elinewidth=0.5, capsize=2)

    #ax11 3-3
    exp1 = exponents[2]
    exp2 = exponents[2]
    data1_x1 = pl.power_law(exp1, xmax1, sample_size1)
    data1_x2 = pl.power_law(exp2, xmax2, sample_size2)
    data2_x1 = pl.power_law(exp1, xmax2, sample_size1)
    data2_x2 = pl.power_law(exp2, xmax1, sample_size2)
    mixed1 = np.concatenate((data1_x1, data1_x2), axis=0)
    mixed2 = np.concatenate((data2_x1, data2_x2), axis=0)
    xmin_fits1 = np.zeros(len(xmin_tofit))
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

    ax11.axhline(y=exp1, color='black', linewidth=0.5)
    ax11.axhline(y=exp2, color='black', linewidth=0.5)
    ax11.plot(np.array(range(1, 7)), xmin_fits1, 'red')
    ax11.scatter(np.array(range(1, 7)), xmin_fits1, marker='o',
                color='red', edgecolor='black')
    ax11.errorbar(np.array(range(1, 7)), xmin_fits1, yerr=xmin_std1,
                ls='none', color='black', elinewidth=0.5, capsize=2)

    ax11.plot(np.array(range(1, 7)), xmin_fits2, 'blue')
    ax11.scatter(np.array(range(1, 7)), xmin_fits2, marker='o',
                color='grey', edgecolor='black')
    ax11.errorbar(np.array(range(1, 7)), xmin_fits2, yerr=xmin_std2,
                ls='none', color='black', elinewidth=0.5, capsize=2)

    #ax12 5-3
    exp1 = exponents[3]
    exp2 = exponents[2]
    data1_x1 = pl.power_law(exp1, xmax1, sample_size1)
    data1_x2 = pl.power_law(exp2, xmax2, sample_size2)
    data2_x1 = pl.power_law(exp1, xmax2, sample_size1)
    data2_x2 = pl.power_law(exp2, xmax1, sample_size2)
    mixed1 = np.concatenate((data1_x1, data1_x2), axis=0)
    mixed2 = np.concatenate((data2_x1, data2_x2), axis=0)
    xmin_fits1 = np.zeros(len(xmin_tofit))
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

    ax12.axhline(y=exp1, color='black', linewidth=0.5)
    ax12.axhline(y=exp2, color='black', linewidth=0.5)
    ax12.plot(np.array(range(1, 7)), xmin_fits1, 'red')
    ax12.scatter(np.array(range(1, 7)), xmin_fits1, marker='o',
                color='red', edgecolor='black')
    ax12.errorbar(np.array(range(1, 7)), xmin_fits1, yerr=xmin_std1,
                ls='none', color='black', elinewidth=0.5, capsize=2)

    ax12.plot(np.array(range(1, 7)), xmin_fits2, 'blue')
    ax12.scatter(np.array(range(1, 7)), xmin_fits2, marker='o',
                color='grey', edgecolor='black')
    ax12.errorbar(np.array(range(1, 7)), xmin_fits2, yerr=xmin_std2,
                ls='none', color='black', elinewidth=0.5, capsize=2)

    ax12.yaxis.set_label_position("right")
    ax12.set_ylabel('3.0', fontsize=15, rotation=270, labelpad=15)

    #ax13 1-5
    exp1 = exponents[0]
    exp2 = exponents[3]
    data1_x1 = pl.power_law(exp1, xmax1, sample_size1)
    data1_x2 = pl.power_law(exp2, xmax2, sample_size2)
    data2_x1 = pl.power_law(exp1, xmax2, sample_size1)
    data2_x2 = pl.power_law(exp2, xmax1, sample_size2)
    mixed1 = np.concatenate((data1_x1, data1_x2), axis=0)
    mixed2 = np.concatenate((data2_x1, data2_x2), axis=0)
    xmin_fits1 = np.zeros(len(xmin_tofit))
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

    ax13.axhline(y=exp1, color='black', linewidth=0.5)
    ax13.axhline(y=exp2, color='black', linewidth=0.5)
    ax13.plot(np.array(range(1, 7)), xmin_fits1, 'red')
    ax13.scatter(np.array(range(1, 7)), xmin_fits1, marker='o',
                color='red', edgecolor='black')
    ax13.errorbar(np.array(range(1, 7)), xmin_fits1, yerr=xmin_std1,
                ls='none', color='black', elinewidth=0.5, capsize=2)

    ax13.plot(np.array(range(1, 7)), xmin_fits2, 'blue')
    ax13.scatter(np.array(range(1, 7)), xmin_fits2, marker='o',
                color='grey', edgecolor='black')
    ax13.errorbar(np.array(range(1, 7)), xmin_fits2, yerr=xmin_std2,
                ls='none', color='black', elinewidth=0.5, capsize=2)

    #ax14 2-5
    exp1 = exponents[1]
    exp2 = exponents[3]
    data1_x1 = pl.power_law(exp1, xmax1, sample_size1)
    data1_x2 = pl.power_law(exp2, xmax2, sample_size2)
    data2_x1 = pl.power_law(exp1, xmax2, sample_size1)
    data2_x2 = pl.power_law(exp2, xmax1, sample_size2)
    mixed1 = np.concatenate((data1_x1, data1_x2), axis=0)
    mixed2 = np.concatenate((data2_x1, data2_x2), axis=0)
    xmin_fits1 = np.zeros(len(xmin_tofit))
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

    ax14.axhline(y=exp1, color='black', linewidth=0.5)
    ax14.axhline(y=exp2, color='black', linewidth=0.5)
    ax14.plot(np.array(range(1, 7)), xmin_fits1, 'red')
    ax14.scatter(np.array(range(1, 7)), xmin_fits1, marker='o',
                color='red', edgecolor='black')
    ax14.errorbar(np.array(range(1, 7)), xmin_fits1, yerr=xmin_std1,
                ls='none', color='black', elinewidth=0.5, capsize=2)

    ax14.plot(np.array(range(1, 7)), xmin_fits2, 'blue')
    ax14.scatter(np.array(range(1, 7)), xmin_fits2, marker='o',
                color='grey', edgecolor='black')
    ax14.errorbar(np.array(range(1, 7)), xmin_fits2, yerr=xmin_std2,
                ls='none', color='black', elinewidth=0.5, capsize=2)

    #ax15 3-5
    exp1 = exponents[2]
    exp2 = exponents[3]
    data1_x1 = pl.power_law(exp1, xmax1, sample_size1)
    data1_x2 = pl.power_law(exp2, xmax2, sample_size2)
    data2_x1 = pl.power_law(exp1, xmax2, sample_size1)
    data2_x2 = pl.power_law(exp2, xmax1, sample_size2)
    mixed1 = np.concatenate((data1_x1, data1_x2), axis=0)
    mixed2 = np.concatenate((data2_x1, data2_x2), axis=0)
    xmin_fits1 = np.zeros(len(xmin_tofit))
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

    ax15.axhline(y=exp1, color='black', linewidth=0.5)
    ax15.axhline(y=exp2, color='black', linewidth=0.5)
    ax15.plot(np.array(range(1, 7)), xmin_fits1, 'red')
    ax15.scatter(np.array(range(1, 7)), xmin_fits1, marker='o',
                color='red', edgecolor='black')
    ax15.errorbar(np.array(range(1, 7)), xmin_fits1, yerr=xmin_std1,
                ls='none', color='black', elinewidth=0.5, capsize=2)

    ax15.plot(np.array(range(1, 7)), xmin_fits2, 'blue')
    ax15.scatter(np.array(range(1, 7)), xmin_fits2, marker='o',
                color='grey', edgecolor='black')
    ax15.errorbar(np.array(range(1, 7)), xmin_fits2, yerr=xmin_std2,
                  ls='none', color='black', elinewidth=0.5, capsize=2)

    #ax16 5-5
    exp1 = exponents[3]
    exp2 = exponents[3]
    data1_x1 = pl.power_law(exp1, xmax1, sample_size1)
    data1_x2 = pl.power_law(exp2, xmax2, sample_size2)
    data2_x1 = pl.power_law(exp1, xmax2, sample_size1)
    data2_x2 = pl.power_law(exp2, xmax1, sample_size2)
    mixed1 = np.concatenate((data1_x1, data1_x2), axis=0)
    mixed2 = np.concatenate((data2_x1, data2_x2), axis=0)
    xmin_fits1 = np.zeros(len(xmin_tofit))
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

    ax16.axhline(y=exp1, color='black', linewidth=0.5)
    ax16.axhline(y=exp2, color='black', linewidth=0.5)
    ax16.plot(np.array(range(1, 7)), xmin_fits1, 'red')
    ax16.scatter(np.array(range(1, 7)), xmin_fits1, marker='o',
                color='red', edgecolor='black')
    ax16.errorbar(np.array(range(1, 7)), xmin_fits1, yerr=xmin_std1,
                ls='none', color='black', elinewidth=0.5, capsize=2)

    ax16.plot(np.array(range(1, 7)), xmin_fits2, 'blue')
    ax16.scatter(np.array(range(1, 7)), xmin_fits2, marker='o',
                color='grey', edgecolor='black')
    ax16.errorbar(np.array(range(1, 7)), xmin_fits2, yerr=xmin_std2,
                ls='none', color='black', elinewidth=0.5, capsize=2)

    ax16.yaxis.set_label_position("right")
    ax16.set_ylabel('4.0', fontsize=15, rotation=270, labelpad=15)


    matplotlib.rcParams.update({'font.size': 6})

    f.text(0.495, -0.05, 'Xmin', ha='center', fontsize=15)
    f.text(-0.03, 0.5, 'Fitted exponent', va='center',
        rotation='vertical', fontsize=15)

    ax1.set_ylim([0, 6])
    ax1.set_xlim([0.5, 6.5])
    plt.tight_layout()

    plt.savefig('mixed_n{}-{}_xmax{}.png'.format(int(sample_size1), int(sample_size2),int(xmax_r)))
    plt.savefig('mixed_n{}-{}_xmax{}.svg'.format(int(sample_size1), int(sample_size2),int(xmax_r)))
