========
Tutorial
========

We begin by loading numpy, matplotlib.pyplot and BayesPowerlaw::

    import numpy as np
    import matplotlib.pyplot as plt
    import BayesPowerlaw as bp

Next we simulate a power law distribution using the power_law function::

    #define variables for simulation
    exponent=2.0
    sample_size=1000
    xmax=10000
    #simulate data following power law distribution
    data = bp.power_law([exponent],[1.0],xmax,sample_size)


We can simply plot the data by running the bayes function while specifying that we don't want to perform the fitting::

    #create a bayes-data object without fitting
    bayes_object = bp.bayes(data, fit=False)
    #plot the distribution without the fit

    plt.figure(figsize=(6,4))
    bayes_object.plot_fit(1.01,scatter_size=100,data_color='gray',edge_color='black',fit=False)
    plt.xlabel('x', fontsize=16)
    plt.ylabel('frequency',fontsize=16)

.. image:: sim_powerlaw.png
   :alt: Simulated power law.

Let's perform the fitting of the simulated power law and get an exponent, this will take up to 30s::

    #perform the fitting
    fit=bp.bayes(data)
    #get the posterior of exponent attribute. Since we only have a singular power law, we need only the first (index = 0) row of the 2D array.
    posterior=fit.gamma_posterior[0]
    #mean of the posterior is our estimated exponent
    est_exponent=np.mean(posterior)

Now let's plot the distribution with the fit::

    plt.figure(figsize=(6,4))
    fit.plot_fit(est_exponent,fit_color='black',scatter_size=100,data_color='gray',edge_color='black',line_width=2)
    plt.xlabel('x', fontsize=16)
    plt.ylabel('frequency',fontsize=16)

.. image:: sim_fit.png
   :alt: Simulated power law with fit.

We can also plot the posterior distribution::

    plt.figure()
    fit.plot_posterior(posterior,color='blue')
    plt.xlabel('posterior', fontsize=16)
    plt.ylabel('exponent',fontsize=16)

.. image:: sim_posterior.png
   :alt: Simulated power law.

