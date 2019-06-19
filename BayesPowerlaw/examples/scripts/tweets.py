import BayesPowerlaw as bp
import numpy as np
import matplotlib.pyplot as plt
import os

data_dir = os.path.dirname(os.path.abspath(__file__)) + '/examples/data'
data = np.loadtxt(data_dir + '/tweet_count.txt')

fit=bp.bayes(data)

plt.figure(figsize=(6, 4))
fit.plot_fit(np.mean(fit.gamma_posterior[0]), fit_color='black', scatter_size=100,
                  data_color='gray', edge_color='black', line_width=2)
plt.ylim(10**-5, 10**0)
plt.xlabel('likes', fontsize=16)
plt.ylabel('frequency', fontsize=16)
plt.title('Likes per Tweet', fontsize=18)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
fit.plot_posterior(fit.gamma_posterior[0], range=[1.6, 1.9], color='blue')
plt.xlabel('exponent', fontsize=16)
plt.ylabel('posterior', fontsize=16)
plt.title('Posterior for Likes per Tweet', fontsize=18)
plt.tight_layout()
plt.show()
