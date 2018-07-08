from setuptools import setup

setup(
	name='potoroo',
	version='0.1',
	description = 'For fitting discrete and continuous power law distribution using Bayesian or Maximum Likelihood approach',
	authors=['Kristina Grigatyte', 'Mickey Atwal'],
	scripts=['powerlaw.py'],
	requires=['scipy','numpy','matplotlib']
	)