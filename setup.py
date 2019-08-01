from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='BayesPowerlaw',
      version='1.1.1',
      description='Fitting power law distributions using Bayesian Inference',
      long_description=readme(),
      classifiers=[
          'Development Status :: 4 - Beta',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.7',
          'Topic :: Scientific/Engineering :: Mathematics',
      ],
      keywords='power law, bayesian inference',
      url='https://github.com/AtwalLab/BayesPowerlaw',
      author='Kristina Grigaityte, Gurinder Atwal',
      author_email='atwal@cshl.edu',
      license='MIT',
      packages=['BayesPowerlaw'],
      package_data={'BayesPowerlaw': ['BayesPowerlaw_data/*']},
      include_package_data=True,
      install_requires=[
          'scipy',
          'numpy',
          'matplotlib',
      ],
      zip_safe=False)
