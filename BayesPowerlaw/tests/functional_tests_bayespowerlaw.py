import sys
sys.path.append('../../')

import BayesPowerlaw as bp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


global_success_counter = 0
global_fail_counter = 0

# Common success and fail lists
bool_fail_list = [0, -1, 'True', 'x', 1]

def test_for_mistake(func, *args, **kw):
    """
    Run a function with the specified parameters and register whether
    success or failure was a mistake
    parameters
    ----------
    func: (function or class constructor)
        An executable function to which *args and **kwargs are passed.
    return
    ------
    None.
    """

    global global_fail_counter
    global global_success_counter

    # print test number
    test_num = global_fail_counter + global_success_counter
    print('Test # %d: ' % test_num, end='')
    #print('Test # %d: ' % test_num)

    # Run function
    obj = func(*args, **kw)
    # Increment appropriate counter
    if obj.mistake:
        global_fail_counter += 1
    else:
        global_success_counter += 1

def test_parameter_values(func,
                          var_name=None,
                          fail_list=[],
                          success_list=[],
                          **kwargs):
    """
    Tests predictable success & failure of different values for a
    specified parameter when passed to a specified function
    parameters
    ----------
    func: (function)
        Executable to test. Can be function or class constructor.
    var_name: (str)
        Name of variable to test. If not specified, function is
        tested for success in the absence of any passed parameters.
    fail_list: (list)
        List of values for specified variable that should fail
    success_list: (list)
        List of values for specified variable that should succeed
    **kwargs:
        Other keyword variables to pass onto func.
    return
    ------
    None.
    """

    # If variable name is specified, test each value in fail_list
    # and success_list
    if var_name is not None:

        # User feedback
        print("Testing %s() parameter %s ..." % (func.__name__, var_name))

        # Test parameter values that should fail
        for x in fail_list:
            kwargs[var_name] = x
            test_for_mistake(func=func, should_fail=True, **kwargs)

        # Test parameter values that should succeed
        for x in success_list:
            kwargs[var_name] = x
            test_for_mistake(func=func, should_fail=False, **kwargs)

        print("Tests passed: %d. Tests failed: %d.\n" %
              (global_success_counter, global_fail_counter))

    # Otherwise, make sure function without parameters succeeds
    else:

        # User feedback
        print("Testing %s() without parameters." % func.__name__)

        # Test function
        test_for_mistake(func=func, should_fail=False, **kwargs)

    # close all figures that might have been generated
    plt.close('all')    

def test_bayes():

    # inputs that successfully execute when entered into Bayes.
    powerlaw_array = bp.power_law([1.5],[1.0],xmax=1000, sample_size=1000)
    random_array = (np.random.rand(50)*10).astype(int)+1

    # df inputs that fail when entered into Bayes.
    bad_array1 = 'x'
    bad_array2 = ['x',1,5,8,9]
    zero_array = [1,0,5,0,3,0]
    empty = []

    # test parameter df
    test_parameter_values(func=bp.bayes, var_name='data',fail_list=[bool_fail_list,bad_array1,bad_array2,zero_array,empty],
                            success_list=[powerlaw_array,random_array])

    test_parameter_values(func=bp.bayes, var_name='gamma_range',fail_list=[bool_fail_list,bad_array1,empty,[True,False],[0,0],[0.1,7]],
                            success_list=[[1.01,5],[2,7],[4,9]],data=powerlaw_array)

    test_parameter_values(func=bp.bayes, var_name='xmin',fail_list=[bool_fail_list,bad_array1,empty,True,0,1000000],
                            success_list=[1,1.5,None],data=powerlaw_array)

    test_parameter_values(func=bp.bayes, var_name='xmax',fail_list=[bool_fail_list,bad_array1,empty,True,0,1],
                            success_list=[10,1000000000.0,None,np.infty],data=powerlaw_array)

    test_parameter_values(func=bp.bayes, var_name='discrete',fail_list=[bool_fail_list,bad_array1,empty,0,1,2],
                            success_list=[True,False],data=powerlaw_array)

    test_parameter_values(func=bp.bayes, var_name='niters',fail_list=[bool_fail_list,bad_array1,empty,True,-1,150.5,99],
                            success_list=[1000,100],data=powerlaw_array)

    test_parameter_values(func=bp.bayes, var_name='sigma',fail_list=[bool_fail_list,bad_array1,empty,[True,False],[-1,-3],[0,0],0.5,[True,2],[True,True]],
                            success_list=[[0.05, 0.05],[1,1]],data=powerlaw_array)

    test_parameter_values(func=bp.bayes, var_name='sigma_burn',fail_list=[bool_fail_list,bad_array1,empty,[True,False],[-1,-3],[0,0],0.5,[2,True],[True,True]],
                            success_list=[[0.05, 0.05],[1,1]],data=powerlaw_array)

    test_parameter_values(func=bp.bayes, var_name='burn_in',fail_list=[bool_fail_list,bad_array1,empty,True,-1,150.5,99],
                            success_list=[1000,100],data=powerlaw_array)

    test_parameter_values(func=bp.bayes, var_name='prior',fail_list=[bool_fail_list,bad_array1,empty,True,'exponential',0],
                            success_list=['jeffrey','flat'],data=powerlaw_array)

    test_parameter_values(func=bp.bayes, var_name='mixed',fail_list=[bool_fail_list,bad_array1,empty,True,'1',0],
                            success_list=[1,4],data=powerlaw_array)

    test_parameter_values(func=bp.bayes, var_name='fit',fail_list=[bool_fail_list,bad_array1,empty,0,1,2],
                            success_list=[True,False],data=powerlaw_array)

    
def test_bayes_plot_fit():

    gamma=1.5

    powerlaw_array = bp.power_law([1.5],[1.0],xmax=1000, sample_size=1000)

    test_parameter_values(func=bp.bayes(powerlaw_array).plot_fit, var_name='gamma_mean',fail_list=[1.0,-1,True,'x',[],[2,4]],
                            success_list=[gamma,2])

    test_parameter_values(func=bp.bayes(powerlaw_array).plot_fit, var_name='data_label',fail_list=[1.0,-1,True,[],[2,4]],
                            success_list=['powerlaw','1',None],gamma_mean=gamma)

    test_parameter_values(func=bp.bayes(powerlaw_array).plot_fit, var_name='scatter_size',fail_list=[-1,0,True,[],[2,4],'x'],
                            success_list=[2,10.5],gamma_mean=gamma)

    test_parameter_values(func=bp.bayes(powerlaw_array).plot_fit, var_name='line_width',fail_list=[-1,0,True,[],[2,4],'x'],
                            success_list=[2,10.5],gamma_mean=gamma)

    test_parameter_values(func=bp.bayes(powerlaw_array).plot_fit, var_name='fit',fail_list=[-1,0,[],[2,4],'x'],
                            success_list=[True, False],gamma_mean=gamma)
            
    test_parameter_values(func=bp.bayes(powerlaw_array).plot_fit, var_name='log',fail_list=[-1,0,[],[2,4],'x'],
                            success_list=[True, False],gamma_mean=gamma)

    test_parameter_values(func=bp.bayes(powerlaw_array).plot_fit, var_name='xmin',fail_list=[0.99,-1,True,[],[2,4],'x'],
                            success_list=[1,3,None],gamma_mean=gamma)


def test_bayes_plot_posterior():

    powerlaw_array = bp.power_law([1.5],[1.0],xmax=1000, sample_size=1000)

    fit1=bp.bayes(powerlaw_array)
    fit2=bp.bayes(powerlaw_array,mixed=2)

    test_parameter_values(func=bp.bayes(powerlaw_array).plot_posterior, var_name='samples',fail_list=[fit1.gamma_posterior,fit2.gamma_posterior, None, fit2.weight_posterior, fit1, True, [], 'x'],
                            success_list=[fit1.gamma_posterior[0],fit2.gamma_posterior[1],fit2.weight_posterior[0]])

def test_maxLikelihood():

    # inputs that successfully execute when entered into Bayes.
    powerlaw_array = bp.power_law([1.5],[1.0],xmax=1000, sample_size=1000)
    random_array = (np.random.rand(50)*10).astype(int)+1

    # df inputs that fail when entered into Bayes.
    bad_array1 = 'x'
    bad_array2 = ['x',1,5,8,9]
    zero_array = [1,0,5,0,3,0]
    empty = []

    # test parameter df
    test_parameter_values(func=bp.maxLikelihood, var_name='data',fail_list=[bool_fail_list,bad_array1,bad_array2,zero_array,empty],
                            success_list=[powerlaw_array,random_array])

    test_parameter_values(func=bp.maxLikelihood, var_name='initial_guess',fail_list=[bad_array1,bad_array2,zero_array,empty,[0,3,5],[2,5,4.4],[2,6],[1,'x',5],[4,2,3]],
                            success_list=[[1,5,10],[1.2,6.4,3]],data=powerlaw_array)

    test_parameter_values(func=bp.maxLikelihood, var_name='discrete',fail_list=[bool_fail_list,bad_array1,empty,0,1,2],
                            success_list=[True,False],data=powerlaw_array)

    test_parameter_values(func=bp.maxLikelihood, var_name='xmin',fail_list=[bool_fail_list,bad_array1,empty,True,0,1000000],
                            success_list=[1,1.5,None],data=powerlaw_array)

    test_parameter_values(func=bp.maxLikelihood, var_name='xmax',fail_list=[bool_fail_list,bad_array1,empty,True,0,1],
                            success_list=[10,1000000000.0,None,np.infty],data=powerlaw_array)



def run_tests():
    """
    Run all Logomaker functional tests. There is 1 test as of 3 July 2019.
    parameters
    ----------
    None.
    return
    ------
    None.
    """

    test_bayes()
    test_bayes_plot_fit()
    test_bayes_plot_posterior()
    test_maxLikelihood()

if __name__ == '__main__':
    run_tests()