from src.bayes import bayes
from src.maxLikelihood import maxLikelihood
from src.power_law import power_law

def demo():
    """
    Performs a demonstration of suftware.
    Parameters
    ----------
    
    None.

    Return
    ------

    None.
    """

    import os
    example_dir = os.path.dirname(__file__)
    example = 'examples/scripts/tweets.py'
    file_name = '%s/%s' % (example_dir, example)
    with open(file_name, 'r') as f:
        content = f.read()
        line = '-------------------------------------------------------------'
        print('Running %s:\n%s\n%s\n%s' %
            (file_name, line, content, line))
    exec(open(file_name).read())


