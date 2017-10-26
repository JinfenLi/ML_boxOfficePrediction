import numpy as np
import pandas as pd
import datastandard2 as ds

def load_data():

    traindata = pd.DataFrame(pd.read_csv('traindata.csv'))
    validata = pd.DataFrame(pd.read_csv('testdata.csv'))
    traindata = np.array(traindata)
    validata = np.array(validata)
    training_data, validation_data, test_data = (traindata,validata,[])
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.

    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""



    # tr_d, va_d, te_d = load_data()
    # training_inputs = wrapdata(tr_d)
    # training_results = tr_d.T[0]
    # training_data = list(zip(training_inputs, training_results))
    # validation_inputs = wrapdata(va_d)
    # validation_data = list(zip(validation_inputs, va_d.T[0]))
    # test_inputs = wrapdata(te_d)
    # test_data = list(zip(test_inputs, te_d.T[0]))
    # print(training_inputs.shape)
    # print(training_data)

    tr_d, va_d, te_d = load_data()
    training_data = wrapdata(tr_d)
    validation_data = wrapdata(va_d)
    # test_data = wrapdata(te_d)
    return (training_data, validation_data, [])


def wrapdata(x):
    x = x.T
    matrix = np.vstack(
        (x[0],x[1],x[4],x[5],x[6],x[7],x[8],x[9],x[10],x[11],x[12],x[13],x[14],x[15]))
    matrix = matrix.T
    return matrix