# ##############################################################################
# Linear solver
# ##############################################################################

import numpy as np
import matplotlib.pyplot as plt
import lxmls.readers.galton as galton

# ##############################################################################


def error_function(x, y, w):
    '''

    Parameters
    ----------
    x Examples
    y (Correct) Output Labels
    w Parameter of linear regression

    Returns The error of the solver when the parameter is w
    -------

    '''
    errors = (np.dot(x, w) - y) ** 2
    error = np.sum(errors)
    return error


# ##############################################################################

def error_gradient(x, y, w):
    '''

    Parameters
    ----------
    x Examples
    y (Correct) Output Labels
    w Parameter of linear regression

    Returns The gradient of error_function wrt to w
    -------

    '''
    gradient = np.empty_like(w)
    gradient[0] = np.sum(np.dot(x, w) - y)
    gradient[1] = np.sum(np.dot((np.dot(x, w) - y), x))
    return gradient

# ##############################################################################


def gradient_descent(x, y, error_func, grad_func):
    '''

    Parameters
    ----------
    x Examples
    y (Correct) Output Labels
    error_func Error function
    grad_func Gradient of the Error function

    Returns The evolution of the error and the parameters
    -------

    '''

    # Precision of the solution
    epsilon = 0.0001

    # Use a fixed small step size
    step_size = 1.0e-10

    # Max iterations
    max_iter = 100

    # Initial value of w
    w = np.random.randn(2, 1)

    # Bias unit added to x
    dim = len(x)
    X = np.empty([dim, 2])
    X[:, 0] = np.ones_like(x)
    X[:, 1] = x

    error_list = []
    w_list = []

    for i in xrange(max_iter):

        error = error_func(X, y, w)
        error_list.append(error)

        old_w = np.empty_like(w)
        old_w[0] = w[0]
        old_w[1] = w[1]

        w_list.append(old_w)

        grad = grad_func(X, y, w)
        w[0] -= step_size * grad[0]
        w[1] -= step_size * grad[1]

        if (np.abs(w - old_w) < epsilon).all():
            print "change in function values too small, leaving"
            return np.array(error_list), np.array(w_list)

    print "exceeded maximum number of iterations, leaving"
    return np.array(error_list), np.array(w_list)


# ##############################################################################


galton_data = galton.load()

# Test
# x_test = np.array([[1, 2], [3, 4], [5, 6]])
# y_test = np.array([[0], [1], [0]])
# w_test = np.array([[2], [4]])
#
# dim1, dim2 = np.shape(galton_data)
#
# x_test = np.empty([dim1, 2])
#
# x_test[:, 0] = np.ones_like(galton_data[:, 0])
# x_test[:, 1] = galton_data[:, 0]
# y_test = galton_data[:, 1]
#
# print error_function(x_test, y_test, w_test)
# print error_gradient(x_test, y_test, w_test)

errors, ws = gradient_descent(galton_data[:, 0], galton_data[:, 1], error_function, error_gradient)

plt.plot(errors, '-')
# plt.plot(ws[:, 0], ws[:, 1], '+')

# plt.ylim(0, np.max(errors))
plt.show()
