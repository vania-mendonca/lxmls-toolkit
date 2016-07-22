import numpy as np
import matplotlib.pyplot as plt


def get_y(x):
    return (x + 2) ** 2 - 16 * np.exp(-((x - 2) ** 2))


def get_grad(x):
    return (2 * x + 4) - 16 * (-2 * x + 4) * np.exp(-((x - 2) ** 2))


# x = np.arange(-8, 8, 0.001)
# y = map(lambda u: get_y(u), x)
# plt.plot(x, y)


# plt.show()


def grad_desc(start_x, eps, prec):
    """
    runs the gradient descent algorithm and returns the list of estimates
    example of use grad_desc(start_x=3.9, eps=0.01, prec=0.00001)
    """
    x_new = start_x
    x_old = start_x + prec * 2
    res = [x_new]
    while abs(x_old - x_new) > prec:
        x_old = x_new
        x_new = x_old - eps * get_grad(x_new)
        res.append(x_new)
    return np.array(res)


result = grad_desc(-4, 0.01, 0.0001)

dim1 = np.shape(result)
x = xrange(np.int(dim1[0]))

plt.plot(x, result, '+')

result = grad_desc(4, 0.01, 0.00001)

dim1 = np.shape(result)
x = xrange(np.int(dim1[0]))

plt.plot(x, result, '+')

plt.show()
