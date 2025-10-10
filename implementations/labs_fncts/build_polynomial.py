import numpy as np
import matplotlib.pyplot as plt


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree.

    Args:
        x: numpy array of shape (N,), N is the number of samples.
        degree: integer.

    Returns:
        poly: numpy array of shape (N,d+1)

    >>> build_poly(np.array([0.0, 1.5]), 2)
    array([[1.  , 0.  , 0.  ],
           [1.  , 1.5 , 2.25]])
    """
    N = x.shape[0]
    poly = np.zeros((N, degree + 1))
    poly[:, 0] = np.ones((N))
    for ii in range(1, degree + 1):
        poly[:, ii] = x**ii

    return poly
