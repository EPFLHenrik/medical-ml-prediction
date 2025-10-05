from useful_functions.gradient_descent import gradient_descent
from useful_functions.stochastic_gradient_descent import stochastic_gradient_descent
from useful_functions.least_squares import least_squares_method
from useful_functions.ridge_regression import ridge_regression_method

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    return gradient_descent(y, tx, initial_w, max_iters, gamma)

def mean_squared_error_sgd(y, tx, initial_w,max_iters, gamma):
    return stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma)

def least_squares(y, tx):
    return least_squares_method(y, tx)

def ridge_regression(y, tx, lambda_):
    return ridge_regression_method(y, tx, lambda_)