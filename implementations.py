from useful_functions.gradient_descent import gradient_descent
from useful_functions.stochastic_gradient_descent import stochastic_gradient_descent

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    return gradient_descent(y, tx, initial_w, max_iters, gamma)

def mean_squared_error_sgd(y, tx, initial_w,max_iters, gamma):
    return stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma)