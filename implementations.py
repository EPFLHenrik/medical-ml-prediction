from utils.all_functions import least_squares_method, ridge_regression_method, gradient_descent, stochastic_gradient_descent

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    return gradient_descent(y, tx, initial_w, max_iters, gamma)

def mean_squared_error_sgd(y, tx, initial_w,max_iters, gamma):
    return stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma)

def least_squares(y, tx):
    return least_squares_method(y, tx)

def ridge_regression(y, tx, lambda_):
    return ridge_regression_method(y, tx, lambda_)