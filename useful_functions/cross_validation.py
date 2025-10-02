from costs import compute_loss
from build_polynomial import build_poly
from ridge_regression import ridge_regression_method
from build_k_indices import build_k_indices
import numpy as np

def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression for a fold corresponding to k_indices

    Args:
        y:          shape=(N,)
        x:          shape=(N,)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:    scalar, cf. ridge_regression()
        degree:     scalar, cf. build_poly()

    Returns:
        train and test root mean square errors rmse = sqrt(2 mse)

    >>> cross_validation(np.array([1.,2.,3.,4.]), np.array([6.,7.,8.,9.]), np.array([[3,2], [0,1]]), 1, 2, 3)
    (0.019866645527597114, 0.33555914361295175)
    """

    # ***************************************************
    # INSERT YOUR CODE HERE
    # get k'th subgroup in test, others in train: TODO
    # ***************************************************
    ind = k_indices[k]
    x_test = x[ind]
    y_test = y[ind]
    mask = np.ones(len(x), dtype=bool)  # start with all True
    mask[ind] = False
    x_train = x[mask]
    y_train = y[mask]
    # ***************************************************
    # INSERT YOUR CODE HERE
    # form data with polynomial degree: TODO
    # ***************************************************
    x_test_poly = build_poly(x_test, degree)
    x_train_poly = build_poly(x_train, degree)
    # ***************************************************
    # INSERT YOUR CODE HERE
    # ridge regression: TODO
    # ***************************************************
    weights = ridge_regression_method(y_train, x_train_poly, lambda_)
    # ***************************************************
    # INSERT YOUR CODE HERE
    # calculate the loss for train and test data: TODO
    # ***************************************************
    loss_tr = np.sqrt(2*compute_loss(y_train, x_train_poly, weights))
    loss_te = np.sqrt(2*compute_loss(y_test, x_test_poly, weights))
    return loss_tr, loss_te

from plots import cross_validation_visualization

def cross_validation_demo(y, x, degree, k_fold, lambdas):
    """cross validation over regularisation parameter lambda.

    Args:
        degree: integer, degree of the polynomial expansion
        k_fold: integer, the number of folds
        lambdas: shape = (p, ) where p is the number of values of lambda to test
    Returns:
        best_lambda : scalar, value of the best lambda
        best_rmse : scalar, the associated root mean squared error for the best lambda
    """

    seed = 12
    degree = degree
    k_fold = k_fold
    lambdas = lambdas
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    best_lambda = 0
    best_rmse = 999999
    # ***************************************************
    # INSERT YOUR CODE HERE
    # cross validation over lambdas: TODO
    # ***************************************************
    for la in lambdas:
        rmse_tr_lambda = []
        rmse_te_lambda = []
        for k in range(k_fold):
            tr, te = cross_validation(y, x, k_indices, k, la, degree)
            rmse_te_lambda.append(te)
            rmse_tr_lambda.append(tr)
            
        avg_te = sum(rmse_te_lambda)/len(rmse_te_lambda)
        avg_tr = sum(rmse_tr_lambda)/len(rmse_tr_lambda)
        rmse_tr.append(avg_tr)
        rmse_te.append(avg_te)
        if avg_te < best_rmse:
            best_rmse = avg_te
            best_lambda = la

    cross_validation_visualization(lambdas, rmse_tr, rmse_te)
    print(
        "For polynomial expansion up to degree %.f, the choice of lambda which leads to the best test rmse is %.5f with a test rmse of %.3f"
        % (degree, best_lambda, best_rmse)
    )
    return best_lambda, best_rmse


best_lambda, best_rmse = cross_validation_demo(7, 4, np.logspace(-4, 0, 30))