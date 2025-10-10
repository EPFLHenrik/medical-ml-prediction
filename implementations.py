from utils.all_functions import least_squares_method, ridge_regression_method, gradient_descent, stochastic_gradient_descent

import numpy as np
import matplotlib.pyplot as plt

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma, loss_type = "MSE"):
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = [compute_loss(y,tx, initial_w, loss_type)] # compute loss for initial_w as well
    w = initial_w.copy()

    for n_iter in range(max_iters):
        # compute gradient
        grad_L = compute_gradient(y,tx, w)

        # update weights
        w = w - gamma* grad_L

        # compute loss with new weights
        loss = compute_loss(y,tx, w, loss_type)

        # store w and loss
        ws.append(w)
        losses.append(loss)
        print(
            "GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
            )
        )

    return w, loss

def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """
    # Normal equations
    a = tx.T @ tx     # Gram matrix
    b = tx.T @ y      # right-hand side

    # Solve the system a w = b
    w = np.linalg.solve(a, b)

    # Compute error and MSE
    e = y - tx @ w
    loss = (e @ e) / (2 * len(y))

    return w, loss


def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    N = y.shape[0]
    e = y - tx @ w
    grad_L = - (1/N) * tx.T @ e
    
    return grad_L

def compute_loss(y, tx, w, loss_type = "MSE"):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2,). The vector of model parameters.
        loss_type (optional) : string, 'MSE' or 'MAE', denoting the type of loss we compute 

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    e = y - tx @ w
    N = len(y)
        
    MAE = np.mean(np.abs(e))
    MSE = 1 / (2*N) * (e.T @ e) 
    
    if loss_type == "MSE":
        return MSE
    elif loss_type == "MAE":
        return MAE
    else:
        raise ValueError("loss_type must be 'MSE' or 'MAE'")
    

def mean_squared_error_sgd(y, tx, initial_w, batch_size, max_iters, gamma, loss_type = "MSE"):


    # Define parameters to store w and loss
    ws = [initial_w]
    losses = [compute_loss(y, tx, initial_w, loss_type)]  # include initial loss
    w = initial_w.copy()

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y,tx,batch_size, num_batches=1, shuffle=True) : 
 
            # compute stochastic gradient on bathc 
            g =  compute_stoch_gradient(y_batch, tx_batch, w)

            # update weights 
            w = w - gamma * g

            # compute loss on full batch with current set of weights
            loss = compute_loss(y,tx, w, loss_type)

            losses.append(loss)
            ws.append(w)

        print(
            "SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
            )
        )
    return w, loss

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient at w from a data sample batch of size B, where B < N, and their corresponding labels.

    Args:
        y: numpy array of shape=(B, )
        tx: numpy array of shape=(B,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        A numpy array of shape (2, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """
    # same as before, gradient computation, just on minibatch of size b = len(y)
    e = y - tx @ w
    N = y.shape[0]
    grad_L = -1/N * tx.T @ e
    return grad_L

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.

    Example:

     Number of batches = 9

     Batch size = 7                              Remainder = 3
     v     v                                         v v
    |-------|-------|-------|-------|-------|-------|---|
        0       7       14      21      28      35   max batches = 6

    If shuffle is False, the returned batches are the ones started from the indexes:
    0, 7, 14, 21, 28, 35, 0, 7, 14

    If shuffle is True, the returned batches start in:
    7, 28, 14, 35, 14, 0, 21, 28, 7

    To prevent the remainder datapoints from ever being taken into account, each of the shuffled indexes is added a random amount
    8, 28, 16, 38, 14, 0, 22, 28, 9

    This way batches might overlap, but the returned batches are slightly more representative.

    Disclaimer: To keep this function simple, individual datapoints are not shuffled. For a more random result consider using a batch_size of 1.

    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)  # NUmber of data points.
    batch_size = min(data_size, batch_size)  # Limit the possible size of the batch.
    max_batches = int(
        data_size / batch_size
    )  # The maximum amount of non-overlapping batches that can be extracted from the data.
    remainder = (
        data_size - max_batches * batch_size
    )  # Points that would be excluded if no overlap is allowed.

    if shuffle:
        # Generate an array of indexes indicating the start of each batch
        idxs = np.random.randint(max_batches, size=num_batches) * batch_size
        if remainder != 0:
            # Add an random offset to the start of each batch to eventually consider the remainder points
            idxs += np.random.randint(remainder + 1, size=num_batches)
    else:
        # If no shuffle is done, the array of indexes is circular.
        idxs = np.array([i % max_batches for i in range(num_batches)]) * batch_size

    for start in idxs:
        start_index = start  # The first data point of the batch
        end_index = (
            start_index + batch_size
        )  # The first data point of the following batch
        yield y[start_index:end_index], tx[start_index:end_index]



def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.

    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 0)
    array([ 0.21212121, -0.12121212])
    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 1)
    array([0.03947092, 0.00319628])
    """
    N, D = tx.shape
    # closed form
    a = tx.T @ tx + 2 * N * lambda_ * np.identity(D)
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    e = y - tx @ w
    loss = (e @ e) / (2 * len(y)) + lambda_ * (w @ w)
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = [calculate_loss(y,tx, initial_w)] # compute loss for initial_w as well
    w = initial_w.copy()

    for n_iter in range(max_iters):
        # compute gradient
        grad_L = calculate_gradient(y,tx, w)

        # update weights
        w = w - gamma* grad_L

        # compute loss with new weights
        loss = calculate_loss(y,tx, w)

        # store w and loss
        ws.append(w)
        losses.append(loss)
        print(
            "GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
            )
        )

    return w, loss

def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array

    >>> sigmoid(np.array([0.1]))
    array([0.52497919])
    >>> sigmoid(np.array([0.1, 0.1]))
    array([0.52497919, 0.52497919])
    """
    sig = (1 + np.exp(-t)) ** (-1)
    return sig


def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a non-negative loss

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(4).reshape(2, 2)
    >>> w = np.c_[[2., 3.]]
    >>> round(calculate_loss(y, tx, w), 8)
    1.52429481
    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    n = y.shape[0]
    pred = sigmoid(tx @ w)
    
    # optional clip for numerical stability
    # eps = 1e-15
    # pred = np.clip(pred, eps, 1 - eps)

    L = - (1 / n) * np.sum(y * np.log(pred) + (1 - y) * np.log(1 - pred))
    return float(L)


def calculate_gradient(y, tx, w):
    """compute the gradient of loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a vector of shape (D, 1)

    >>> np.set_printoptions(8)
    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> calculate_gradient(y, tx, w)
    array([[-0.10370763],
           [ 0.2067104 ],
           [ 0.51712843]])
    """
    grad = tx.T @ (sigmoid(tx @ w) - y) / y.shape[0]
    return grad



def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
     # Define parameters to store w and loss
    ws = [initial_w]
    losses = [calculate_loss(y,tx, initial_w)] # compute loss for initial_w as well
    w = initial_w.copy()

    for n_iter in range(max_iters):
        # compute gradient
        loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
        w = w - gradient*gamma
        loss =  calculate_loss(y,tx,w)

        # store w and loss
        ws.append(w)
        losses.append(loss)
        print(
            "GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
            )
        )

    return w, loss


def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss and gradient.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        lambda_: scalar

    Returns:
        loss: scalar number
        gradient: shape=(D, 1)

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> lambda_ = 0.1
    >>> loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    >>> round(loss, 8)
    0.62137268
    >>> gradient
    array([[-0.08370763],
           [ 0.2467104 ],
           [ 0.57712843]])
    """
    loss = calculate_loss(y, tx, w) 
    gradient = tx.T @ (sigmoid(tx @ w) - y) / y.shape[0] + 2*lambda_*w
    #hessian = calculate_hessian(y, tx, w)
    return float(loss), gradient