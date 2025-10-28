import numpy as np
import matplotlib.pyplot as plt


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    Gradient Descent for MSE loss.

    Args:
        y (np.ndarray): Target values, shape (N,)
        tx (np.ndarray): Feature matrix, shape (N,D)
        initial_w (np.ndarray): Initial weights, shape (D,)
        max_iters (int): Number of iterations
        gamma (float): Step size

    Returns:
        w (np.ndarray): Final weights after gradient descent, shape (D,)
        loss (np.ndarray): Final loss as 0-dim array
    """
    w = initial_w.copy()
    for n_iters in range(max_iters):
        grad = compute_gradient(y, tx, w)
        w -= gamma * grad
    loss = compute_loss(y, tx, w)
    return w, np.array(loss)


def least_squares(y, tx):
    """
    Closed-form solution for least squares.

    Args:
        y (np.ndarray): Target values, shape (N,)
        tx (np.ndarray): Feature matrix, shape (N,D)

    Returns:
        w (np.ndarray): Optimal weights, shape (D,)
        loss (np.ndarray): MSE loss as 0-dim array
    """
    a = tx.T @ tx
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    e = y - tx @ w
    loss = (e @ e) / (2 * len(y))
    return w, np.array(loss)


def compute_gradient(y, tx, w):
    """
    Compute gradient of MSE loss.

    Args:
        y (np.ndarray): Target values, shape (N,)
        tx (np.ndarray): Feature matrix, shape (N,D)
        w (np.ndarray): Weights, shape (D,)

    Returns:
        grad (np.ndarray): Gradient, shape (D,)
    """
    N = y.shape[0]
    e = y - tx @ w
    grad = -(tx.T @ e) / N
    return grad


def compute_loss(y, tx, w):
    """
    Compute MSE loss.

    Args:
        y (np.ndarray): Target values, shape (N,)
        tx (np.ndarray): Feature matrix, shape (N,D)
        w (np.ndarray): Weights, shape (D,)

    Returns:
        loss (np.ndarray): Loss as 0-dim array
    """
    e = y - tx @ w
    N = len(y)
    loss = (e.T @ e) / (2 * N)
    return np.array(loss)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """
    Stochastic Gradient Descent for MSE loss.

    Args:
        y (np.ndarray): Target values, shape (N,)
        tx (np.ndarray): Feature matrix, shape (N,D)
        initial_w (np.ndarray): Initial weights, shape (D,)
        max_iters (int): Number of iterations
        gamma (float): Step size

    Returns:
        w (np.ndarray): Final weights, shape (D,)
        loss (np.ndarray): Final loss as 0-dim array
    """
    w = initial_w.copy()
    for n_iters in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1, shuffle=True):
            grad = compute_stoch_gradient(y_batch, tx_batch, w)
            w -= gamma * grad
    loss = compute_loss(y, tx, w)
    return w, np.array(loss)


def compute_stoch_gradient(y, tx, w):
    """
    Compute stochastic gradient for MSE loss.

    Args:
        y (np.ndarray): Target batch, shape (B,)
        tx (np.ndarray): Feature batch, shape (B,D)
        w (np.ndarray): Weights, shape (D,)

    Returns:
        grad (np.ndarray): Gradient, shape (D,)
    """
    N = y.shape[0]
    e = y - tx @ w
    grad = -(tx.T @ e) / N
    return grad


def batch_iter(y, tx, batch_size=1, num_batches=1, shuffle=True):
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
    """
    Closed-form ridge regression.

    Args:
        y (np.ndarray): Target values, shape (N,)
        tx (np.ndarray): Feature matrix, shape (N,D)
        lambda_ (float): Regularization strength

    Returns:
        w (np.ndarray): Optimal weights, shape (D,)
        loss (np.ndarray): MSE loss as 0-dim array
    """
    N, D = tx.shape
    a = tx.T @ tx + 2 * N * lambda_ * np.identity(D)
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    e = y - tx @ w
    loss = (e @ e) / (2 * N)
    return w, np.array(loss)

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent.

    Args:
        y (np.ndarray): Target vector, shape (N,)
        tx (np.ndarray): Feature matrix, shape (N,D)
        initial_w (np.ndarray): Initial weights, shape (D,)
        max_iters (int): Number of iterations
        gamma (float): Learning rate

    Returns:
        w (np.ndarray): Last weights, shape (D,)
        loss (np.ndarray): Last loss, 0-dim array
    """
    w = initial_w.copy()
    for n_iters in range(max_iters):
        grad = calculate_gradient(y, tx, w)
        w -= gamma * grad
    loss = calculate_loss(y, tx, w)
    return w, np.array(loss)


def sigmoid(t):
    """
    Sigmoid function.

    Args:
        t (float or np.ndarray): Input

    Returns:
        (float or np.ndarray): Sigmoid of input
    """
    return 1 / (1 + np.exp(-t))

def calculate_loss(y, tx, w):
    """
    Logistic negative log-likelihood.

    Args:
        y (np.ndarray): Targets, shape (N,) or (N,1)
        tx (np.ndarray): Feature matrix, shape (N,D)
        w (np.ndarray): Weights, shape (D,)

    Returns:
        loss (np.ndarray): 0-dim array
    """
    y = y.flatten()
    pred = sigmoid(tx @ w)
    pred = np.clip(pred, 1e-15, 1 - 1e-15)
    loss = -np.sum(y * np.log(pred) + (1 - y) * np.log(1 - pred)) / y.shape[0]
    return np.array(loss)


def calculate_gradient(y, tx, w):
    """
    Gradient of logistic loss.

    Args:
        y (np.ndarray): Target vector, shape (N,)
        tx (np.ndarray): Feature matrix, shape (N,D)
        w (np.ndarray): Weights, shape (D,)

    Returns:
        grad (np.ndarray): Gradient vector, shape (D,)
    """
    grad = tx.T @ (sigmoid(tx @ w) - y) / y.shape[0]
    return grad


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression using gradient descent.

    Args:
        y (np.ndarray): Target vector, shape (N,)
        tx (np.ndarray): Feature matrix, shape (N,D)
        lambda_ (float): Regularization strength
        initial_w (np.ndarray): Initial weights, shape (D,)
        max_iters (int): Number of iterations
        gamma (float): Learning rate

    Returns:
        w (np.ndarray): Last weights, shape (D,)
        loss (np.ndarray): Last loss, 0-dim array
    """
    w = initial_w.copy()
    for n_iters in range(max_iters):
        _, grad = penalized_logistic_regression(y, tx, w, lambda_)
        w -= gamma * grad
    loss = calculate_loss(y, tx, w)
    return w, np.array(loss)


def penalized_logistic_regression(y, tx, w, lambda_):
    """
    Penalized logistic regression: compute loss + gradient.

    Args:
        y (np.ndarray): Target vector, shape (N,)
        tx (np.ndarray): Feature matrix, shape (N,D)
        w (np.ndarray): Weights, shape (D,)
        lambda_ (float): Regularization strength

    Returns:
        loss (np.ndarray): 0-dim array
        grad (np.ndarray): Gradient, shape (D,)
    """
    loss = calculate_loss(y, tx, w) + lambda_ * np.sum(w ** 2)
    grad = tx.T @ (sigmoid(tx @ w) - y) / y.shape[0] + 2 * lambda_ * w
    return np.array(loss), grad


def weighted_logistic_regression(
    y, tx, initial_w, max_iters, gamma, lambda_=0.0, tol=None
):
    """
    Weighted logistic regression using gradient descent.
    Adds sample-level weighting to handle class imbalance.

    Args:
        y (np.ndarray): Binary labels (0/1), shape (N, 1) or (N,)
        tx (np.ndarray): Feature matrix, shape (N, D)
        initial_w (np.ndarray): Initial weights, shape (D, 1)
        max_iters (int): Maximum number of iterations
        gamma (float): Learning rate
        lambda_ (float): Regularization strength (optional)
        tol (float): Optional early stopping tolerance

    Returns:
        w (np.ndarray): Learned weights, shape (D, 1)
        losses (list): List of loss values across iterations
    """

    if y.ndim == 1:
        y = y.reshape(-1, 1)
    w = initial_w.copy()
    N = len(y)

    assert tx.shape[0] == y.shape[0], f"Shape mismatch: X={tx.shape}, y={y.shape}"
    assert tx.shape[1] == w.shape[0], f"W shape mismatch: X={tx.shape}, w={w.shape}"

    # class weights: balance positive and negative classes
    n_pos = np.sum(y == 1)
    n_neg = N - n_pos
    pos_weight = N / (2 * n_pos + 1e-10)
    neg_weight = N / (2 * n_neg + 1e-10)
    sample_weights = np.where(y == 1, pos_weight, neg_weight)

    losses = []

    for it in range(max_iters):
        # predictions
        pred = sigmoid(tx @ w)
        pred = np.clip(pred, 1e-15, 1 - 1e-15)

        # weighted loss (with optional L2 regularization)
        loss = -np.sum(
            sample_weights * (y * np.log(pred) + (1 - y) * np.log(1 - pred))
        ) / np.sum(sample_weights)
        # loss += lambda_ * np.sum(w ** 2) # we dont add this right??
        losses.append(loss)

        # weighted gradient
        grad = (tx.T @ (sample_weights * (pred - y))) / np.sum(sample_weights)
        grad += 2 * lambda_ * w
        w -= gamma * grad

        if tol is not None and it > 5:
            rel_change = abs(losses[-2] - losses[-1]) / (abs(losses[-2]) + 1e-10)
            if rel_change < tol:
                break

    return w, np.array(losses)
