#first stint of mse_gd that will be later copy-pasted into implementations.py

import numpy as np

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    N = y.shape[0]  # number of samples
    w = initial_w

    for n_iter in range(max_iters):
        e = y - tx @ w
        
        # compute loss (MSE)
        loss = np.sum(e**2) / (2 * N)
        
        # compute gradient
        grad = -(tx.T @ e) / N
        
        # update weights
        w = w - gamma * grad

        # print progress (generalized)
        w_str = ", ".join([f"w{i}={wi:.4f}" for i, wi in enumerate(w)])
        print(f"GD iter. {n_iter}/{max_iters-1}: loss={loss:.6f}, {w_str}")

    return w, loss