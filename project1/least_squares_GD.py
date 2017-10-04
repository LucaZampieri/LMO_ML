# -*- coding: utf-8 -*-
"""Least Squares (or Mean Absolute) Gradient Descent"""

def compute_loss(y, tx, w, fct):
    """Calculate the loss.
    You can calculate the loss using mse or mae.
    """
    e = y-tx.dot(w)
    if fct=='mse':
        return 1./(2.*y.shape[0])*e.dot(e)
    else if fct=='mae':
        return 1./(2*y.shape[0])*np.sum(np.abs(e))
    else:
        raise NotImplementedError
    
    
def compute_gradient(y, tx, w, fct):
    """Compute the gradient."""
    e = y-tx.dot(w)
    if fct=='mse':
        return -1./y.shape[0]*tx.T.dot(e)
    else if fct=='mae':
        return -1./y.shape[0]*tx.T.dot(np.sign(e))
    else:
        raise NotImplementedError

        
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        loss = compute_loss(y,tx,w,'mse')
        gradLw = compute_gradient(y,tx,w)
        w = w-gamma*gradLw
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws