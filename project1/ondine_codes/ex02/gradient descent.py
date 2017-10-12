# -*- coding: utf-8 -*-
"""Gradient Descent"""
import costs

def compute_gradient(y, tx, w, fct='mse'):
    """Compute the gradient."""
    e = y-tx.dot(w)
    if fct=='mse':
        return -1./y.shape[0]*tx.T.dot(e)
    else: #MAE
        return -1./y.shape[0]*tx.T.dot(np.sign(e)) # note: sign(0)=0


def gradient_descent(y, tx, initial_w, max_iters, gamma, fct='mse'):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        loss = costs.compute_loss(y,tx,w,fct)
        gradLw = compute_gradient(y,tx,w,fct)
        w = w-gamma*gradLw
        # store w and loss
        ws.append(w)
        losses.append(loss)
        #print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws

