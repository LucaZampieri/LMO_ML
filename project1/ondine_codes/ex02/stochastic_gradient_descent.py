# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""
import costs

def compute_stoch_gradient(y, tx, w, fct='mse'):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    e = y-np.dot(tx,w)
    if fct=='mse':
        return -1./y.shape[0]*tx.T.dot(e)
    else: #MAE
        return -1./y.shape[0]*tx.T.dot(np.sign(e)) # note: sign(0)=0


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma, fct='mse'):
    """Stochastic gradient descent algorithm."""
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        loss = 0
        gradLw = 0
        for mini_y, mini_tx in batch_iter(batch_size=batch_size,tx=tx,y=y):
            loss += costs.compute_loss(mini_y,mini_tx,w,fct)/batch_size
            gradLw += compute_stoch_gradient(mini_y,mini_tx,w,fct)/batch_size
        w = w-gamma*gradLw
        ws.append(w)
        losses.append(loss)
        #print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws
