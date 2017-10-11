# -*- coding: utf-8 -*-
"""Least Squares (or Mean Absolute) Gradient Descent"""

import numpy as np 

def compute_loss(y, tx, w, fct='mse'):
    """Calculate the loss.
    You can calculate the loss using mse or mae.
    """
    e = y-tx.dot(w)
    if fct=='mse':
        return 1./(2.*y.shape[0])*e.T@e
    elif fct=='mae':
        return 1./(2*y.shape[0])*np.sum(np.abs(e))
    else:
        raise NotImplementedError
    
    
def compute_gradient(y, tx, w, fct='mse'):
    """Compute the gradient."""
    e = y-tx.dot(w)
    if fct=='mse':
        return -1./y.shape[0]*tx.T@e
    elif fct=='mae':
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


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        loss = 0
        gradLw = 0
        for mini_y, mini_tx in batch_iter(batch_size=batch_size,tx=tx,y=y):
            loss += compute_loss(mini_y,mini_tx,w,'mse')/batch_size
            gradLw += compute_gradient(mini_y,mini_tx,w,'mse')/batch_size
        w = w-gamma*gradLw
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws

def least_squares_matteo(y, tx):
    """calculate the least squares solution."""
    # ***************************************************
    
    w_opt=(np.linalg.inv((tx.T.dot(tx))).dot(tx.T)).dot(y);
    if(len(w_opt.shape)==1):
        w_opt=w_opt.reshape(len(w_opt),1);
    # INSERT YOUR CODE HERE
    # least squares: TODO
    # returns mse, and optimal weights
    # ***************************************************
    
    return w_opt

def least_squares_luca(y, tx):
    """calculate the least squares solution."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # least squares: TODO
    # returns mse, and optimal weights
    # ***************************************************
    Gram = tx.T@tx
    w_star = np.linalg.inv(Gram)@tx.T@y
    e = y-tx@w_star
    mse = 1/2/y.shape[0]*e.T@e
    print(len(w_star.shape))
    return mse, w_star
    raise NotImplementedError
    
def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    A=tx.T.dot(tx);
    lambda_=lambda_*(2*y.shape[0]);
    w_opt=(np.linalg.inv(A+lambda_*np.identity(A.shape[0])).dot(tx.T)).dot(y); 
    
    return w_opt
    # ridge regression: TODO
    # ***************************************************