# -*- coding: utf-8 -*-
"""Least Squares (or Mean Absolute) Gradient Descent"""

import numpy as np 
from numpy import matlib
import costs


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        loss = costs.compute_loss(y,tx,w,'mse')
        gradLw = costs.compute_gradient(y,tx,w)
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
            loss += costs.compute_loss(mini_y,mini_tx,w,'mse')/batch_size
            gradLw += costs.compute_gradient(mini_y,mini_tx,w,'mse')/batch_size
        w = w-gamma*gradLw
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws
    
    
def least_squares(y, tx, fct='none'):
    """Calculate the least squares solution."""
    wstar = np.linalg.solve(tx.T.dot(tx),(tx.T).dot(y))
    if (fct=='mse' or fct=='rmse'):
        loss = costs.compute_loss(y,tx,wstar,fct)
        return mse, wstar
    else: #'none'
        return wstar


def ridge_regression(y, tx, lambda_, fct='none'):
    """Implement ridge regression."""
    wstar = np.linalg.solve(tx.T.dot(tx)+2*len(y)*lambda_*np.matlib.identity(tx.shape[1]),(tx.T).dot(y))
    if (fct=='mse' or fct=='rmse'):
        ridge = costs.compute_ridge_loss(y,tx,wstar,lambda_,fct)
        return ridge, wstar
    else: #'none'
        return wstar


# **********************************************************************
# ************************ TO CLEAN ************************************
# **********************************************************************


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
    #print(len(w_star.shape))
    return mse, w_star
    raise NotImplementedError
    
def ridge_regression_luca_matteo(y, tx, lambda_):
    """implement ridge regression."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    A=tx.T.dot(tx);
    lambda_=lambda_*(2*y.shape[0]);
    w_opt=(np.linalg.inv(A+lambda_*np.identity(A.shape[0])).dot(tx.T)).dot(y); 
    
    return w_opt
    # ridge regression: TODO
    # ***************************************************
