# -*- coding: utf-8 -*-
"""Least Squares (or Mean Absolute) Gradient Descent"""

import numpy as np 
from numpy import matlib
import lib.costs as costs

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
        return loss, wstar
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


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic Regression using gradient descent"""
    
    # Init parameters
    threshold = 1e-8
    losses = []
    
    # Logistic regression loop
    for iter in range(max_iters):
        
        # Calculate actual loss and gradient
        loss = costs.compute_logreg_loss(y, tx, initial_w)
        grad = costs.compute_gradient(y, tx, initial_w, 'logreg')
        
        # Update initial_w and keep the losses into memory
        initial_w = initial_w - gamma*grad
        losses.append(loss)
        
        # Converge criteria
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return initial_w, losses[-1]


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent"""
   
    # Init parameters
    threshold = 1e-8
    losses = []
    
    # Regularized logistic regression loop
    for iter in range(max_iters):
        
        # Calculate actual loss and gradient
        loss = costs.compute_logreg_loss(y, tx, initial_w) + lambda_*np.sum(initial_w*initial_w)
        grad = costs.compute_gradient(y, tx, initial_w, 'logreg') + 2*lambda_*initial_w
        
        # Update initial_w and keep the losses into memory
        initial_w = initial_w - gamma*grad
        losses.append(loss)
        
        # Converge criteria
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return initial_w, losses[-1]


def reg_logistic_regression_newton(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent"""
   
    # Init parameters
    threshold = 1e-8
    losses = []
    
    # Regularized logistic regression loop - Newton method
    for iter in range(max_iters):
		
        # Calculate actual loss and gradient
        loss = costs.compute_logreg_loss(y, tx, initial_w) + lambda_*np.sum(initial_w*initial_w)
        grad = costs.compute_gradient(y, tx, initial_w, 'logreg') + 2*lambda_*initial_w
        H = costs.compute_logreg_hessian(y, tx, initial_w) + 2*lambda_*np.identity(initial_w.shape[0])
        
        # Update initial_w and keep the losses into memory
        initial_w = initial_w - gamma*np.linalg.inv(H).dot(grad)
        losses.append(loss)
        
        # Converge criteria
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return initial_w, losses[-1]
