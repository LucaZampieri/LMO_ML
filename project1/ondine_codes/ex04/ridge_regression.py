# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np
import costs
from numpy import matlib

def compute_ridge_mse(y,tx,w,lambda_):
    mse = costs.compute_mse(y,tx,w)
    return mse + lambda_*np.linalg.norm(w)**2
    

def compute_ridge_rmse(y,tx,w,lambda_):
    return np.sqrt(2*compute_ridge_mse(y,tx,w,lambda_))
    

def ridge_regression(y, tx, lambda_, fct='none'):
    """implement ridge regression."""
    wstar = np.linalg.solve(tx.T.dot(tx)+2*y.shape[0]*lambda_*np.matlib.identity(tx.shape[1]),(tx.T).dot(y))
    if fct=='mse':
        ridge = compute_ridge_mse(y,tx,wstar,lambda_)
        return ridge, wstar
    elif fct=='rmse':
        ridge = compute_ridge_rmse(y,tx,wstar,lambda_)
        return ridge, wstar
    else: #'none'
        return wstar
