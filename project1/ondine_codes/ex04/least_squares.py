# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np
import costs

def least_squares(y, tx, fct='none'):
    """calculate the least squares solution."""
    wstar = np.linalg.solve(tx.T.dot(tx),(tx.T).dot(y))
    if fct=='mse':
        mse = costs.compute_mse(y,tx,wstar)
        return mse, wstar
    elif fct=='rmse':
        rmse = costs.compute_rmse(y,tx,wstar)
        return rmse, wstar
    else: #'none'
        return wstar
