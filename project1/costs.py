# -*- coding: utf-8 -*-
"""TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""

import numpy as np 


def compute_loss(y, tx, w, fct='mse'):
    """Calculate the loss using mse, rmse or mae.
    """
    e = y-tx.dot(w)
    if fct=='mse':
        return 1./(2.*len(y))*e.T@e
    elif fct=='mae':
        return 1./(2*len(y))*np.sum(np.abs(e))
    elif fct=='rmse':
		return np.sqrt(2*compute_loss(y,tx,w,'mse'))
    else:
        raise NotImplementedError
        
        
def compute_ridge_loss(y, tx, w, lambda_, fct='mse'):
	"""Calculate the loss using ridge regression, using mse, rmse or mae.
    """
    if fct=='mse' || fct=='mae':
		loss = compute_loss(y,tx,w,fct)
		return loss + lambda_*np.linalg.norm(w)**2
	elif fct=='rmse':
		return np.sqrt(2*compute_ridge_loss(y,tx,w,lambda_,'mse'))
	else:
		raise NotImplementedError

    
def compute_gradient(y, tx, w, fct='mse'):
    """Compute the gradient."""
    e = y-tx.dot(w)
    if fct=='mse':
        return -1./len(y)*tx.T@e
    elif fct=='mae':
        return -1./len(y)*tx.T.dot(np.sign(e))
    else:
        raise NotImplementedError

