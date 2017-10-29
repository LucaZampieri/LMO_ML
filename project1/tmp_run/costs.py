# -*- coding: utf-8 -*-
"""TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""

import numpy as np 


def compute_loss(y, tx, w, fct='mse'):
	"""Compute the loss using mse, rmse or mae.
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
	"""Compute the loss using ridge regression, using mse, rmse or mae.
	"""
	if (fct=='mse' or fct=='mae'):
		loss = compute_loss(y,tx,w,fct)
		return loss + lambda_*np.linalg.norm(w)**2
	elif fct=='rmse':
		return np.sqrt(2*compute_ridge_loss(y,tx,w,lambda_,'mse'))
	else:
		raise NotImplementedError


def compute_logreg_loss(y, tx, w):
    """Compute the cost by negative log likelihood."""
    dot_prod = tx.dot(w)
    log = np.log(1+np.exp(dot_prod))
    loss = np.ones(len(y)).dot(log) - (y.T.dot(tx)).dot(w)
    
    return loss
    
    
def compute_gradient(y, tx, w, fct='mse'):
	"""Compute the gradient of the MSE, MAE or the logistic regression losses."""
	e = y-tx.dot(w)
	if fct=='mse':
		return -1./len(y)*tx.T@e
	elif fct=='mae':
		return -1./len(y)*tx.T.dot(np.sign(e))
	elif fct=='logreg':
		return (tx.T).dot(sigmoid(tx.dot(w))-y)
	else:
		raise NotImplementedError
    

def compute_logreg_hessian(y, tx, w):
    """Compute the hessian of the logistic regression loss function"""
    sig_dot_prod = sigmoid(tx.dot(w))
    S = sig_dot_prod*(1-sig_dot_prod) 
    return ((tx.T)*S).dot(tx)
    
    
# *************************** HELPERS *********************************
    
def sigmoid(z):
    """Sigmoid function."""
    result = np.exp(z)/(1+np.exp(z))
    return result