# File cost.py computing the losses with MSE, MAE, RMSE and for the ridge regression.

# Authors: Ondine Chanon, Matteo Ciprian, Luca Zampieri
# license: MIT

# Import external libraries
import numpy as np 


def compute_loss(y, tx, w, fct='mse'):
	"""Compute the loss using the mean square error (MSE), the root mean square
	   error (RMSE) or the mean absolute error (MAE).

	   Input: 
		- y: array of output values.
		- tx: array or matrix of data.
		- w: array of the given regression weights.
		- fct: (optional parameter) string specifying the type of loss function. 
		       It can be either 'mse', 'mae' or 'rmse'. By default='mse'.
	   Ouput: 
		- scalar, the value of the chosen loss function.
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
	"""Compute the loss for the ridge regression, using the mean square error (MSE), 
	   the root mean square error (RMSE) or the mean absolute error (MAE), plus a 
	   stabilizing term controlled with parameter lambda_.

	   Input: 
		- y: array of output values.
		- tx: array or matrix of data.
		- w: array of the given regression weights.
		- lambda_: scalar, stabilizing parameter.
		- fct: (optional parameter) string specifying the type of loss function. 
		       It can be either 'mse', 'mae' or 'rmse'. By default='mse'.
	    Ouput: 
		- scalar, the value of the ridge regression loss function. 
	"""
	if (fct=='mse' or fct=='mae'):
		loss = compute_loss(y,tx,w,fct)
		return loss + lambda_*np.linalg.norm(w)**2
	elif fct=='rmse':
		return np.sqrt(2*compute_ridge_loss(y,tx,w,lambda_,'mse'))
	else:
		raise NotImplementedError

