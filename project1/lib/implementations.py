# -*- coding: utf-8 -*-
"""Least Squares (or Mean Absolute) Gradient Descent"""

import numpy as np 
from numpy import matlib

# choosse between the next two for importing costs
# import costs
import lib.costs as costs


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def least_squares_GD(y, tx, initial_w, max_iters, gamma,fct):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        loss = costs.compute_loss(y,tx,w,fct)
        gradLw = costs.compute_gradient(y,tx,w)
        w = w-gamma*gradLw
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return  ws[-1], losses[-1]


def least_squares_SGD(y, tx, initial_w, max_iters, gamma, batch_size):
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
        #print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
        #      bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return ws[-1], losses[-1]
    
    
def least_squares(y, tx, fct):
    """Calculate the least squares solution."""
    wstar = np.linalg.solve(tx.T.dot(tx),(tx.T).dot(y))
    if (fct=='mse' or fct=='rmse'):
        loss = costs.compute_loss(y,tx,wstar,fct)
        return wstar, loss 
    else: 
        raise NotImplementedError

def ridge_regression(y, tx, lambda_, fct='none'):
    """Implement ridge regression."""
    wstar = np.linalg.solve(tx.T.dot(tx)+2*len(y)*lambda_*np.matlib.identity(tx.shape[1]),(tx.T).dot(y))
    if (fct=='mse' or fct=='rmse'):
        ridge = costs.compute_ridge_loss(y,tx,wstar,lambda_,fct)
        return wstar, ridge
    else: #'none'
        raise NotImplementedError 


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




####################
def compute_log_like (y, tx, initial_w):
    
  #reashaping: if a vector v(n,) is received this reashape in v(n,1)>>> this permit to work better 
    if(len(initial_w.shape)==1):
        initial_w=initial_w.reshape(len(initial_w),1);
    if(len(y.shape)==1):
        y=y.reshape(len(y),1);  
    if(len(tx.shape)==1):
        tx=tx.reshape(len(tx),1); 
 

 #calculating log_like
    log_like=0;
    for i in range(0,len(y)):
        log_like=log_like+(np.log(1+np.exp(tx[i,:].T.dot(initial_w)))-y[i,:].dot(tx[i,:].dot(initial_w)));
        
    return log_like;

####################

    
def logistic_gradient_descent(y, tx, initial_w,max_iters,gamma):
   #reashaping
    threshold = 1e-10
    if(len(initial_w.shape)==1):
        initial_w=initial_w.reshape(len(initial_w),1);
    if(len(y.shape)==1):
        y=y.reshape(len(y),1);  
    if(len(tx.shape)==1):
        tx=tx.reshape(len(tx),1);  
    log_like_list=[];
    #iterating to find the min
    w_opt=initial_w;
    log_like=0;
    for j in range(1,max_iters):
        
        
        if(len(initial_w.shape)==1):
            initial_w=initial_w.reshape(len(initial_w),1);
            
            
        log_like=compute_log_like(y, tx, initial_w);
        
        
        if len(log_like_list) > 1 and np.abs(log_like_list[-1] -log_like_list[-2]) < threshold:
            break
        log_like_list.append(log_like);
        
        
        v=tx.dot(initial_w);
        sigma=np.zeros(v.shape);
        for i in range(0,len(v)):
                sigma[i,:]= np.exp(v[i,:])/((1+np.exp(v[i,:])));
                

        grad_logistic=tx.T.dot((sigma-y));
        initial_w=initial_w-gamma*grad_logistic;
        
        w_opt=initial_w;
        print("Gradient Descent logistic ({bi}/{ti}): loss={l}".format(
              bi=j, ti=max_iters - 1, l=log_like))       
        
    #foundamental reshape: to be consistent with the input. If we want a vector w_opt of 
    # dimension N x 0 and not N x 1 this part of algo performes this reshaping.
    
    
    if(w_opt.shape[1]==1):
    
            w_opt_1=np.zeros(w_opt.shape[0]);
            for i in range (0,w_opt.shape[0]):
                w_opt_1[i]=w_opt[i];
                
            
    return w_opt_1,log_like;


####################




    
def logistic_regression_mat (y, tx, initial_w, max_iters, gamma):
      
        
    [w_opt,log_like] =logistic_gradient_descent(y, tx, initial_w,max_iters,gamma);
    loss=log_like;
    return w_opt,log_like
