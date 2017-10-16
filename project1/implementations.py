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

    return ws,losses


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
    #print(len(w_star.shape))
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
    
# compute the log_likehood function =loss function 
def compute_log_like_reg (y, tx, initial_w,lambda_):
    
   
    if(len(initial_w.shape)==1):
        initial_w=initial_w.reshape(len(initial_w),1);
    if(len(y.shape)==1):
        y=y.reshape(len(y),1);  
    if(len(tx.shape)==1):
        tx=tx.reshape(len(tx),1); 
    
    log_like=0;
    for i in range(0,len(y)):
        log_like=log_like+(np.log(1+np.exp(tx[i,:].T.dot(initial_w)))-y[i,:].dot(tx[i,:].dot(initial_w)));
        
    log_like=log_like+lambda_*0.5*(initial_w.T.dot(initial_w));
    
    
    return log_like;



   #find the w tha minimize the log-likehood >>> maximize a priori probability
def logistic_gradient_descent_reg(y, tx, initial_w,max_iters,gamma,lambda_):
   
    if(len(initial_w.shape)==1):
        initial_w=initial_w.reshape(len(initial_w),1);
    if(len(y.shape)==1):
        y=y.reshape(len(y),1);  
    if(len(tx.shape)==1):
        tx=tx.reshape(len(tx),1);  
    
    for j in range(1,max_iters):
        
        log_like=compute_log_like_reg(y, tx, initial_w,lambda_);
        if(len(initial_w.shape)==1):
            initial_w=initial_w.reshape(len(initial_w),1);
        v=tx.dot(initial_w);
        sigma=np.zeros(v.shape);
        for i in range(0,len(v)):
                sigma[i,:]= np.exp(v[i,:])/((1+np.exp(v[i,:])));
                
        #different from normal-logistic a new term is added to compute the gradient
        #basically "lambda_*initial_w" is the derivate of "lambda_*(norm2(initial_w))^2
        
        grad_logistic_reg=tx.T.dot((sigma-y))+lambda_*initial_w;
        
        initial_w=initial_w-gamma*grad_logistic_reg;
        w_opt=initial_w;
        print("Gradient Descent logistic Regularized ({bi}/{ti}): log_like={l}".format(
              bi=j, ti=max_iters - 1, l=log_like))       
        
        
    if(w_opt.shape[1]==1):
            w_opt_1=np.zeros(w_opt.shape[0]);
            for i in range (0,w_opt.shape[0]):
                w_opt_1[i]=w_opt[i];
                
            
    return w_opt_1,log_like;

    
def logistic_regression_reg (y, tx, initial_w, max_iters, gamma,lambda_):
         
    [w_opt,log_like] =logistic_gradient_descent_reg(y, tx, initial_w,max_iters,gamma,lambda_);
    loss=log_like;
    return w_opt,log_like