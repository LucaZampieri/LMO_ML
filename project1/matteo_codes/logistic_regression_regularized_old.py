import numpy as np


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
          