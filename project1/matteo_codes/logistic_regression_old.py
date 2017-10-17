import numpy as np


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
    if(len(initial_w.shape)==1):
        initial_w=initial_w.reshape(len(initial_w),1);
    if(len(y.shape)==1):
        y=y.reshape(len(y),1);  
    if(len(tx.shape)==1):
        tx=tx.reshape(len(tx),1);  
    
    #iterating to find the min
    for j in range(1,max_iters):
        
        log_like=compute_log_like(y, tx, initial_w);
        if(len(initial_w.shape)==1):
            initial_w=initial_w.reshape(len(initial_w),1);
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




    
def logistic_regression (y, tx, initial_w, max_iters, gamma):
      
        
    [w_opt,log_like] =logistic_gradient_descent(y, tx, initial_w,max_iters,gamma);
    loss=log_like;
    return w_opt,log_like
          