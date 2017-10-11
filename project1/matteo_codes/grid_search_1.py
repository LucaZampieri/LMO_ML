
import numpy as np

#range_w =bidimensional array  D X 2 , the i-th row contains the min [0] and the max [1] value in which the w[i] is contained.


#number_of_sample= unidimensional array D X 1, the i-th row contains contains the number of sample of the w[i] that are included in the interval [range_w[i,0],range_w[i,1]]


def grid_search(y, tx, range_w,number_of_sample):
    """Algorithm for grid search."""
    global grid_w;
    
    #generating the grid
    grid_w = generate_grid_w(range_w,number_of_sample)
    print(grid_w);
  
    #recursive algorithm, recursion on n and w 
    w=[];   
    n=0;
    losses=compute_recursive_loss(y,tx,n,w);
                
    # ***************************************************
    #raise NotImplementedError


def generate_grid_w(range_w,number_of_sample):
    grid_w=[];
    for i in range (0,range_w.shape[0]-1):
        # PROBLEM: append method doesnt work correctly , it doesnt add the list to grid_w[i]
        # the alg should add an array containing the various instances of the w-ith weight to grid_w[i]
        # this for each i in D
        np.append(grid_w,np.linspace(range_w[i,0],range_w[i,1] , number_of_sample[i]),axis=0);
   
    return grid_w
                        
def compute_recursive_loss(y,tx,n,w):
    
    if(n<tx.shape[1]-1):
        losses=[];      
        #scannig the n-th row of range_w
        for i in range (0,range_w[n,:].shape[0]-1):
             # adding i-th instance of the n-th coefficents to the vector of weights 
            np.append(w,range_w[n,i]);        
            np.append(losses,compute_recursive_loss(y,tx,n+1,w));
    else :
            return compute_loss(y,tx,w) 
                        
    return losses
                 
                        
                       
                        
                        
                   
