from helpers_functions import *
from features_extraction import *
from features_increase import *
import numpy as np


def label_to_img_GMM(imgwidth, imgheight, w, h, labels):
    
    """
   From a vector of labels ( label in this case means the class of a specific cluster)          this function returns  a segmented image where each patch is  coloured according to the class
    """
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            im[j:j+w, i:i+h] = labels[idx]
            idx = idx + 1
    return im



def build_new_X(Z_GMM,X,n_components):
    """ 
    Adding the features obtained with GMM to X train matrix
    INPUT: - train matrix X of [n-rows,n_features] 
           - Z_GMM a [n ,1] dimensional vector obtained with image segmentation through                GMM. Z_GMM[i] corresponds to the class assigne to the patch which corresponds              to the i-th row of the matrix X
           - n_components: number of classes of GMM or in other words the number of                    clusters
    OUTPUT: A new train matrix with new features extracted thanks to GMM. 
 """
    R_mean=np.zeros(len(Z_GMM));
    G_mean=np.zeros(len(Z_GMM));
    B_mean=np.zeros(len(Z_GMM));

    R_var=np.zeros(len(Z_GMM));
    G_var=np.zeros(len(Z_GMM));
    B_var=np.zeros(len(Z_GMM));
    grey_mean=np.zeros(len(Z_GMM));
    grey_var=np.zeros(len(Z_GMM));
    difference_from_mean=np.zeros(len(Z_GMM));
    
    for i in range(0,n_components):
        Z_GMM_array=np.asarray(Z_GMM)
        mask_clusters=[Z_GMM_array==i]
       
        X_cluster=X[mask_clusters];
        R_mean[mask_clusters]=np.mean(X_cluster[:,0])
        # print(np.mean(X_cluster[:,0]))
        G_mean[mask_clusters]=np.mean(X_cluster[:,1])
        B_mean[mask_clusters]=np.mean(X_cluster[:,2])
        R_var[mask_clusters]=np.var(X_cluster[:,0])
        G_var[mask_clusters]=np.var(X_cluster[:,1])
        B_var[mask_clusters]=np.var(X_cluster[:,2])
        
        grey_mean[mask_clusters]=np.mean(X_cluster[:,6])
        difference_from_mean[mask_clusters] =X_cluster[:,6]-grey_mean[mask_clusters];
        grey_var[mask_clusters]=np.var(X_cluster[:,6])
        


    X=add_feature(X,R_mean);
    X=add_feature(X,G_mean);
    X=add_feature(X,B_mean);


    X=add_feature(X,R_var);
    X=add_feature(X,G_var);
    X=add_feature(X,B_var);

    X=add_feature(X,grey_mean);
    X=add_feature(X,grey_var);
    X=add_feature(X,difference_from_mean);
    return X

