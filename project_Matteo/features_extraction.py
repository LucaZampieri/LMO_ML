
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image
import pandas as pd
from skimage import data, io, filters
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
from helpers_functions import *


# Extract 6-dimensional features consisting of average RGB color as well as variance
def extract_features(img):
    feat_m = np.mean(img, axis=(0,1))
    feat_v = np.var(img, axis=(0,1))
    feat = np.append(feat_m, feat_v)
    return feat

# Extract 2-dimensional features consisting of average gray color as well as variance
def extract_features_2d(img_patch):
    feat_m = np.mean(img_patch)
    feat_v = np.var(img_patch)
    feat = np.append(feat_m, feat_v)
    return feat

# Extract features for a given image
def extract_img_features(filename,n_features,patch_size,add_RGB_features,add_grey_features,add_RGB_spectrum_features,add_grey_spectrum_features):
    img = load_image(filename)
    img_patches = np.asarray(img_crop(img, patch_size, patch_size))
    print(img_patches.shape)
    X =extract_new_features(img_patches,n_features,add_RGB_features,add_grey_features,add_RGB_spectrum_features,add_grey_spectrum_features)
    return X

def create_train_table(n_column,n_features):
    return np.zeros([n_column,n_features]);

def add_feature(train_tx,new_column_feature):
        train_tx_new=np.ones([train_tx.shape[0],train_tx.shape[1]+1]);
      
        train_tx_new[:,0:train_tx.shape[1]]=train_tx;
        train_tx_new[:,train_tx.shape[1]]=new_column_feature;
        return train_tx_new;

def extract_mean_RGB(img):
    R = img[:,:,0];
    G = img[:,:,1];
    B = img[:,:,2];
    mean_R = np.mean(R);
    mean_G  = np.mean(G);
    mean_B=  np.mean(B);
    return [mean_R,mean_G,mean_B]


def extract_variance_RGB(img):
    R = img[:,:,0];
    G = img[:,:,1];
    B = img[:,:,2];
    var_R = np.var(R);
    var_G  = np.var(G);
    var_B=  np.var(B);
    return [var_R,var_G,var_B]

def get_spectrum(img_patches):
    dim=img_patches.shape;
    imgs_fourier=np.zeros([dim[0],dim[1],dim[2],dim[3],2])
    for i in range(len(img_patches)):
        img=img_patches[i];
        f = np.fft.fft2(img)
        imgs_fourier[i,:,:,:,0] = np.angle(f);
        imgs_fourier[i,:,:,:,1] = np.abs(f);
        
        
    return imgs_fourier;
        
    
def get_spectrum_grey(img_patches):
    dim=img_patches.shape;
    imgs_fourier=np.zeros([dim[0],dim[1],dim[2],2])
    for i in range(len(img_patches)):
        img=img_patches[i];
        img=RGB_to_grey(img)
        
        f = np.fft.fft2(img)
        imgs_fourier[i,:,:,0] = np.angle(f);
        imgs_fourier[i,:,:,1] = np.abs(f);
    return imgs_fourier;
        
def extract_mean_spectrum_grey(spectrum):
    phase = spectrum[:,:,0];
    abs_= spectrum[:,:,1];
    mean_phase=np.mean(phase);
    mean_abs=np.mean(abs_);
    return [mean_phase,mean_abs]

def extract_variance_spectrum_grey(spectrum):
    phase = spectrum[:,:,0];
    abs_= spectrum[:,:,1];
    mean_phase=np.mean(phase);
    mean_abs=np.mean(abs_);
    return [mean_phase,mean_abs]
    
    

    
def extract_mean_spectrum(spectrum):
    phase_1 = spectrum[:,:,0,0];
    phase_2= spectrum[:,:,1,0];
    phase_3= spectrum[:,:,2,0];
    abs_1 = spectrum[:,:,0,1];
    abs_2= spectrum[:,:,1,1];
    abs_3= spectrum[:,:,2,1];
    
    mean_phase_1=np.mean(phase_1);
    mean_phase_2=np.mean(phase_2);
    mean_phase_3=np.mean(phase_3);
    mean_abs_1=np.mean(abs_1);
    mean_abs_2=np.mean(abs_2);
    mean_abs_3=np.mean(abs_1);
    
   
    return  [mean_phase_1,mean_phase_2,mean_phase_3,mean_abs_1,mean_abs_2,mean_abs_3]

    

def extract_variance_spectrum(spectrum):
    phase_1 = spectrum[:,:,0,0];
    phase_2= spectrum[:,:,1,0];
    phase_3= spectrum[:,:,2,0];
    abs_1 = spectrum[:,:,0,1];
    abs_2= spectrum[:,:,1,1];
    abs_3= spectrum[:,:,2,1];
    
    var_phase_1=np.var(phase_1);
    var_phase_2=np.var(phase_2);
    var_phase_3=np.var(phase_3);
    var_abs_1=np.var(abs_1);
    var_abs_2=np.var(abs_2);
    var_abs_3=np.var(abs_1);
    return  [var_phase_1,var_phase_2,var_phase_3,var_abs_1,var_abs_2,var_abs_3]




def extract_new_features(img_patches,n_features,add_RGB_features,add_grey_features,add_RGB_spectrum_features,add_grey_spectrum_features):
    
    
    train_tx=create_train_table(len(img_patches),n_features);
    
    
    if(add_RGB_features==1):
        for i in range(len(img_patches)):
            index_feature=0;
            [mean_R,mean_G,mean_B]=extract_mean_RGB(img_patches[i]);
            train_tx[i,index_feature]=mean_R;
            index_feature=index_feature+1;
            train_tx[i,index_feature]=mean_G;
            index_feature=index_feature+1;
            train_tx[i,index_feature]=mean_B;
            index_feature=index_feature+1;
            [var_R,var_G,var_B]=extract_variance_RGB(img_patches[i]);
            train_tx[i,index_feature]=var_R;
            index_feature=index_feature+1;
            train_tx[i,index_feature]=var_G;
            index_feature=index_feature+1;
            train_tx[i,index_feature]=var_B;
            index_feature=index_feature+1;
            prev_index_feature=index_feature;
            
            
            
    if(add_RGB_spectrum_features==1):
        img_patches_fourier=get_spectrum(img_patches)
        for i in range(len(img_patches_fourier)):
            [mean_abs1,mean_abs2,mean_abs3,mean_phase1,mean_phase2,mean_phase3]=extract_mean_spectrum(img_patches_fourier[i]);
            index_feature= prev_index_feature
            
            train_tx[i,index_feature]=mean_abs1;
            index_feature=index_feature+1;
            train_tx[i,index_feature]=mean_abs2;
            index_feature=index_feature+1;
            train_tx[i,index_feature]=mean_abs3;
            index_feature=index_feature+1;
            train_tx[i,index_feature]=mean_phase1;
            index_feature=index_feature+1;
            train_tx[i,index_feature]=mean_phase2;
            index_feature=index_feature+1;
            train_tx[i,index_feature]=mean_phase3;
            index_feature=index_feature+1;
             
            
    

            [var_abs1,var_abs2,var_abs3,var_phase1,var_phase2,var_phase3]=extract_variance_spectrum(img_patches_fourier[i]);    
            train_tx[i,index_feature]=var_abs2;
            index_feature=index_feature+1;
            train_tx[i,index_feature]=var_abs2;
            index_feature=index_feature+1;
            train_tx[i,index_feature]=var_abs3;
            index_feature=index_feature+1;
            train_tx[i,index_feature]=var_phase1;
            index_feature=index_feature+1;
            train_tx[i,index_feature]=var_phase2;
            index_feature=index_feature+1;
            train_tx[i,index_feature]=var_phase3;
            index_feature=index_feature+1;
            
            
            
    prev_index_feature=index_feature;
# extract the spectrum with grey_scale
    if(add_grey_spectrum_features==1):
        img_patches_fourier=get_spectrum_grey(img_patches)
        print(img_patches_fourier.shape)
        for i in range(len(img_patches_fourier)):
            
            index_feature= prev_index_feature
            [mean_abs,mean_phase]=extract_mean_spectrum_grey(img_patches_fourier[i]);
            train_tx[i,index_feature]=mean_abs;
            index_feature=index_feature+1;
            train_tx[i,index_feature]=mean_phase;
            index_feature=index_feature+1;
            [var_abs,var_phase]=extract_variance_spectrum_grey(img_patches_fourier[i]);    
            train_tx[i,index_feature]=var_abs;
            index_feature=index_feature+1;
            train_tx[i,index_feature]=var_phase;
            index_feature=index_feature+1;
    
    
    
    prev_index_feature=index_feature;
    if(add_grey_features==1):
    
#extract average grey features
        for i in range(len(img_patches)):
            index_feature= prev_index_feature
            [mean_grey,var_grey]=extract_features_2d(img_patches[i])
            train_tx[i,index_feature]=mean_grey;
            index_feature=index_feature+1;
            train_tx[i,index_feature]=var_grey;
            index_feature=index_feature+1;


        
    return train_tx
        


