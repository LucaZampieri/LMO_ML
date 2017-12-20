import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image
import glob


""" This functions "correct" the image filling the holes. For instance if a white patch (foreground) is sorrounded by black (background)  it is probably a black so the function set it to black  (or the opposite) """

def binary_to_uint8(img):
    rimg = (img * 255).round().astype(np.uint8)
    return rimg


def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches


def load_image(infilename):
    data = mpimg.imread(infilename)
    print(infilename);
    return data

def patch_to_img(imdim, w, h, patches):
    if(len(imdim) < 3):
        im=np.zeros([imdim[0],imdim[1]])
    else:
            im=np.zeros([imdim[0],imdim[1],imdim[2]])
    imgwidth = imdim[0]
    imgheight = imdim[0]
    
    idx = 0

    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if(len(imdim) < 3):
                im[j:j+w, i:i+h] =patches[idx]
                idx = idx + 1
            
            else:
                im[j:j+w, i:i+h,:] =patches[idx]
                idx = idx + 1
            
    return im



def get_neighbours_index(i,j,n_r,n_c):
    """ Given the position of the patch ( i,j) it return the indexes of the neighbours in the one dimensional list img_patches"""
    i=int(i);
    j=int(j);
    n_r=int(n_r)
    n_c=int(n_c)
    if (i==0):
            if(j==0):
                return([j*n_r+i+1,(j+1)*n_r+i,(j+1)*n_r+i+1])
            else:
                if(j==n_c-1):
                    return([(j-1)*n_r+i,(j-1)*n_r+i+1,j*n_r+i+1])
                else:
                    return([(j-1)*n_r+i,(j+1)*n_r+i,(j-1)*n_r+i+1,j*n_r+i+1,(j+1)*n_r+i+1])
    else: 
            
        if (i==n_r-1):
                if(j==0):
                    return([j*n_r+i-1,(j+1)*n_r+i-1,(j+1)*n_r+i])
                else:
                    if(j==n_c-1):
                        return([(j-1)*n_r+i-1,j*n_r+i-1,(j-1)*n_r+i])
                    else:
                        return([(j-1)*n_r+i-1,j*n_r+i-1,(j+1)*n_r+i-1,(j-1)*n_r+i,(j+1)*n_r+i])
                
        else: 
            if(j==0):
                return([j*n_r+i-1,(j+1)*n_r+i-1,(j+1)*n_r+i,j*n_r+i+1,(j+1)*n_r+i+1])
            else:
                if(j==n_c-1):
                    return([j*n_r+i-1,(j-1)*n_r+i-1,(j-1)*n_r+i,j*n_r+i+1,(j-1)*n_r+i+1])
                else:
                    return([j*n_r+i-1,(j-1)*n_r+i-1,(j-1)*n_r+i,j*n_r+i+1,(j-1)*n_r+i+1,(j+1)*n_r+i-1,(j+1)*n_r+i,(j+1)*n_r+i+1])
            
def fill_the_gaps_on_patches(im_patches,im_dim,w,h):
    """ It receive a series of patches and the dimensions of the image. FOr each patches this function scan the patch's neighbor and if it is sorrounded by patch of different colour change the colour of the patch"""
    n_patch_row=int(im_dim[0]/w);
    n_patch_col=int(im_dim[1]/h);
    
    means_list=[]
    for k in range(len(im_patches)):
        mean=np.mean(im_patches[k]);
        means_list.append(mean)
    
    
    for k in range(len(im_patches)):
        
        index_row=int(k%n_patch_col);
       
      
        index_col=int(k//n_patch_col);
       
        
        patch=im_patches[k];
        mean=np.mean(patch);
        list_of_index=get_neighbours_index(index_row,index_col,n_patch_row,n_patch_col)
        means_neighbours=[]
      
        for i1 in range(len(list_of_index)):
            list_of_index[i1]
            means_neighbours.append(means_list[list_of_index[i1]]);

            
                                        
        count_white=0;
        count_black=0;
        for i in range(len(means_neighbours)):
            if(means_neighbours[i]==1):
                count_white=count_white+1;
                
            if(means_neighbours[i]==0):
                count_black=count_black+1;
        
        if(means_list[k]==0):
            if (len(means_neighbours)==8):
                    if (count_white>5):
                        im_patches[k]=np.ones([w,h])
                       
                        
            if (len(means_neighbours)==3):     
                    if (count_white>2):
                        im_patches[k]=np.ones([w,h])
                        
                        
            if (len(means_neighbours)==5):      
                    if (count_white>4):
                        im_patches[k]=np.ones([w,h])
                        
                        
        if(means_list[k]==1):
            if (len(means_neighbours)==8):
                    if (count_black>7): # here we are more conservative
                        im_patches[k]=np.zeros([w,h])
                       
                       
            if (len(means_neighbours)==3):     
                    if (count_black>2):
                        im_patches[k]=np.zeros([w,h])
                       
                        
            if (len(means_neighbours)==5):      
                    if (count_black>4):
                        im_patches[k]=np.zeros([w,h])
                        
                       
    
    return im_patches


    
    
    
def fill_the_gaps_image(img,w,h,im_dim):
    """ It receive an image just readen by the function below and also the dimensions of the patch , w and h, and the dimension of the image"""
    im_patches=img_crop(img,w,h);
    new_im_patches=fill_the_gaps_on_patches(im_patches,im_dim,w,h);
    new_im=patch_to_img(im_dim, w, h, new_im_patches)
    return new_im;



# the path of the images has to be "name_folder/"
def fill_the_gaps(folder,w,h):
    """ Receiving the path of the images this function save in the current folder the images corrected"""
    root_dir = folder
    image_dir = root_dir;
    files = os.listdir(image_dir)
    files = list(files)

    n=len(files)
    print("Loading " + str(n) + " images")
     
    imgs_output = [load_image(image_dir + files[i]) for i in range(n) if files[i].endswith('.png')]
    new_imgs_output=imgs_output;
    print(len(imgs_output));
    im_dim=imgs_output[0].shape;
    
   
    for i in range(len(imgs_output)):
        new_imgs_output[i]=fill_the_gaps_image(imgs_output[i],w,h,im_dim);
        img=new_imgs_output[i];
        ind=i+1;
        predicted_im=binary_to_uint8(img);
        
        Image.fromarray(predicted_im).save('prediction_corrected' + '%.3d' % ind + '.png')
        
            
    
   


    

                
    

                
