"""File with funtion for computing the F-Score """
import os,sys
import matplotlib.image as mpimg
from helpers_functions import *
import numpy as numpy
import cv2
# Assign a label to a patch v
def value_to_class(v,foreground_threshold):
    # you can remark the hot encoding
     # percentage of pixels > 1 required to assign a foreground label to a patch
    df = numpy.sum(v)
    if df > foreground_threshold:
        return [0, 1]
    else:
        return [1, 0]


def resize_to_256(img):
    return cv2.resize(img, (256,256), interpolation = cv2.INTER_AREA)


# Extract label images
def extract_labels(filename, num_images):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    gt_imgs = []
    for i in range(1, num_images+1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            
            img = mpimg.imread(image_filename)
           
            gt_imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    num_images = len(gt_imgs)
    gt_patches = [img_crop(gt_imgs[i], 16, 16) for i in range(num_images)]
    data = numpy.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    labels = numpy.asarray([value_to_class(numpy.mean(data[i]),0.25) for i in range(len(data))])

    # Convert to dense 1-hot representation.
    return labels.astype(numpy.float32)

# Extract label images
def extract_labels_for_validation(foldername, num_images,th):
    """Extract the labels of prediction into a 1-hot matrix [image index, label index]."""
    gt_imgs = []
    for i in range(1, num_images+1):
        imageid = str('prediction_' + '%.3d' % i)
        image_filename = foldername + imageid + ".png"
        if os.path.isfile(image_filename):
            
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print ('File ' + image_filename + ' does Not exist')

    num_images = len(gt_imgs)
    gt_patches = [img_crop(gt_imgs[i], 16, 16) for i in range(num_images)]
    data = numpy.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    labels = numpy.asarray([value_to_class(numpy.mean(data[i]),th) for i in range(len(data))])

    # Convert to dense 1-hot representation.
    return labels.astype(numpy.float32)




def compute_FScore(gt_folder,prediction_folder, num_image,th):
    labels = extract_labels(gt_folder, num_image)
    predictions = extract_labels_for_validation(prediction_folder, num_image,th)
    
    # arrays with position of given number
    id_true_label = numpy.where(labels[:,0]==1)
    id_false_label = numpy.where(labels[:,0]==0)
    id_true_prediction = numpy.where(predictions[:,0]==1)
    id_false_prediction = numpy.where(predictions[:,0]==0)
    # TP = T + P where P = Positive, T = True and so on
    TP = (numpy.isin(id_true_prediction,id_true_label)== True).sum()
    FP = (numpy.isin(id_true_prediction,id_false_label)== True).sum()
    TN = (numpy.isin(id_false_prediction,id_false_label)== True).sum()
    FN = (numpy.isin(id_false_prediction,id_true_label)== True).sum()
    #print(TP,FP,TN,FN)
    precision = (TP )/(TP + FP)
    recall = (TP)/(TP + FN)
    Fscore = 2*precision*recall/(precision+recall)
    # print(precision, recall, Fscore)
    return Fscore




def compute_accuracy(gt_folder,prediction_folder, num_image,th):
    labels = extract_labels(gt_folder, num_image);
    predictions = extract_labels_for_validation(prediction_folder, num_image,th);
    cont=0;
    for i in range(0,len(labels[:,1])):
        if(labels[i,1]==predictions[i,1]):
                cont=cont+1;
    accuracy=cont/len(labels);
    return accuracy

def cross_validation_on_thresh_per_pixel(gt_folder,prediction_folder, num_image,th):
    print("Starting Cross Validation on the threshold"); 
    
    
    accuracies=np.zeros(len(th));
    i=0;
    acc_Max=0;
    for single_th in th:
        accuracies[i]=compute_FScore(gt_folder,prediction_folder, num_image,single_th);
        print("Threshold:")
        print(single_th)
        print("F-score:")
        print(accuracies[i]);
        
        if(accuracies[i]>acc_Max):
            acc_Max=accuracies[i];
            best_th=single_th;
            
        i=i+1;
       
    
    
    return best_th
        
        
    
