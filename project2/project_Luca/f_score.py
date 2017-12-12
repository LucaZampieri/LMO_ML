"""File with funtion for computing the F-Score """

# Extract label images
def extract_labels_for_validation(foldername, num_images):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    gt_imgs = []
    for i in range(1, num_images+1):
        imageid = str('predicted_groundtruth_') + str(i)
        image_filename = foldername + imageid + ".png"
        if os.path.isfile(image_filename):
            print ('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print ('File ' + image_filename + ' does Not exist')

    num_images = len(gt_imgs)
    gt_patches = [img_crop(gt_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]
    data = numpy.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    labels = numpy.asarray([value_to_class(numpy.mean(data[i])) for i in range(len(data))])

    # Convert to dense 1-hot representation.
    return labels.astype(numpy.float32)


def compute_FScore(gt_folder,prediction_folder, num_image):
    labels = extract_labels(gt_folder, num_images)
    predictions = extract_labels_for_validation(prediction_folder, num_image)
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


def conv_layer(self, input,w,b, activation=relu, name='conv'): # channels_in,channels_out
    with tf.name_scope(name):
        conv = tf.nn.conv2d(input,w,strides=[1,1,1,1],padding="SAME")
        if activation = leaky_relu:
            layer = tf.nn.leaky_relu(conv + b, alpha=0.2, name=None)
        else:
            layer = tf.nn.relu(conv + b)
        #####tf.nn.relu(tf.nn.bias_add(conv, b))
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        #tf.summary.histogram("activations", act)
        return layer
