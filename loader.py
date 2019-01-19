# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
import glob

def encoder(data):
    ncols = 20
    labels_one_hot = (data.ravel()[np.newaxis] == np.arange(ncols)[:, np.newaxis]).T
    labels_one_hot.shape = data.shape + (ncols,)
    return labels_one_hot

def load_data():
    """
    Loads data into numpy arrays ready for Keras training / testing
    Returns: Tuple with train and test datasets
    """
    
    X_train = []
    y_train = []
    
    for filename in sorted(glob.glob('data/train/image/*.jpg')): #assuming gif
        im = np.asarray(Image.open(filename).resize((128,128)))
        X_train.append(im)
    
    for filename in sorted(glob.glob('data/train/gt/*.png')): #assuming gif
        im = np.asarray(Image.open(filename).resize((128,128)))
        y_train.append(im)
        
    X_test = []
    y_test = []
    
    for filename in sorted(glob.glob('data/test/image/*.jpg')): #assuming gif
        im = np.asarray(Image.open(filename).resize((128,128)))
        X_test.append(im)
    
    for filename in sorted(glob.glob('data/test/gt/*.png')): #assuming gif
        im = np.asarray(Image.open(filename).resize((128,128)))
        y_test.append(im)
        
    return (np.array(X_train), encoder(np.array(y_train)), np.array(X_test), encoder(np.array(y_test)))
        
        

