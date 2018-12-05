#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 17:58:33 2018

@author: chongduan
"""
import numpy as np
import os
import cv2
from scipy.io import loadmat
from matplotlib import pyplot as plt

def load_data(path):
    # load the 32x32 images from SVHN
    temp_train = loadmat(os.path.join(path, "train_32x32.mat"))
    X_train = np.transpose(temp_train['X'], (3,0,1,2))
    y_train = temp_train['y'][:,0]
    temp_test = loadmat(os.path.join(path, "test_32x32.mat"))
    X_test = np.transpose(temp_test['X'], (3,0,1,2))
    y_test = temp_test['y'][:,0]
    temp_extra = loadmat(os.path.join(path, "extra_32x32.mat"))
    X_extra = np.transpose(temp_extra['X'], (3,0,1,2))
    y_extra = temp_extra['y'][:,0]    
    
    return (X_train, y_train, X_test, y_test, X_extra, y_extra)


def plot_images(img, labels, nrows, ncols, pred_labels=None):
    """
    Plot nrows x ncols example images
    """
    
    # Check labels are one hot encoding or not
    if len(labels.shape) == 2:
        labels = np.argmax(labels, axis=1)
    
    # Randomly permute img and labels
    idx = np.random.permutation(img.shape[0])
    img = img[idx,:,:,:]
    labels = labels[idx]
    if pred_labels is not None:     
        pred_labels = pred_labels[idx] # Keep the same permutation

    fig, axes = plt.subplots(nrows, ncols)
    for i, ax in enumerate(axes.flat):
        
        if pred_labels is not None:
            title = "True: {0}, Pred: {1}".format(labels[i], pred_labels[i])
        else:
            title = "True: {}".format(labels[i])
        
        ax.imshow(np.squeeze(img[i]))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)


def balanced_subsample(y, num_samples):
    """
    Return subsample indices with balanced class distribution
    
    Arg: 
        y:              labels
        num_samples:    num of samples for each label
    
    Return:
        idx:            indices for the balanced subsample
    """
    
    idx = []
    # For every label in the dataset
    for label in np.unique(y):
        # Get the index of all images with a specific label
        idx_all = np.where(y==label)[0]
        # Draw a random sample from the images
        random_sample = np.random.choice(idx_all, size=num_samples, replace=False)
        # Add the random sample to our subsample list
        idx += random_sample.tolist()
    
    return idx


def plot_images_multi_digit(images, nrows, ncols, cls_true, cls_pred=None):
    
    # Initialize figure
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 2*nrows))
    
    # Randomly select nrows * ncols images
    rs = np.random.choice(images.shape[0], nrows*ncols)
    
    # For every axes object in the grid
    for i, ax in zip(rs, axes.flat): 
        
        # Pretty string with actual number
        true_number = ''.join(str(x) for x in cls_true[i] if x != 10)
        
        if cls_pred is None:
            title = "True: {0}".format(true_number)
        else:
            # Pretty string with predicted number
            pred_number = ''.join(str(x) for x in cls_pred[i] if x != 10)
            title = "True: {0}, Pred: {1}".format(true_number, pred_number) 
            
        ax.imshow(images[i,:,:,0], cmap='binary')
        ax.set_title(title)   
        ax.set_xticks([]); ax.set_yticks([])


def rgb2gray(images):
    """
    Y = 0.2989R + 0.5870G + 0.1140B
    """
    return np.expand_dims(np.dot(images, [0.2989, 0.5870, 0.1140]), axis=3)    
    

def subtract_mean(images):
    """ Helper function for subtracting the mean of every image
    """
    for i in range(images.shape[0]):
        images[i] -= images[i].mean()
    
    return images

def image_pyramid(image, n):
    """
    Helper function for generating image pyramid
    """
    
    img = image.copy()
    res = []
    res.append(img)
    for i in range(n-1):
        img = cv2.pyrDown(img)
        res.append(img)
    
    return res
 
def non_max_suppression(boxes, overlapThresh):
    """
    Based on Malisiewicz et al.
    
    Input:
        boxes: 2-d numpy array, each row is [x1, y1, x2, y2]
        overlapThresh:  threshold for overlapping boxes
        
    Output:
        boxes after non-max-suppression
    """
	# if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
 
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
 
	# initialize the list of picked indexes	
    pick = []
 
	# grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
 
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
 
	# keep looping while some indexes still remain in the indexes
	# list
    while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
 
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
		# compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
 
		# compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
 
		# delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],np.where(overlap > overlapThresh)[0])))
 
	# return only the bounding boxes that were picked using the
	# integer data type
    return boxes[pick].astype("int"), pick


def boxes_to_bigbox(boxes):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
 
	# grab the coordinates of the bounding boxes
    x1 = np.min(boxes[:,0])
    y1 = np.min(boxes[:,1])
    x2 = np.max(boxes[:,2])
    y2 = np.max(boxes[:,3])
    
    return (x1, y1, x2, y2)




