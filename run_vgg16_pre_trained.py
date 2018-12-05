#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 17:09:17 2018

Noted to TAs:

This scripts requires some processed SVHN data, which are not included
in the submission due to space limit

@author: chongduan
"""
import tensorflow as tf
import numpy as np
import h5py
from utils import subtract_mean, image_pyramid, non_max_suppression, boxes_to_bigbox
from sklearn.metrics import confusion_matrix
import cv2
import os

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions


from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

from sklearn.metrics import accuracy_score
from models import create_VGG16_based_model

### Prepare data
# Load data: Single Digit from SVHN
h5f = h5py.File('../data/SVHN_single_digit.h5', 'r')

# Load the training, test and validation set
X_train = h5f['X_train'][:]
y_train = h5f['y_train'][:]
X_test = h5f['X_test'][:]
y_test = h5f['y_test'][:]
X_val = h5f['X_val'][:]
y_val = h5f['y_val'][:]

# Subtract the mean from every image
X_train = subtract_mean(X_train)
X_test = subtract_mean(X_test)
X_val = subtract_mean(X_val)

# Format to 3 channels
X_train_3ch = np.concatenate((X_train, X_train, X_train), axis = 3)
X_test_3ch = np.concatenate((X_test, X_test, X_test), axis = 3)
X_val_3ch = np.concatenate((X_val, X_val, X_val), axis = 3)


# Start Fine-tuning
tf.keras.backend.clear_session()
model = create_VGG16_based_model(pretrained = True)

# Create checkpoint callback
checkpoint_path = "../checkpoints-VGG16-pre-trained/cp.ckpt"
#checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                 save_weights_only=True,
                                                 verbose=1)

# Create early stopping callback
stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')


# Fit model
history = model.fit(X_train_3ch,y_train,
          batch_size=128, epochs=50,
          shuffle=True, verbose=1,
          validation_data=(X_val_3ch, y_val),
          callbacks = [cp_callback, stop_callback])


## summarize history for accuracy
#plt.figure(figsize=(8,4))
#plt.subplot(1,2,2)
#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.title('Model 2: accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'validation'], loc='upper left')
## summarize history for loss
#plt.subplot(1,2,1)
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('Model 2: loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'validation'], loc='upper left')
#
#plt.tight_layout()
#plt.savefig('model2', dpi=300)
#plt.show()



## Make predictions
#y_test_pred = model.predict(X_test_3ch, batch_size=128, verbose=1)
#
#
#### Accuracy_score
#accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_test_pred, axis=1))
#print('Classification Accuracy: {:0.2f}%'.format(accuracy*100))
#
#### Confusion matrix
#cm = confusion_matrix(y_true=np.argmax(y_test, axis=1), y_pred=np.argmax(y_test_pred, axis=1))
#
## Normalize the confusion matrix
#cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100.0
#
## Visualize the confusion matrix
#plt.figure(figsize=(12, 8))
#sns.heatmap(cm, annot=True, cmap='Reds', fmt='.1f', square=True)

#### Build a fresh model, and load the weights
#model = create_VGG16_based_model(pretrained = True)
#model.load_weights(checkpoint_path)
#
## Make predictions again
#y_test_pred = model.predict(X_test_3ch, batch_size=128, verbose=1)
#
#### Plot confusion matrix
#plt.figure(figsize=(12, 8))
## Calculate the confusion matrix
#cm = confusion_matrix(y_true=np.argmax(y_test, axis=1), y_pred=np.argmax(y_test_pred, axis=1))
#
## Normalize the confusion matrix
#cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100.0
#
## Visualize the confusion matrix
#sns.heatmap(cm, annot=True, cmap='Reds', fmt='.1f', square=True)

#### Evaluate the model with sliding window
#path = "../data/test_image_from_video.png"
#test_image = cv2.resize(cv2.imread(path, 0), dsize=(600,400))
#plt.imshow(test_image,cmap='gray')
#plt.show()
##images = image_pyramid(test_image, 4)
##test_image = images[1]
#height, width = test_image.shape
#boxes = []
#pred_nums = []
#for i in range(0,height-32,4):
#    for j in range(0,width-32,4):
#        test_image_for_model = np.expand_dims(np.expand_dims(test_image[i:i+32, j:j+32], axis=0), axis=3)
#        test_image_for_model = test_image_for_model.astype(np.float32)
#        test_image_for_model = subtract_mean(test_image_for_model)
#        test_3ch = np.concatenate((test_image_for_model, test_image_for_model, test_image_for_model), axis = 3)
#        
#        test_pred = model.predict(test_3ch)
#        
#        num_pred = np.argmax(test_pred[0])
#        pred_prob = np.max(test_pred[0])
#        
#        if num_pred != 10 and pred_prob > 0.9:
#            boxes.append([j,i,j+32, i+32])
#            pred_nums.append(str(num_pred))
#
#boxes = np.stack(boxes, axis=0)
#img_out = test_image.copy()
#for idx, box in enumerate(boxes):
#    cv2.rectangle(img_out,(box[0],box[1]),(box[2], box[3]),(0,255,0),3)
#    cv2.putText(img_out,
#                pred_nums[idx],
#                (box[0],box[1]),
#                cv2.FONT_HERSHEY_SIMPLEX,
#                1.0, (0, 255, 0), lineType=cv2.LINE_AA)
#plt.figure(figsize=(12, 8))
#plt.imshow(img_out)
#plt.show()
#
#x1, y1, x2, y2 = boxes_to_bigbox(boxes)
#
#bigbox_image = cv2.resize(test_image[y1:y2, x1:x2], dsize=(32,32),
#                          interpolation = cv2.INTER_CUBIC)
#plt.imshow(bigbox_image)
#plt.show()
#
#boxes_sel, pick = non_max_suppression(boxes, 0.5)
#pred_nums_sel = [pred_nums[i] for i in pick]
#img_out = test_image.copy()
#for idx, box in enumerate(boxes_sel):
#    cv2.rectangle(img_out,(box[0],box[1]),(box[2], box[3]),(0,255,0),3)
#    cv2.putText(img_out,
#                pred_nums_sel[idx],
#                (box[0],box[1]),
#                cv2.FONT_HERSHEY_SIMPLEX,
#                1.0, (0, 255, 0), lineType=cv2.LINE_AA)
#plt.figure(figsize=(12, 8))
#plt.imshow(img_out)
#plt.show()