#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 18:13:29 2018

@author: chongduan
"""

import tensorflow as tf
import numpy as np
import cv2
import os
from utils import subtract_mean, boxes_to_bigbox
from utils_graph import conv_layer, flatten_tensor, fc_layer
from models import create_VGG16_based_model

"""    
Build Single Digit Detection Model using pre-trained VGG16. 
"""
tf.keras.backend.clear_session()

model = create_VGG16_based_model(pretrained = True)

checkpoint_path = "./checkpoints-pre-trained-VGG16/cp.ckpt"

model.load_weights(checkpoint_path)


"""
Prepare to load video
"""
path = './data/test_video_processed.avi'
cap = cv2.VideoCapture(path)

# Frames to process and save
frames = [0, 30, 55, 235, 270]

# Create output directory if it does not exist
save_dir = './graded_images'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Initialize frame_num
frame_num = 0
while(cap.isOpened()):
    
    ret, frame = cap.read()
    
    if not ret:
        break
    
    if frame_num in frames:
        
        # Print progress
        print("Processing frame {} ...".format(frames.index(frame_num)))
        
        ### Evaluate the model with sliding window
        input_image = cv2.resize(frame, dsize=(0,0), fx=0.5, fy=0.5)
        test_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        
        #images = image_pyramid(test_image, 4)
        
        height, width = test_image.shape
        windows = []
        windows_pos = []
        for i in range(0,height-32, 8):
            for j in range(0,width-32, 8):
                windows.append(test_image[i:i+32, j:j+32])
                windows_pos.append([j,i,j+32, i+32])
        
        windows_pos = np.stack(windows_pos)
        windows = np.expand_dims(np.stack(windows, axis=0).astype(np.float32),axis=3)
        windows = subtract_mean(windows)
        windows_3ch = np.concatenate((windows, windows, windows), axis = 3)
        pred = model.predict(windows_3ch)
        pred_num = np.argmax(pred, axis=1)
        pred_num_prob = np.max(pred, axis=1)
        idx = np.logical_and(pred_num != 10, pred_num_prob > 0.5)

        # Select the windows with digits identified, and its associated probilities
        probs = pred_num_prob[idx]
        boxes = windows_pos[idx,]
        
        # sort the idx based on the probabilities, descending order
        idx_sorted = sorted(range(len(probs)), key = lambda x: probs[x], reverse=True)
        
        # Only take the first ten highest probablity boxes
        if len(idx_sorted) > 10:
            idx_sorted = idx_sorted[:10]
        boxes = boxes[idx_sorted,]
        
        # Get the digits region
        x1, y1, x2, y2 = boxes_to_bigbox(boxes)
        bigbox_image = cv2.resize(test_image[y1:y2, x1:x2], dsize=(32,32),
                                  interpolation = cv2.INTER_CUBIC)
        
        """
        Build Multi-Digit Recognization Model using tensorflow directly
        """
        # Reset graph to avoid "variable already exists" error
        tf.reset_default_graph()
        
        # Get data dimensions
        img_height, img_width, num_channels = 32, 32, 1
        
        # Get label information
        num_digits, num_labels = 5, 11
        
        ## Some hyperparameters for the model
        # Block 1
        filter_size1 = filter_size2 = 5          
        num_filters1 = num_filters2 = 32        
        
        # Block 2
        filter_size3 = filter_size4 = 5          
        num_filters3 = num_filters4 = 64
        
        # Block 3
        filter_size5 = filter_size6 = filter_size7 = 5          
        num_filters5 = num_filters6 = num_filters7 = 128  
        
        # Fully-connected layers
        fc1_size = fc2_size = 256
        
        ### Initialize tensors for the graph
        with tf.name_scope("input"):
            
            # Placeholders for feeding input images
            x = tf.placeholder(tf.float32, shape=(None, img_height, img_width, num_channels), name='x')
            y_ = tf.placeholder(tf.int64, shape=[None, num_digits], name='y_')
        
        with tf.name_scope("dropout"):
            
            # Dropout rate applied to the input layer
            p_keep_1 = tf.placeholder(tf.float32)
            tf.summary.scalar('input_keep_probability', p_keep_1)
        
            # Dropout rate applied after the pooling layers
            p_keep_2 = tf.placeholder(tf.float32)
            tf.summary.scalar('conv_keep_probability', p_keep_2)
        
            # Dropout rate using between the fully-connected layers
            p_keep_3 = tf.placeholder(tf.float32)
            tf.summary.scalar('fc_keep_probability', p_keep_3)
              
        ### Stack layers to build the graph/model
        # Apply dropout to the input layer
        drop_input = tf.nn.dropout(x, p_keep_1) 
        
        # Block 1
        conv_1 = conv_layer(drop_input, filter_size1, num_channels, num_filters1, "conv_1", pooling=False)
        conv_2 = conv_layer(conv_1, filter_size2, num_filters1, num_filters2, "conv_2", pooling=True)
        drop_block1 = tf.nn.dropout(conv_2, p_keep_2) # Dropout
        
        # Block 2
        conv_3 = conv_layer(conv_2, filter_size3, num_filters2, num_filters3, "conv_3", pooling=False)
        conv_4 = conv_layer(conv_3, filter_size4, num_filters3, num_filters4, "conv_4", pooling=True)
        drop_block2 = tf.nn.dropout(conv_4, p_keep_2) # Dropout
        
        # Block 3
        conv_5 = conv_layer(drop_block2, filter_size5, num_filters4, num_filters5, "conv_5", pooling=False)
        conv_6 = conv_layer(conv_5, filter_size6, num_filters5, num_filters6, "conv_6", pooling=False)
        conv_7 = conv_layer(conv_6, filter_size7, num_filters6, num_filters7, "conv_7", pooling=True)
        flat_tensor, num_activations = flatten_tensor(tf.nn.dropout(conv_7, p_keep_3)) # Dropout
        
        # Fully-connected 1
        fc_1 = fc_layer(flat_tensor, num_activations, fc1_size, 'fc_1', relu=True)
        drop_fc2 = tf.nn.dropout(fc_1, p_keep_3) # Dropout
        
        # Fully-connected 2
        fc_2 = fc_layer(drop_fc2, fc1_size, fc2_size, 'fc_2', relu=True)
        
        # Paralell softmax layers
        logits_1 = fc_layer(fc_2, fc2_size, num_labels, 'softmax1')
        logits_2 = fc_layer(fc_2, fc2_size, num_labels, 'softmax2')
        logits_3 = fc_layer(fc_2, fc2_size, num_labels, 'softmax3')
        logits_4 = fc_layer(fc_2, fc2_size, num_labels, 'softmax4')
        logits_5 = fc_layer(fc_2, fc2_size, num_labels, 'softmax5')
        
        # Prediction
        y_pred = [logits_1, logits_2, logits_3, logits_4, logits_5]
        
        # The class-number is the index of the largest element
        y_pred_cls = tf.transpose(tf.argmax(y_pred, axis=2))
        
        with tf.name_scope('loss'):
            
            # Calculate the loss for each individual digit in the sequence
            loss1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_1, labels=y_[:, 0]))
            loss2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_2, labels=y_[:, 1]))
            loss3 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_3, labels=y_[:, 2]))
            loss4 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_4, labels=y_[:, 3]))
            loss5 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_5, labels=y_[:, 4]))
        
            # Calculate the total loss for all predictions
            loss = loss1 + loss2 + loss3 + loss4 + loss5
            tf.summary.scalar('loss', loss)
            
        
        ### Optimization Method
        with tf.name_scope('optimizer'):
            
            # Global step is required to compute the decayed learning rate
            global_step = tf.Variable(0, trainable=False)
        
            # Apply exponential decay to the learning rate
            learning_rate = tf.train.exponential_decay(1e-3, global_step, 7500, 0.5, staircase=True)
        
            # Construct a new Adam optimizer
            optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss, global_step=global_step)
        
        ### Classification Method
        with tf.name_scope("accuracy"):
            
            # Predicted class equals the true class of each image
            # If one digit is wrong, the whole class is wrong (see tf.reduce_min)
            correct_prediction = tf.reduce_min(tf.cast(tf.equal(y_pred_cls, y_), tf.float32), 1)
        
            # Cast predictions to float and calculate the mean
            accuracy = tf.reduce_mean(correct_prediction) * 100.0
            
            # Add scalar summary for accuracy tensor
            tf.summary.scalar('accuracy', accuracy)
        
        ### Tensorflow Run
        # Launch the graph in a session
        session = tf.Session()
        
        saver = tf.train.Saver()
        
        # Restore the trained model
        print("Restoring last checkpoint ...")
        
        # Finds the filename of latest saved checkpoint file
        ckpt = tf.train.get_checkpoint_state('./checkpoints/')
        
        # Load the weights in the checkpoint.
        saver.restore(session, ckpt.model_checkpoint_path)
        print("Restored checkpoint from:", ckpt.model_checkpoint_path)
        
        ### Evaluate Model
        test_image_for_model = np.expand_dims(np.expand_dims(bigbox_image, axis=0), axis=3)
        test_image_for_model = test_image_for_model.astype(np.float32)
        test_image_for_model = subtract_mean(test_image_for_model)
        
        pred, logits = session.run([y_pred_cls, y_pred], feed_dict={x: test_image_for_model,
                                                  p_keep_1: 1.,
                                                  p_keep_2: 1.,
                                                  p_keep_3: 1.})
        
        # Get a string of the pred nums
        pred_number = ''.join(str(x) for x in pred[0] if x != 10)
        img_out = input_image.copy()
        cv2.rectangle(img_out,(x1,y1),(x2, y2),(0,255,0),3)
        cv2.putText(img_out,
                    'Pred: {}'.format(pred_number),
                    (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), lineType=cv2.LINE_AA)
        
        ### Write image
        cv2.imwrite('./graded_images/{}.png'.format(frames.index(frame_num)), img_out)
    
    ### Update frame number
    frame_num += 1

cap.release()
cv2.destroyAllWindows()