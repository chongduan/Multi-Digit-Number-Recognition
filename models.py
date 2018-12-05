#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 18:15:20 2018

@author: chongduan
"""

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD


### Model
def create_VGG16_based_model(pretrained = True):
    
    # load a VGG network
    if pretrained:
        vgg16 = VGG16(weights='imagenet', include_top=False)
    else:
        vgg16 = VGG16(weights=None, include_top=False)

    # Create custmized input
    input_layer = Input(shape=(32,32,3), name = 'image_input')
    vgg16_output = vgg16(input_layer)
    
    #Add the fully-connected layers 
    flatten_layer = Flatten(name='flatten')(vgg16_output)
    fc1 = Dense(4096, activation='relu', name='fc1')(flatten_layer)
    fc2 = Dense(4096, activation='relu', name='fc2')(fc1)
    output = Dense(11, activation='softmax', name='predictions')(fc2)
    
    #Create your own model 
    my_model = Model(input_layer, output)

#    if pretrained:
#        #Set the VGG layers to non-trainable (weights will not be updated)
#        for layer in my_model.layers[:2]:
#            layer.trainable = False
    
    # Learning rate is set to be 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    my_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return my_model
        
        