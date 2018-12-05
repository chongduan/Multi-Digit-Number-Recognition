#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 23:34:54 2018

@author: chongduan
"""

import cv2
import numpy as np
import os

"""
Figure
"""
images = []
for root, dirs, files in os.walk("./graded_images"):  
    for filename in files:
        print(filename)
        images.append(cv2.imread(os.path.join('./graded_images', filename)))

# Create montage
img_out = np.zeros((1080,2880,3))
img_out[:540, :960, :] = images[0]
img_out[:540, 960:1920, :] = images[1]
img_out[:540, 1920:, :] = images[2]
img_out[540:, :960, :] = images[3]
img_out[540:, 960:1920, :] = images[4]

cv2.imwrite('final_images.png', img_out)


### Negative samples
image1 = cv2.imread('negative_218.png')
image2 = cv2.imread('negative_237.png')
image_out = np.zeros((540, 1920,3))
image_out[:,:960,:] = image1
image_out[:,960:,:] = image2
cv2.imwrite('negative.png', image_out)




