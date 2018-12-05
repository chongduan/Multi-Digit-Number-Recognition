#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 13:48:19 2018

@author: chongduan
"""

import numpy as np
import cv2

path = '../data/test_video.MOV'
cap = cv2.VideoCapture(path)

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create VideoWriter object.
out = cv2.VideoWriter('../data/test_video_processed.avi',
                      cv2.VideoWriter_fourcc('M','J','P','G'),
                      30,
                      (frame_width,frame_height))

frame_num = 1 
while(True):
    ret, frame = cap.read()
    if ret == True:
        # Add noise
        if frame_num >= 25 and frame_num < 35:
            frame = np.double(frame) + np.random.normal(0, 30, (frame_height, frame_width, 3))
            frame = frame.astype(np.uint8)
            
        # Darker Light
        if frame_num >= 55 and frame_num < 65:
            frame = (np.double(frame) * 0.5).astype(np.uint8)


        # Write the frame into the file 'output.avi'
        out.write(frame)
        # Display the resulting frame    
        cv2.imshow('frame',frame)
         
        # Press Q on keyboard to stop recording
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break
    # Break the loop 
    else:
        break
    
    # Update frame num
    frame_num = frame_num + 1
 
# When everything done, release the video capture and video write objects
cap.release()
out.release()

# Closes all the frames
cv2.destroyAllWindows() 
 