import ctypes
import enum
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import tools_kinect_2 as tools_kinect
import time
import scipy.io as sio

d, calibration = tools_kinect.initialise_device_with_calibration()
im = tools_kinect.get_image(d)

 

# Create empty array for depth and RGB 
# -> I am not sure of the sizes of the images, Run first without saving to know the shapes. 
s_x_depth = 474
s_y_depth = 430
s_x_rgb = 720
s_y_rgb = 1280
depth_img_full = np.empty((s_x_depth,s_y_depth,N)) # Depth  
color_img_full = np.empty((s_x_rgb,s_y_rgb,4,N)) #RGB  
# depth_img_full_transformed = np.empty((720,1280,N)) # Depth compensated for deformation of the lens. This takes time and is not always necessary. 
#  I think you need GPU for it. I suggest to try first without. 


N = 10 #number of frames to take/save
for index_image in range(N):
    im= tools_kinect.get_image(d)
    # get depth and RGB image
    depth = im.get_depth_array()
    print(depth.shape)
    colour = im.get_colour_array()
    # transformed_depth_array = tools_kinect.transform_depth_image_to_colour(calibration, im.depth_image)

    depth_img_full[:,:,index_image] = depth
    color_img_full[:,:,:,index_image] = colour

    
name = 'array.mat'
dictio = {}
dictio['depth'] = depth_img_full
dictio['colour'] = color_img_full
sio.savemat(name, dictio)
tools_kinect.k4a_device_close(d)

