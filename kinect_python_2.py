import ctypes
import enum
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import tools_kinect 
import time
import scipy.io as sio
d, calibration = tools_kinect.initialise_device_with_calibration()
im = tools_kinect.get_image(d)



N = 100

depth_img_full = np.empty((576,640,N)) #np.empty((474,430,N))
color_img_full = np.empty((720,1280,4,N))
transformed_depth_img_full =  np.empty((720,1280,N))
time_begin = time.time()
for index_image in range(N):
    print(index_image)
    im= tools_kinect.get_image(d)
    transformed_depth_array = tools_kinect.transform_depth_image_to_colour(calibration, im.depth_image)
    # print(im.get_depth_array().shape)
    depth = im.get_depth_array()
    colour = im.get_colour_array()
    #depth_cropped = depth[51:525,105:535]
    transformed_depth_img_full[:,:,index_image] = transformed_depth_array
    depth_img_full[:,:,index_image] = depth
    # print(depth_cropped.shape)
    color_img_full[:,:,:,index_image] = colour
    # print(im.get_colour_array().shape)

time_end = time.time()
print( "processing time="+str(time_end-time_begin))
fig, axs = plt.subplots(1, 2)
axs[0].imshow(colour)
axs[1].imshow(depth)
plt.show()

name = 'first_test.mat'
dictio = {}
dictio['transformed_depth_img_full'] = transformed_depth_img_full
dictio['depth'] = depth_img_full
dictio['colour'] = color_img_full
sio.savemat(name, dictio)
tools_kinect.k4a_device_close(d)

