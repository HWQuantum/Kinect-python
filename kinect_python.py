import ctypes
import enum
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import tools_kinect
import time
import scipy.io as sio
d = tools_kinect.initialise_device()
im = tools_kinect.get_image(d)

# fig, axs = plt.subplots(1, 2)
# depth = im.get_depth_array()
# depth_cropped = depth[51:525,105:535]
# axs[0].imshow(depth_cropped)
# axs[1].imshow(im.get_colour_array())
# plt.show()

N = 10

depth_img_full = np.empty((474,430,N))
color_img_full = np.empty((720,1280,4,N))
for index_image in range(N):
    im= tools_kinect.get_image(d)
    # print(im.get_depth_array().shape)
    depth = im.get_depth_array()
    depth_cropped = depth[51:525,105:535]
    depth_img_full[:,:,index_image] = depth_cropped
    # print(depth_cropped.shape)
    colour = im.get_colour_array()
    color_img_full[:,:,:,index_image] = colour
    # print(im.get_colour_array().shape)
    
name = 'first_test.mat'
dictio = {}
dictio['depth'] = depth_img_full
dictio['colour'] = color_img_full
sio.savemat(name, dictio)
tools_kinect.k4a_device_close(d)

