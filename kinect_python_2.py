import ctypes
import enum
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import tools_kinect 
import time
import cv2
import scipy.io as sio
d, calibration = tools_kinect.initialise_device_with_calibration()
im = tools_kinect.get_image(d)



N = 300
print('TAKE DATA')
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

color_img_full = color_img_full[50:, 325:925,:3,:]
transformed_depth_img_full = transformed_depth_img_full[50:, 325:925,:]
depth_img_full = depth_img_full[51:525, 105:535,:]

# # # # # Processing
print('PROCESSING')
color_img_resized = np.empty((223,200,3,N))
transformed_depth_img_resized =  np.empty((223,200,N))
for index_image in range(N):
    color_img_resized[:,:,:,index_image] = cv2.resize(color_img_full[:,:,:,index_image], fx=3,fy=3,  dsize=(200, 223), interpolation= cv2.INTER_NEAREST)
    transformed_depth_img_resized[:,:,index_image] = cv2.resize(transformed_depth_img_full[:,:,index_image], fx=3,fy=3,  dsize=(200, 223), interpolation= cv2.INTER_NEAREST)
    

    


fig, axs = plt.subplots(1, 2)
axs[0].imshow(transformed_depth_img_resized[:,:,0])
axs[1].imshow(color_img_resized[:,:,0])
plt.show()

print('SAVE')
name = '3D_proof_of_concept_'
dictio = {}
dictio['transformed_depth_img_full'] = transformed_depth_img_resized
sio.savemat(name+'depth_transformed.mat', dictio)
dictio = {}
dictio['depth'] = depth_img_full
sio.savemat(name+'depth.mat', dictio)
dictio = {}
dictio['colour'] = color_img_resized
sio.savemat(name+'color.mat', dictio)
tools_kinect.k4a_device_close(d)

