import cv2 as cv
import numpy as np
import os
import glob

#equidistant_params = [fx fy cx cy k1 k2 k3 k4] #from camera.txt file in dataset
p = [0.373004838186, 0.372994740336, 0.498890050897, 0.502729380663, 0.00348238940225, 0.000715034845216, -0.00205323614187, 0.000202936735918]

distCoeff = np.array([[0.00348238940225, 0.000715034845216, -0.00205323614187, 0.000202936735918]])

directory = '/home/tau/Desktop/monocular_data/dataset-corridor1_512_16/dso/cam0/undistorted_images/'

class MyImage:
    def __init__(self, img_name):
        self.img = cv.imread(img_name)
        self.__name = img_name

    def __str__(self):
        return self.__name

images = []

for image in glob.glob("/home/tau/Desktop/monocular_data/dataset-corridor1_512_16/dso/cam0/images/*.png"):

    images.append(image)

images.sort()

i = 0

for image in images:

    image_1 = cv.imread(image, 0)

    height,width = image_1.shape

    camera_matrix = np.array([[width*p[0] , 0 , width*p[2]-0.5] , [0 ,height*p[1], height*p[3]-0.5] , [0,0,1] ])

    size_factor = 1
    K_new = np.copy(camera_matrix)
    K_new[0, 2] *= size_factor
    K_new[1, 2] *= size_factor
    
    output = cv.fisheye.undistortImage(image_1, camera_matrix,distCoeff,Knew=K_new,new_size=(size_factor*width, size_factor*height))

    cv.imshow('original', image_1)
    cv.imshow('undistorted',output)
    cv.imwrite(directory + str(i) +".png" , output)
    i+=1
    cv.waitKey(10)
  
