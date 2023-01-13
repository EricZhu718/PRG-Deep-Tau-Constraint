from copy import deepcopy
import os
import string
import cv2 as cv
import numpy as np
import glob
from interpolate import get_interpolation
import pandas as pd
import matplotlib.pyplot as plt
import copy
import os

from reference_frame import get_info_from_frame

import argparse
import copy
import time
from pathlib import Path
import numpy as np

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from tqdm import tqdm

import sys

def undistort_images(data_path):
    os.makedirs(data_path + '/processed_images/', exist_ok=True)
    contents = open(data_path + '/dso/cam0/camera.txt', "r").read().split(' ')
    contents[8] = contents[8].split('\n')[0]
    camera_params = contents[1:9]
    camera_params = [float(val) for val in camera_params]
    distCoeff = np.array([camera_params[-4], camera_params[-3], camera_params[-2], camera_params[-1]])

    images = list(glob.glob(data_path + '/dso/cam0/images/*.png'))    
    def sort_key(image_name):
        return int(image_name.split('.')[0].split('/')[-1])
    images.sort(key=sort_key)

    for i, image in tqdm(enumerate(images), total=len(images)):
        image = cv.imread(image, 0)
        height,width = image.shape
        camera_matrix = np.array([[width*camera_params[0] , 0 , width*camera_params[2]-0.5] , [0 ,height*camera_params[1], height*camera_params[3]-0.5] , [0,0,1] ])
        size_factor = 1
        K_new = np.copy(camera_matrix)
        K_new[0, 2] *= size_factor
        K_new[1, 2] *= size_factor
        undistorted = cv.fisheye.undistortImage(image, camera_matrix, distCoeff,Knew=K_new,new_size=(size_factor*width, size_factor*height))

        # cv.imshow('distorted',image)
        # cv.imshow('undistorted', undistorted)
        # cv.waitKey(1)

        img_name = ('0' * (5-len(str(i)))) + str(i) + '.png'
        cv.imwrite(os.path.join(data_path, 'processed_images', 'undistorted_images', img_name), undistorted)

def run_yolo(data_path):
    print(data_path)

    print('\nStarting yolo\n')
    os.system('python3 detect2.py \
               --weights object_detector_weights/yolov7.pt \
               --source ' + os.path.join(data_path, 'dso', 'cam0', 'undistorted_images'))
    print('\nDone with yolo\n')

if __name__ == '__main__':
    data_path = 'data/monocular_data/dataset-corridor1_512_16'
    undistort_images(data_path)
    run_yolo(data_path)
