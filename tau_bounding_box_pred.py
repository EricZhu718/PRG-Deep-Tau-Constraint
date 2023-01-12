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

import sys

sys.path.append('yolov7')

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


def undistort(path:string):
    os.makedirs(path + '/processed_images/', exist_ok=True)
    contents = open(path + '/dso/cam0/camera.txt', "r").read().split(' ')
    contents[8] = contents[8].split('\n')[0]
    camera_params = contents[1:9]
    # print(camera_params)
    camera_params = [float(val) for val in camera_params]
    distCoeff = np.array([camera_params[-4], camera_params[-3], camera_params[-2], camera_params[-1]])
    images = []
    for image in glob.glob(path + '/dso/cam0/images/*.png'):
        images.append(image)
    
    def sort_key(image_name):
        return int(image_name.split('.')[0].split('/')[-1])
    
    images.sort(key=sort_key)

    frame_times = pd.read_csv(path + '/mav0/cam0/data.csv')
    rot_interp = get_interpolation(path)
    mcap_start_time = pd.read_csv(path + '/mav0/mocap0/data.csv')['#timestamp [ns]'][0]
    frame_start_time = pd.read_csv(path+'/mav0/cam0/data.csv')['#timestamp [ns]'][0]
    imu_start_time = pd.read_csv(path + '/mav0/imu0/data.csv')['#timestamp [ns]'][0]

    start_time = max(mcap_start_time, frame_start_time, imu_start_time)

    start_index = 0
    while frame_times['#timestamp [ns]'][start_index] < start_time:
        start_index += 1
        images.remove(images[0])

    # plt.ion()
    xyz= np.asarray([[[[column], [row], [1]] for column in range(512)] for row in range(512)])

    i = 0

    offset_mat = np.array(((1.0, 0.0, 0.0),
                        (0.0, 1.0, 0.0),
                        (0.0, 0.0, 1.0)))

    Rcr = np.array(((-1, 0, 0),
                        (0, 0, -1),
                        (0, -1, 0)))

    counter = 0                  ###
    det = [0,0,0,0,0,0]
    center_ref = np.array((0,0)) ###
    confidance = 0               ###
    reset_ref = False
    l2_ref = float('inf')
    center_prev = np.array((0,0))
    center = np.array((0,0))
    save_img = True
    view_img = True
    colors = [random.randint(0, 255) for _ in range(3)]
    label = 'clock'
    counter_black_screen = 0
    center_curr = np.array((0,0))
    a = 0
    # image = image[200:]
   
    for image in images:
        # if i < 126:
        #     a += 1
        #     i = i+1
        #     continue
        
        def derotate_image(frame, K, rot_mat):

            R_fc_to_c = rot_mat.T

            mask = ((rot_mat @ np.linalg.inv(K) @ xyz)[:,:,2,0] > 0).astype(np.uint8)
            
            frame = (mask * frame)

            top_two_rows = (K @ R_fc_to_c @ np.linalg.inv(K))[0:2, :]
            bottom_row = (R_fc_to_c @ np.linalg.inv(K))[2, :]

            map_pixel_c_to_fc = np.vstack((top_two_rows, bottom_row))
            map_pixel_c_to_fc_opencv = np.float32(map_pixel_c_to_fc.flatten().reshape(3,3))

            frame_derotated = cv.warpPerspective(frame, map_pixel_c_to_fc_opencv, (frame.shape[1], frame.shape[0]), flags=cv.WARP_INVERSE_MAP+cv.INTER_LINEAR)
            return mask*255,frame_derotated

        image = cv.imread(image, 0)
        height,width = image.shape
        camera_matrix = np.array([[width*camera_params[0] , 0 , width*camera_params[2]-0.5] , [0 ,height*camera_params[1], height*camera_params[3]-0.5] , [0,0,1] ])
        size_factor = 1
        K_new = np.copy(camera_matrix)
        K_new[0, 2] *= size_factor
        K_new[1, 2] *= size_factor
        undistorted = cv.fisheye.undistortImage(image, camera_matrix, distCoeff,Knew=K_new,new_size=(size_factor*width, size_factor*height))
        # cv.imshow('distorted',undistorted)
        # # cv.imshow('undistorted', undistorted)
        # cv.waitKey(0)
        # start_rot_mat = rot_interp(frame_times['#timestamp [ns]'][start_index]).as_matrix()
        # start_rot_mat = np.array(start_rot_mat)

        # end_rot_mat = rot_interp(frame_times['#timestamp [ns]'][start_index + i]).as_matrix()
        # end_rot_mat = np.array(end_rot_mat)
        # # print(end_rot_mat)

        # trans_rot_mat =  np.linalg.inv(start_rot_mat) @ end_rot_mat
        # trans_rot_mat = offset_mat @ trans_rot_mat
        
        # trans_rot_mat = Rcr @ trans_rot_mat @ Rcr.T
       
        # mask, derotated_image = derotate_image(undistorted, K_new, trans_rot_mat)

        # cv.imshow('derotated', derotated_image)
        # cv.waitKey(0)

        cv.imwrite(path + '/processed_images/undistorted_images/' + ('0' * (5-len(str(i)))) + str(i) + '.png', undistorted)
        # cv.imshow('derotated_images',derotated_image)
        # cv.waitKey(10)
        # print('-------------')
        # print(i)
        # image = cv.imread(image, )

        if i == 0:

            os.system('python3 yolov7/detect2.py \
                        --weights object_detector_weights/yolov7.pt \
                        --source ' + path + '/processed_images/undistorted_images/'  + ('0' * (5-len(str(i)))) + str(i) + '.png')

        # print('0' * (5-len(str(i))) + str(i))
        # processed_image = cv.imread(path + '/processed_images/undistorted_images/'  + ('0' * (5-len(str(i)))) + str(i) + '.png',0)
        # yolo = cv.imread('/home/tau/Documents/GitHub/PRG-Deep-Tau-Constraint/runs/detect/exp/' + ('0' * (5-len(str(i)))) + str(i) + '.png',0)
        # cv.imshow('image_passed_to_yolo', processed_image)
        det = torch.load('/home/tau/Desktop/det_tensor/det_tensor.pt')
        # cv.waitKey(0)
        # print('-----------')
        # print(det)
        # im0_copy = copy.deepcopy(derotated_image) 

        # res = np.zeros((im0_copy.shape),dtype = np.uint8)
        # im_draw = copy.deepcopy(derotated_image) 
        for *xyxy, conf, cls in reversed(det):

        #     im0_copy = copy.deepcopy(derotated_image) 
        #     # im0_copy = copy.deepcopy(derotated_image) 
        #     # res = np.zeros((im0_copy.shape),dtype = np.uint8)
        #     im_draw = copy.deepcopy(derotated_image) 
        #     # cv.imshow('derotated_image', im_draw)

            undistorted_image1 = cv.imread(path + '/processed_images/undistorted_images/'  + ('0' * (5-len(str(0)))) + str(0) + '.png',0)
            
            c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
            
            if i > 0 and i < 10:

                crop_images1 = undistorted_image1[c1[1]:c2[1],c1[0]:c2[0]]
                crop_images2 = undistorted[c1[1]:c2[1],c1[0]:c2[0]]
                crop_images = cv.vconcat([crop_images1,crop_images2])
                cv.imwrite(path + '/processed_images/cropped_images/' + ('0' * (5-len(str(i)))) + str(i) + '.png', crop_images)



                # cv.imshow('cropped_image', crop_images)
            # cv.waitKey(0)

            # print('-------------------------')
            # print(c1,c2)
            # cv.rectangle(undistorted,c1,c2,(0,0,255),1)
                # cv.imshow('yolo', undistorted)
                # cv.waitKey(0)
        # print('====================')
        #     if counter == 0:                         ###

        #         confidance = max(conf,confidance)    ###

        #         if confidance == conf:             

        #             center_ref[0] = c1[1] + (c2[1]-c1[1])/2
        #             center_ref[1] = c1[0] + (c2[0]-c1[0])/2

        #             res = copy.deepcopy(im0_copy)
        #             # final_image = cv.hconcat([processed_image,yolo,im0_copy])
        #             # cv.imshow('final_image', final_image)
        #             # cv.waitKey(0)
        #             # cv.destroyAllWindows()
                    
        #             res[:c1[1],:] = 0 ###
                    
        #             res[:,:c1[0]] = 0 ###
        #             res[c2[1]:,:] = 0 ###
        #             res[:,c2[0]:] = 0 ###
                    
        #             center_prev = center_ref
        #             center_curr = center_prev

        #             # final_image = cv.hconcat([processed_image,yolo,res])
        #             # cv.imshow('final_image', final_image)
        #             # cv.waitKey(0)
        #             # cv.destroyAllWindows()

        # #                 # res = im0_copy

        #     else:

        #         center = np.array((0,0))
        #         center[0] = c1[1] + (c2[1]-c1[1])/2
        #         center[1] = c1[0] + (c2[0]-c1[0])/2

        #         l2 = np.linalg.norm(center - center_ref)
        #         l2_prev_center = np.linalg.norm(center - center_prev)

        #         if l2 < l2_ref and l2_prev_center < 10: 

        #             l2_ref = l2
        #             center_curr = center
        #             res = copy.deepcopy(im0_copy)
        #             res[:c1[1],:] = 0 ###
        #             res[:,:c1[0]] = 0 ###
        #             res[c2[1]:,:] = 0 ###
        #             res[:,c2[0]:] = 0 ###
       
        # # #     # res += im0_copy ###
        # # #         # cv2.circle(img=im0,center=c2,radius=2,color=(255,255,255),thickness= 0) ###
        # # #         # cv2.circle(img=im0,center=c1,radius=2,color=(255,255,255),thickness= 0) ###
        # # print(processed_image.shape)
        # # print(yolo.shape)
        # # print(res.shape)

        # center_prev = copy.deepcopy(center_curr)
        # final_image = cv.hconcat([derotated_image,res])
        # # cv.imshow('final_image', final_image)
        # cv.imwrite(path + '/final_image/' + ('0' * (5-len(str(i)))) + str(i) + '.png', final_image)
        # cv.imwrite(path + '/res/' + ('0' * (5-len(str(i)))) + str(i) + '.png', res)

        # # cv.waitKey(1)

        # l2_ref = float('inf')
        # confidance = 0

        # if len(np.nonzero(res)[1]):
        #     counter += 1

        # if not len(np.nonzero(res)[1]):
        #     counter_black_screen += 1
        
        
        # # print('i'+'-----------' + str(i))
        # # print(len(np.nonzero(res)[1]))
        # # print('Counter'+ '-----------------' + str(counter))
        # # print('black_screen_counter' + '--------------------'+ str(counter_black_screen))
        # cv.imshow('final_image'+ '  ' + str(i),final_image)
        # cv.waitKey(2000)
        # cv.destroyAllWindows()
        # if counter_black_screen == 10 or counter == 0:

        #     get_info_from_frame (i, path, path)
        #     offset_mat = (np.linalg.inv(start_rot_mat) @ end_rot_mat).T
        #     counter_black_screen = 0
        #     counter = 0
        
        # # print(counter)
        # # black_pixels = (512*512-np.count_nonzero(derotated_image))/512**2
        
        # # if black_pixels > 0.75:

        # # # # if counter == 0:
        # #     print(str(i))
        # #     get_info_from_frame (i, path, path)
        # #     offset_mat = (np.linalg.inv(start_rot_mat) @ end_rot_mat).T
        # #     # cv.imwrite(path + '/processed_images/' + ('0' * (5-len(str(i)))) + str(i) + '.png', undistorted)
        # #     # cv.imwrite(path + '/processed_images/' + ('0' * (5-len(str(i)))) + str(i) + '.png', im0)
        # #     cv.imshow('derotated',undistorted)
        # #     cv.waitKey(1)
        # # else:
        # #     cv.imshow('derotated',derotated_image)
        # #     cv.waitKey(1)
        # #     # cv.imwrite(path + '/processed_images/' + ('0' * (5-len(str(i)))) + str(i) + '.png', derotated_image)
        # # #     # cv.imwrite(path + '/processed_images/' + ('0' * (5-len(str(i)))) + str(i) + '.png', im0)

        # # cv.imshow(str(a-1),res)
        # # cv.imshow('im_draw', im_draw)
        # # cv.waitKey(0)
        # # cv.destroyAllWindows()
        
        i += 1
        a += 1

        # if i==10:
        #     break

if __name__ == '__main__':
    undistort('/home/tau/Desktop/monocular_data/dataset-corridor1_512_16')
    
