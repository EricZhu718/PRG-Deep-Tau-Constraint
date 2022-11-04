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

from reference_frame import get_info_from_frame

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
    images.sort()
    images = images
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

    plt.ion()
    xyz= np.asarray([[[[column], [row], [1]] for column in range(512)] for row in range(512)])

    i = 0

    offset_mat = np.array(((1.0, 0.0, 0.0),
                        (0.0, 1.0, 0.0),
                        (0.0, 0.0, 1.0)))

    Rcr = np.array(((-1, 0, 0),
                        (0, 0, -1),
                        (0, -1, 0)))

    for image in images:
        
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
        
        start_rot_mat = rot_interp(frame_times['#timestamp [ns]'][start_index]).as_matrix()
        start_rot_mat = np.array(start_rot_mat)

        end_rot_mat = rot_interp(frame_times['#timestamp [ns]'][start_index + i]).as_matrix()
        end_rot_mat = np.array(end_rot_mat)

        trans_rot_mat =  np.linalg.inv(start_rot_mat) @ end_rot_mat
        trans_rot_mat = offset_mat @ trans_rot_mat
        

        trans_rot_mat = Rcr @ trans_rot_mat @ Rcr.T

        cv.imshow('undistorted',undistorted)

        mask, derotated_image = derotate_image(undistorted, K_new, trans_rot_mat)

        black_pixels = (512*512-np.count_nonzero(derotated_image))/512**2
        
        if black_pixels > 0.75:
            print(str(i))
            get_info_from_frame (i, path, path)
            offset_mat = (np.linalg.inv(start_rot_mat) @ end_rot_mat).T
            cv.imwrite(path + '/processed_images/' + '0' * (5-len(str(i))) + str(i) + '.png', undistorted)
        else:
            cv.imwrite(path + '/processed_images/' + '0' * (5-len(str(i))) + str(i) + '.png', derotated_image)

        # cv.imshow('mask',mask)
        cv.imshow('derotated',derotated_image)
        cv.waitKey(1)
       
        i+=1

if __name__ == '__main__':
    undistort('/home/tau/Desktop/monocular_data/dataset-corridor1_512_16')