from math import cos, sin
import cv2 as cv
import numpy as np
import glob
import csv
from scipy.spatial.transform import Rotation as R
from interpolate import get_interpolation
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def derotate_image(frame, K, rot_mat):
    R_fc_to_c = rot_mat.T
    #R_c_to_fc = DCM(q=q_c_to_fc)
    #R_fc_to_c = R_c_to_fc.A.transpose()

    # Derive this by considering p1 = (K R K_inv) (Z(X)/Z(RX)) p0
    top_two_rows = (K @ R_fc_to_c @ np.linalg.inv(K))[0:2, :]
    bottom_row = (R_fc_to_c @ np.linalg.inv(K))[2, :]

    map_pixel_c_to_fc = np.vstack((top_two_rows, bottom_row))
    map_pixel_c_to_fc_opencv = np.float32(map_pixel_c_to_fc.flatten().reshape(3,3))

    frame_derotated = cv.warpPerspective(frame, map_pixel_c_to_fc_opencv, (frame.shape[1], frame.shape[0]), flags=cv.WARP_INVERSE_MAP+cv.INTER_LINEAR)
    return frame_derotated

#equidistant_params = [fx fy cx cy k1 k2 k3 k4] #from camera.txt file in dataset
p = [0.373004838186, 0.372994740336, 0.498890050897, 0.502729380663, 0.00348238940225, 0.000715034845216, -0.00205323614187, 0.000202936735918]
distCoeff = np.array([[0.00348238940225, 0.000715034845216, -0.00205323614187, 0.000202936735918]])
directory = '/home/tau/Desktop/monocular_data/dataset-corridor1_512_16/dso/cam0/derotated_images/'

images = []
for image in glob.glob("/home/tau/Desktop/monocular_data/dataset-corridor1_512_16/dso/cam0/images/*.png"):
    images.append(image)
images.sort()
frame_times = pd.read_csv('/home/tau/Desktop/monocular_data/dataset-corridor1_512_16/mav0/cam0/data.csv')
(x_interp, y_interp, z_interp, rot_interp) = get_interpolation()
i = 0
global_trans_rot_mat = None


plt.ion()
for image in images:
    image = cv.imread(image, 0)
    height,width = image.shape
    camera_matrix = np.array([[width*p[0] , 0 , width*p[2]-0.5] , [0 ,height*p[1], height*p[3]-0.5] , [0,0,1] ])
    size_factor = 1
    K_new = np.copy(camera_matrix)
    K_new[0, 2] *= size_factor
    K_new[1, 2] *= size_factor
    undistorted = cv.fisheye.undistortImage(image, camera_matrix, distCoeff,Knew=K_new,new_size=(size_factor*width, size_factor*height))
    
    start_rot_mat = rot_interp(frame_times['#timestamp [ns]'][0]).as_matrix()
    start_rot_mat = np.array(start_rot_mat)

    end_rot_mat = rot_interp(frame_times['#timestamp [ns]'][i]).as_matrix()
    end_rot_mat = np.array(end_rot_mat)

    trans_rot_mat =  np.linalg.inv(start_rot_mat) @ end_rot_mat
    trans_rot_mat =  end_rot_mat @ np.linalg.inv(start_rot_mat)

    Rcr = np.array(((-1, 0, 0),
                    (0, 0, -1),
                    (0, -1, 0)))
    # Rcr = np.eye(3)

    trans_rot_mat = Rcr @ trans_rot_mat @ Rcr.T

    # plt.clf()
    # plt.xlim([-1, 1])
    # plt.ylim([-1, 1])
    # # plt.gca().invert_yaxis()
    # plt.plot([0, trans_rot_mat[0][0]/ (trans_rot_mat[2][0]+1)], [0, trans_rot_mat[1][0]/ (trans_rot_mat[2][0]+1)], 'r-')
    # plt.plot([0, trans_rot_mat[0][1]/ (trans_rot_mat[2][1]+1)], [0, trans_rot_mat[1][1]/ (trans_rot_mat[2][1]+1)], 'g-')
    # plt.plot([0, trans_rot_mat[0][2]/ (trans_rot_mat[2][2]+1)], [0, trans_rot_mat[1][2]/ (trans_rot_mat[2][2]+1)], 'b-')
    
    # print((trans_rot_mat[2][2]+1))
    # print(np.linalg.det(trans_rot_mat))
    plt.pause(0.01)         

    
    # trans_rot_mat = np.linalg.inv(trans_rot_mat)
    

    # cv.imshow('original', image)
    
    # undistorted = undistorted[:, ::-1]

    cv.imshow('undistorted',undistorted)
    
    
    cv.imshow('derotated',derotate_image(undistorted, K_new, trans_rot_mat))
    # cv.imwrite(directory + str(i) +"derotated.png" , output)
    cv.waitKey(50)
    i+=1
