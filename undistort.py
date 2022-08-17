from copy import deepcopy
import string
import cv2 as cv
import numpy as np
import glob
from interpolate import get_interpolation
import pandas as pd
import matplotlib.pyplot as plt
import copy

def undistort(path:string):
    
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
    i = 0

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
    for image in images:
        
        def derotate_image(frame, K, rot_mat):

            # R = copy.deepcopy(rot_mat)

            R_fc_to_c = rot_mat.T

            #R_c_to_fc = DCM(q=q_c_to_fc)
            #R_fc_to_c = R_c_to_fc.A.transpose()

            # Derive this by considering p1 = (K R K_inv) (Z(X)/Z(RX)) p0
            # x, y = np.meshgrid(np.linspace(0,511,512), np.linspace(0,511,512))

            # x = np.fromfunction(lambda i, j: j % 512, (1, 512**2))[0]
            # # print(x)
            # y = np.fromfunction(lambda i, j: j // 512, (1, 512**2))[0]
            # # print(y)
            # z = np.ones((1, 512**2))[0]
            # # print(z)
            # xyz = np.vstack((x,y,z))

            

            # print(xyz[10][0])

            mask = ((rot_mat @ np.linalg.inv(K) @ xyz)[:,:,2,0] > 0).astype(np.uint8)
            # print(mask)
            
            # print(R_fc_to_c @ np.linalg.inv(K) @ xyz[10,0])
            # print(__)
            # print(y)

            # derotated_pixels = R_fc_to_c @ np.linalg.inv(K) @ xyz
            # # print(derotated_pixels)
            # # print(np.where(derotated_pixels[:2]<0))
            # # print(derotated_pixels[2] < 0)
            # # print((derotated_pixels[2] > 0).reshape(512, 512))

            # derotated_pixels = np.where(derotated_pixels[:2]<0)

            # bool_mat = derotated_pixels[:512]

            # for i in range(1,512):
            #     temp = derotated_pixels[i*512:((i+1)*512)]
            #     bool_mat = np.vstack((bool_mat,temp))

            # # print(bool_mat)

            # frame = frame * (derotated_pixels[2] > 0).reshape(512, 512)
            # print(frame)
            
            frame = (mask * frame)
            # print(frame)
            

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
        # trans_rot_mat =  end_rot_mat @ np.linalg.inv(start_rot_mat)

        Rcr = np.array(((-1, 0, 0),
                        (0, 0, -1),
                        (0, -1, 0)))

        # Rcr = np.eye(3)
        # Rcr = Rcr.T

        trans_rot_mat = Rcr @ trans_rot_mat @ Rcr.T

        # plt.clf()
        # plt.xlim([-1, 1])
        # plt.ylim([-1, 1])
        # # # plt.gca().invert_yaxis()
        # plt.plot([0, trans_rot_mat[0][0]/ (trans_rot_mat[2][0]+1)], [0, -trans_rot_mat[1][0]/ (trans_rot_mat[2][0]+1)], 'r-')
        # plt.plot([0, trans_rot_mat[0][1]/ (trans_rot_mat[2][1]+1)], [0, -trans_rot_mat[1][1]/ (trans_rot_mat[2][1]+1)], 'g-')
        # plt.plot([0, trans_rot_mat[0][2]/ (trans_rot_mat[2][2]+1)], [0, -trans_rot_mat[1][2]/ (trans_rot_mat[2][2]+1)], 'b-')
        
        # # print((trans_rot_mat[2][2]+1))
        # # print(np.linalg.det(trans_rot_mat))
        # plt.pause(0.01)         
        # plt.show()
        
        # trans_rot_mat = np.linalg.inv(trans_rot_mat)
        

        # cv.imshow('original', image)
        
        # undistorted = undistorted[:, ::-1]

        cv.imshow('undistorted',undistorted)

        mask, derotated_image = derotate_image(undistorted, K_new, trans_rot_mat)
        cv.imshow('mask',mask)
        cv.imshow('derotated',derotated_image)
        # cv.imwrite(directory + str(i) +"derotated.png" , output)
        cv.waitKey(0)
        # break
        i+=1

if __name__ == '__main__':
    
    # path = '/home/tau/Desktop/monocular_data/dataset-corridor3_512_16'

    # imu_gt = np.array(pd.read_csv(path + '/dso/gt_imu.csv')['# timestamp[ns]'])
    # imu_gt = (imu_gt - imu_gt[0]) / 10**9
    # imu_gt_diff = imu_gt[1:] - imu_gt[0:-1]

    # frames_times = np.array(pd.read_csv(path + '/mav0/cam0/data.csv')['#timestamp [ns]'])
    # frames_times = (frames_times - frames_times[0]) / 10**9
    # frames_diff = frames_times[1:] - frames_times[0:-1]

    # imu = np.array(pd.read_csv(path + '/mav0/imu0/data.csv')['#timestamp [ns]'])
    # imu = (imu - imu[0]) / 10**9
    # imu_diff = imu[1:] - imu[0:-1]

    # plt.plot(imu_gt_diff)
    # plt.plot(frames_diff)
    # plt.plot(imu_diff)
    # plt.show()
    undistort('/home/tau/Desktop/monocular_data/dataset-corridor1_512_16')
