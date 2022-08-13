import cv2 as cv
import numpy as np
import glob
import csv
from scipy import interpolate
from scipy.spatial.transform import Slerp, Rotation as R
import pandas as pd
import matplotlib.pyplot as plt



def get_interpolation():
    position_file = pd.read_csv('/home/tau/Desktop/monocular_data/dataset-corridor1_512_16/mav0/mocap0/data.csv')
    x_interp, y_interp, z_interp = \
        interpolate.interp1d(position_file['#timestamp [ns]'], position_file[' p_RS_R_x [m]']), \
        interpolate.interp1d(position_file['#timestamp [ns]'], position_file[' p_RS_R_y [m]']), \
        interpolate.interp1d(position_file['#timestamp [ns]'], position_file[' p_RS_R_z [m]'])


    rotations = R.from_quat(position_file.loc[:, [' q_RS_x []', ' q_RS_y []', ' q_RS_z []', ' q_RS_w []']])
    quaternion_interp = Slerp(position_file['#timestamp [ns]'], rotations)
    return (x_interp, y_interp, z_interp, quaternion_interp)

if __name__ == '__main__':
    position_file = pd.read_csv('/home/tau/Desktop/monocular_data/dataset-corridor1_512_16/mav0/mocap0/data.csv')
    x_interp, y_interp, z_interp = \
        interpolate.interp1d(position_file['#timestamp [ns]'], position_file[' p_RS_R_x [m]']), \
        interpolate.interp1d(position_file['#timestamp [ns]'], position_file[' p_RS_R_y [m]']), \
        interpolate.interp1d(position_file['#timestamp [ns]'], position_file[' p_RS_R_z [m]'])


    rotations = R.from_quat(position_file.loc[:, [' q_RS_x []', ' q_RS_y []', ' q_RS_z []', ' q_RS_w []']])
    quaternion_interp = Slerp(position_file['#timestamp [ns]'], rotations)

    rand_times = np.sort(np.random.rand(1000)) * (position_file['#timestamp [ns]'][len(position_file['#timestamp [ns]'])-1] - position_file['#timestamp [ns]'][0]) \
        + position_file['#timestamp [ns]'][0]
    print(quaternion_interp(rand_times).as_quat())
    # plt.plot(rand_times, np.array(quaternion_interp(rand_times).as_quat())[:,0])
    # plt.plot(position_file['#timestamp [ns]'], position_file[' q_RS_w []'])
    # plt.plot(position_file['#timestamp [ns]'], position_file[' q_RS_x []'])
    # plt.plot(position_file['#timestamp [ns]'], position_file[' q_RS_y []'])
    # plt.plot(position_file['#timestamp [ns]'], position_file[' q_RS_z []'])
    plt.plot(position_file['#timestamp [ns]'])



    plt.show()
    
    
    

    




