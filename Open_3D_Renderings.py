import open3d as o3d
import numpy as np
from time import time
from scipy.spatial.transform import Rotation as Rot
import cv2
import math
import matplotlib.pyplot as plt
import os

# function outputs images, acceleration, time, and delta accel at each frame as a 2D list. The function is
# called in the Dataset.py file, namely the custom dataset class uses it when it's outputing data during training
def get_sim_data(camera_pos_func, camera_vel_func, camera_accel_func, time_step = 1/10, max_time = 5):

    # helper function for finding orientation
    def get_rot(pos1,pos2):
        R_ovi = Rot.from_quat(pos1[3:]).as_matrix()
        R_ovi2 = Rot.from_quat(pos2[3:]).as_matrix()
        R_voi2 = np.linalg.inv(R_ovi2)
        R_ii2 = R_ovi@R_voi2
        return R_ii2

       # remove previously saved images
    # for filename in os.listdir("C:/Users/ezhu2/Documents/GitHub/Perception-and-Robotics-Group/Saved_Images"):
        # print(filename)
        # os.remove("C:/Users/ezhu2/Documents/GitHub/Perception-and-Robotics-Group/Saved_Images/" + filename)


    # o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    source = o3d.io.read_triangle_mesh("C:/Users/ezhu2/Documents/GitHub/Perception-and-Robotics-Group/PLY_Files/Cushion.ply") 

    # create array of camera positions, times, accelerations, and delta of acceleration
    camera_pos = [] # z direction
    accel_arr = [] # z direction
    time_arr = []
    delta_accel_arr = [] # z direction

    for i in range(int(max_time / time_step)):
        time = i * time_step
        camera_pos.append(camera_pos_func(time))
        accel_arr.append(camera_accel_func(time)[2])
        time_arr.append(time)
        delta_accel_arr.append(camera_pos[i][2] - camera_pos[0][2] - camera_vel_func(0)[2] * time)
        # print(camera_pos[i])

    camera_pos = np.array(camera_pos)
    

    vis = o3d.visualization.Visualizer() # initialize window
    vis.create_window() # initialize window
    vis.add_geometry(source) # include the model in the visualization

    ctr = vis.get_view_control() 
    ctr.set_constant_z_far(10000) # all models over 10000 units away or more are not rendered
    ctr.set_constant_z_near(1) # all models closer to 1 unit away should not be rendered

    inital_pos = camera_pos[0,:3]
    R_rig = np.eye(3)
    # find the 4 by 4 camera transition matrix
    trans_rigv = np.row_stack((np.column_stack((R_rig, np.array(inital_pos))), np.array([0,0,0,1])))
    source.transform(trans_rigv) # transform the model but not the camera
    vis.update_geometry(source) # update
    vis.poll_events() # update
    vis.update_renderer() # update

    img_arr = []

    for i in range(camera_pos.shape[0]-1):
        
        # change in camera pos
        delta_cam = camera_pos[i+1,:3] - camera_pos[i,:3]
            
        # get the rotation matrices
        R_rig = get_rot(camera_pos[i,:], camera_pos[i+1,:])
        
        # find the 4 by 4 camera transition matrix
        trans_rigv = np.row_stack((np.column_stack((R_rig, np.array(delta_cam))), np.array([0,0,0,1])))
        trans_v_rig = trans_rigv

        trans_o_rig = trans_v_rig # combine the car and camera transformations into one matrix transformation
        source.transform(trans_o_rig) # transform the model but not the camera
        vis.update_geometry(source) # update
        vis.poll_events() # update
        vis.update_renderer() # update

        # vis.capture_screen_image("C:/Users/ezhu2/Documents/GitHub/Perception-and-Robotics-Group/Saved_Images/frame" + str(i) + ".png")

        image = np.asarray(vis.capture_screen_float_buffer())
        img_arr.append(image)
    vis.destroy_window()
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Info)
    
    
    img_arr = np.array(img_arr, dtype=float)
    img_arr = np.uint8(img_arr * 255)
    accel_arr = np.array(accel_arr, dtype=float)
    time_arr = np.array(time_arr, dtype=float)
    delta_accel_arr = np.array(delta_accel_arr)

    return [img_arr, accel_arr, time_arr, delta_accel_arr]


if __name__ == '__main__':
    def pos(time):
        return [50 * math.sin(time), 50 * math.cos(time),  -12 * time**2 - 100, 0, 0, 0, 1]
    
    def vel(time):
        return [50 * math.cos(time), -50 * math.sin(time),  -24 * time, 0, 0, 0, 1]
    
    def accel(time):
        return [-50 * math.sin(time), -50 * math.cos(time),  -24, 0, 0, 0, 1]

    data = get_sim_data(pos, vel, accel)
    
    frames = data[0]
    print("acceleration:")
    print(data[1])
    print("\ntime:")
    print(data[2])
    print("\ndelta:")
    print(data[3])

    print()
    # for frame in frames:
        # print(type(frame))
        # plt.imshow(frame)
        # plt.show()
    


