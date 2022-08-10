import cv2 as cv
import numpy as np
import glob
from derotate import derotate_image,quat_inv_no_norm,quat_mult
import csv
from scipy.spatial.transform import Rotation as R

def rotationMatrixToQuaternion(m):

    t = np.matrix.trace(m)
    q = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    if(t > 0):
        t = np.sqrt(t + 1)
        q[3] = 0.5 * t
        t = 0.5/t
        q[0] = (m[2,1] - m[1,2]) * t
        q[1] = (m[0,2] - m[2,0]) * t
        q[2] = (m[1,0] - m[0,1]) * t

    else:
        i = 0
        if (m[1,1] > m[0,0]):
            i = 1
        if (m[2,2] > m[i,i]):
            i = 2
        j = (i+1)%3
        k = (j+1)%3

        t = np.sqrt(m[i,i] - m[j,j] - m[k,k] + 1)
        q[i] = 0.5 * t
        t = 0.5 / t
        q[3] = (m[k,j] - m[j,k]) * t
        q[j] = (m[j,i] + m[i,j]) * t
        q[k] = (m[k,i] + m[i,k]) * t

    return q


quaternions = []
with open('/home/tau/Desktop/monocular_data/dataset-corridor1_512_16/mav0/mocap0/data.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    # line_count = 0
    i = 0
    quaternions = []

    for row in csv_reader:
        # if line_count == 0:
        #     print(f'Column names are {", ".join(row)}')
        #     line_count += 1
        # else:
        #     print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
        #     line_count += 1
        if i == 0:
            i += 1
            continue

        quaternions.append([row[4], row[5], row[6], row[7]])

        i += 1
        if i==3:
            break
    # print(f'Processed {line_count} lines.')
    # print(f'quaternion{quaternions}')

#equidistant_params = [fx fy cx cy k1 k2 k3 k4] #from camera.txt file in dataset
p = [0.373004838186, 0.372994740336, 0.498890050897, 0.502729380663, 0.00348238940225, 0.000715034845216, -0.00205323614187, 0.000202936735918]

distCoeff = np.array([[0.00348238940225, 0.000715034845216, -0.00205323614187, 0.000202936735918]])

directory = '/home/tau/Desktop/monocular_data/dataset-corridor1_512_16/dso/cam0/derotated_images/'

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

    if i==1:
        q1 = quaternions[0]
        # print(q1)
        q1 = R.from_quat((q1[1], q1[2], q1[3], q1[0])).as_matrix()
        # print(np.linalg.inv(q1))
        q2 = quaternions[1]
        q2 = R.from_quat((q2[1], q2[2], q2[3], q2[0])).as_matrix()

        q = np.matmul(np.linalg.inv(q1),q2)
        q = rotationMatrixToQuaternion(q)
        # print(q)
        output = derotate_image(output,camera_matrix, q)

    cv.imshow('original', image_1)
    cv.imshow('undistorted',output)
    cv.imwrite(directory + str(i) +"derotated.png" , output)
    cv.waitKey(10)
    i+=1

    if i== 2:
        break

