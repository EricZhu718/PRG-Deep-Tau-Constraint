import numpy as np
import cv2 as cv
from scipy.spatial.transform import Rotation as R

def derotate_image(frame, K, q_c_to_fc):
    R_c_to_fc = R.from_quat((q_c_to_fc[1], q_c_to_fc[2], q_c_to_fc[3], q_c_to_fc[0])).as_matrix()
    R_fc_to_c = R_c_to_fc.T
    #R_c_to_fc = DCM(q=q_c_to_fc)
    #R_fc_to_c = R_c_to_fc.A.transpose()

    # Derive this by considering p1 = (K R K_inv) (Z(X)/Z(RX)) p0
    top_two_rows = (K @ R_fc_to_c @ np.linalg.inv(K))[0:2, :]
    bottom_row = (R_fc_to_c @ np.linalg.inv(K))[2, :]

    map_pixel_c_to_fc = np.vstack((top_two_rows, bottom_row))
    map_pixel_c_to_fc_opencv = np.float32(map_pixel_c_to_fc.flatten().reshape(3,3))

    frame_derotated = cv.warpPerspective(frame, map_pixel_c_to_fc_opencv, (frame.shape[1], frame.shape[0]), flags=cv.WARP_INVERSE_MAP+cv.INTER_LINEAR)
    return frame_derotated

# Hamilton product
# https://en.wikipedia.org/wiki/Quaternion#Hamilton_product
# Scalar first
def quat_mult(q, p):
    r = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    r[0] = q[0]*p[0] - q[1]*p[1] - q[2]*p[2] - q[3]*p[3]
    r[1] = q[0]*p[1] + q[1]*p[0] + q[2]*p[3] - q[3]*p[2]
    r[2] = q[0]*p[2] - q[1]*p[3] + q[2]*p[0] + q[3]*p[1]
    r[3] = q[0]*p[3] + q[1]*p[2] - q[2]*p[1] + q[3]*p[0]
    return r

# Scalar first
def quat_inv_no_norm(q):
    q_inv = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    print(q_inv)
    print(q)
    q_inv[0] = q[0]
    q_inv[1:4] = -q[1:4]
    return q_inv

# Scalar first
# Forward eular, could do trapezoidal
def integrate_quaternion(q, gyr, dt):
    p = np.array([0.0, gyr[0], gyr[1], gyr[2]], dtype=np.float32)
    dot_q = 0.5 * quat_mult(q, p)
    q_unpacked = q + dt * dot_q
    return q_unpacked