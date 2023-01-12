
# from glob import glob
from matplotlib import pyplot as plt
# import cv2 
# import random
import numpy as np
# from numpy.linalg import inv
import torch
# from torch import Tensor
# import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# import math
# from time import time
# from scipy.spatial.transform import Rotation as Rot
# import math
# from torch import scalar_tensor
# import time
# import copy
# import gc

from draw_bounding_box import calculate_phi,get_bounding_box

# from google.colab import drive, files
import os
import sys
import torch
# import pytorch3d
import warnings

from data import SplineDataset, load_data_set
from model import Model
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt




if __name__ == '__main__':
    model = torch.load('object_detector_weights/yolov7.pt')
    print(model)
    


