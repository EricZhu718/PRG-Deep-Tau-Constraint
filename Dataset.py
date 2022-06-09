from glob import glob
from matplotlib import pyplot as plt
import cv2
import random
import numpy as np
from numpy.linalg import inv
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

from Open_3D_Renderings import get_sim_data

class VisualDataset(Dataset):
    data = []
    size = 0

    def __init__(self, data_size):
        global data, size
        size = data_size

        random.seed = 100
        

    def len(self):
        return size
    
    def __getitem__(self, index):
        random_val = 2 * random.random() - 1
        random_val2 = 2 * random.random() - 1
        random_val3 = 2 * random.random() - 1

        def pos(time):
            return [50 * random_val * math.sin(time) + 20 * random_val2 * math.cos(time), 
                        50 * random_val * math.cos(time) + 20 * random_val2 * math.sin(time),  -2.5 * time**2 * random_val3 + 10, 0, 0, 0, 1]
        def vel(time):
            return [50 * random_val * math.cos(time) - 20 * random_val2 * math.sin(time), 
                        -50 * random_val * math.sin(time) + 20 * random_val2 * math.cos(time),  -5 * time * random_val3, 0, 0, 0, 1]
        def accel(time):
            return [-50 * random_val * math.sin(time) - 20 * random_val2 * math.cos(time), 
                        -50 * random_val * math.cos(time) - 20 * random_val2 * math.sin(time),  -5 * random_val3, 0, 0, 0, 1]
        return get_sim_data(pos, vel, accel)