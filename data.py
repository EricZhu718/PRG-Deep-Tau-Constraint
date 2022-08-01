from glob import glob
from matplotlib import pyplot as plt
import cv2
import random
import numpy as np
from numpy.linalg import inv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from time import time
from scipy.spatial.transform import Rotation as Rot
import math
from torch import scalar_tensor
import time
import copy

# from google.colab import drive, files
import os
import sys
import torch
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt


import sys
import os
import math
import numpy as np
import random


class DirectoryLoadedDataset(Dataset):
  
  def __init__(self, dir_path):
    num_files = len(os.listdir(dir_path))
    self.data = [None for i in range(num_files)]
    if dir_path[-1] != '/':
      dir_path += '/'
    counter = 0
    for file in os.listdir(dir_path):
      # print(file)
      self.data[counter] = torch.load(dir_path + file)
      counter += 1
    
  def __len__(self):
    return len(self.data)
    
  def __getitem__(self, index):
    return self.data[index]


def save_data_set(dataset:Dataset, directory_name:str) -> None:
  try:
    os.mkdir(directory_name)
  finally:
    for i in range(len(dataset)):
      if directory_name[-1] == '/':
        path = directory_name + str(i)
      else:
        path = directory_name + '/' + str(i)
      torch.save(dataset.__getitem__(i), path)


# directory must only contain tensor files, no directories
def load_data_set(directory_name:str) -> Dataset:
  print('loading data from ' + directory_name)
  return DirectoryLoadedDataset(directory_name)


# if __name__ == '__main__':
  # save_data_set(SplineDataset(1000, frames = 200), "../Video_Datasets/1000Videos200Frames/")
  # save_data_set(SplineDataset(1000, frames = 150), "../Video_Datasets/1000Videos150Frames/")
  # save_data_set(SplineDataset(750, frames = 150), "../Video_Datasets/750Videos150Frames/")
  # save_data_set(SplineDataset(500, frames = 150), "../Video_Datasets/500Videos150Frames/")
  # save_data_set(SplineDataset(250, frames = 150), "../Video_Datasets/250Videos150Frames/")
  # save_data_set(SplineDataset(100, frames = 150), "../Video_Datasets/100Videos150Frames/")
  # save_data_set(SplineDataset(50, frames = 150), "../Video_Datasets/50Videos150Frames/")
  # save_data_set(SplineDataset(20, frames = 150), "../Video_Datasets/20Videos150Frames/")

  # save_data_set(SplineDataset(1000, frames = 100), "../Video_Datasets/1000Videos100Frames/")
  # save_data_set(SplineDataset(750, frames = 100), "../Video_Datasets/750Videos100Frames/")
  # save_data_set(SplineDataset(500, frames = 100), "../Video_Datasets/500Videos100Frames/")
  # save_data_set(SplineDataset(250, frames = 100), "../Video_Datasets/250Videos100Frames/")
  # save_data_set(SplineDataset(100, frames = 100), "../Video_Datasets/100Videos100Frames/")
  # save_data_set(SplineDataset(50, frames = 100), "../Video_Datasets/50Videos100Frames/")
  # save_data_set(SplineDataset(20, frames = 100), "../Video_Datasets/20Videos100Frames/")

  # save_data_set(SplineDataset(1000, frames = 50), "../Video_Datasets/1000Videos50Frames/")
  # save_data_set(SplineDataset(750, frames = 50), "../Video_Datasets/750Videos50Frames/")
  # save_data_set(SplineDataset(500, frames = 50), "../Video_Datasets/500Videos50Frames/")
  # save_data_set(SplineDataset(250, frames = 50), "../Video_Datasets/250Videos50Frames/")
  # save_data_set(SplineDataset(100, frames = 50), "../Video_Datasets/100Videos50Frames/")
  # save_data_set(SplineDataset(50, frames = 50), "../Video_Datasets/50Videos50Frames/")
  # save_data_set(SplineDataset(20, frames = 50), "../Video_Datasets/20Videos50Frames/")

  # save_data_set(SplineDataset(1000, frames = 20), "../Video_Datasets/1000Videos20Frames/")
  # save_data_set(SplineDataset(750, frames = 20), "../Video_Datasets/750Videos20Frames/")
  # save_data_set(SplineDataset(500, frames = 20), "../Video_Datasets/500Videos20Frames/")
  # save_data_set(SplineDataset(250, frames = 20), "../Video_Datasets/250Videos20Frames/")
  # save_data_set(SplineDataset(100, frames = 20), "../Video_Datasets/100Videos20Frames/")
  # save_data_set(SplineDataset(50, frames = 20), "../Video_Datasets/50Videos20Frames/")
  # save_data_set(SplineDataset(20, frames = 20), "../Video_Datasets/20Videos20Frames/")

  # save_data_set(SplineDataset(1000, frames = 10), "../Video_Datasets/1000Videos10Frames/")
  # save_data_set(SplineDataset(750, frames = 10), "../Video_Datasets/750Videos10Frames/")
  # save_data_set(SplineDataset(500, frames = 10), "../Video_Datasets/500Videos10Frames/")
  # save_data_set(SplineDataset(250, frames = 10), "../Video_Datasets/250Videos10Frames/")
  # save_data_set(SplineDataset(100, frames = 10), "../Video_Datasets/100Videos10Frames/")
  # save_data_set(SplineDataset(50, frames = 10), "../Video_Datasets/50Videos10Frames/")
  # save_data_set(SplineDataset(20, frames = 10), "../Video_Datasets/20Videos10Frames/")

