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
import math
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
import pytorch3d
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
# from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)
import sys
import os
import math
import numpy as np
import gdown
import random
sys.path.append(os.path.abspath(''))
import torch.utils.tensorboard
from torch.utils.tensorboard import SummaryWriter
from sklearn.linear_model import LinearRegression




class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        
        
        self.layer1 = nn.Sequential(nn.Conv2d(6,32,3,padding=1),
                                    nn.BatchNorm2d(32),
                                    nn.Tanh())                     
        self.layer2 = nn.Sequential(nn.Conv2d(32,32,3,padding=1),
                                    nn.BatchNorm2d(32),
                                    nn.Tanh(),
                                    nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(nn.Conv2d(32,32,3,padding=1),
                                    nn.BatchNorm2d(32),
                                    nn.Tanh())
        self.layer4 = nn.Sequential(nn.Conv2d(32,32,3,padding=1),
                                    nn.BatchNorm2d(32),
                                    nn.Tanh(),
                                    nn.MaxPool2d(2))
        self.layer5 = nn.Sequential(nn.Conv2d(32,64,3,padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.Tanh())
        self.layer6 = nn.Sequential(nn.Conv2d(64,64,3,padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.Tanh(),
                                    nn.MaxPool2d(2))
        self.layer7 = nn.Sequential(nn.Conv2d(64,64,3,padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.Tanh())
        self.layer8 = nn.Sequential(nn.Conv2d(64,64,3,padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.Tanh(),
                                    nn.MaxPool2d(2))

        
        self.fc1 = nn.Linear(4096,1024)
        self.relu1 = nn.Tanh()
        self.fc2 = nn.Linear(1024,64)
        self.relu2 = nn.Tanh()
        self.fc3 = nn.Linear(64, 3)

    def forward(self,x):
        
        ###################
        ## Test Network ### # Shallow network for testing
        ###################
        # print("Hi")
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.fc1(out.reshape((len(out),) + (torch.numel(out[0]),)))
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out
        