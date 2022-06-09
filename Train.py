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
from Dataset import VisualDataset
from torch.utils.data import Dataset, DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(2,64,3,padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())
                                    
        self.layer2 = nn.Sequential(nn.Conv2d(64,64,3,padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(nn.Conv2d(64,64,3,padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(64,64,3,padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        self.layer5 = nn.Sequential(nn.Conv2d(64,128,3,padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())        
        self.layer6 = nn.Sequential(nn.Conv2d(128,128,3,padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        self.layer7 = nn.Sequential(nn.Conv2d(128,128,3,padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())
        self.layer8 = nn.Sequential(nn.Conv2d(128,128,3,padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())
        self.fc1 = nn.Linear(128*16*16,1024)
        self.fc2 = nn.Linear(1024,8)
        self.fc3 = nn.Linear(8, 1)
        
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = out.view(-1,128* 16* 16)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


batch_size = 32
TrainingData = VisualDataset(64) # generate 1024 images (set to 64 for testing)
ValidationData = VisualDataset(64) # generate 64 images
TestData = VisualDataset(64) # generate 64 images

TrainLoader = DataLoader(TrainingData,batch_size)
ValidationLoader = DataLoader(ValidationData,batch_size)
TestLoader = DataLoader(TestData,batch_size)


# custom loss function,
# the my_outputs is a tensor matrix with 1 column representing phi predictions from our NN
# deltas_and_times is a tensor matrix with 2 columns: the first one is double integral of acceleration, the second are the time values 
def custom_loss(my_outputs, deltas_and_times): # my_outputs are the phi output approximations, auxillary_info are the time and delta info
    my_outputs.reshape(my_outputs.size()[0], 1)
    deltas = deltas_and_times[:,0]
    deltas = deltas.reshape(deltas.size()[0], 1) # delta is a single column of values
    times = deltas_and_times[:,1]
    times = times.reshape(times.size()[0], 1) # times is a single column of values

    phi_and_time = torch.cat((torch.sub(my_outputs, 1.0), times), 1) # make a matrix where first column is phi-1, second column is time

    # solve the least squares for Z(0) and Z'(0)
    transpose = torch.transpose(phi_and_time, 0, 1)
    product = torch.matmul(transpose, phi_and_time) # 2 by 2 matrix
    inverse = torch.inverse(product)

    print(torch.matmul(inverse, transpose))
    print(deltas)
    Z_and_Z_vel = torch.matmul(torch.matmul(inverse, transpose), deltas) # first entry is estimated Z(0), second is estimated Z'(0)
    
    residues = torch.sub(torch.matmul(phi_and_time, Z_and_Z_vel), deltas) # difference between predicted delta values and true delta values

    return torch.norm(residues) # returns the norm of the residue vector

sample_phi_outputs = torch.tensor([[0], [1], [2], [15], [10]], dtype = float)
sample_deltas_and_times = torch.tensor([[1,2], [5,4], [4,6], [13, 20], [20, 10]], dtype = float)

print(custom_loss(sample_phi_outputs, sample_deltas_and_times))



epochs = 10
criterion = custom_loss
