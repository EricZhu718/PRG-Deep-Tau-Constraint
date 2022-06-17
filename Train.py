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
import time
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print(torch.cuda.is_available())
print(device)



# actual model:
class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        
        ###################
        ## Test Network ### #Shallow network for testing
        ###################
        
        self.layer1 = nn.Sequential(nn.Conv2d(6,32,3,padding=1),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU())                     
        self.layer2 = nn.Sequential(nn.Conv2d(32,32,3,padding=1),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(nn.Conv2d(32,32,3,padding=1),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(32,32,3,padding=1),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        self.layer5 = nn.Sequential(nn.Conv2d(32,64,3,padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())
        self.layer6 = nn.Sequential(nn.Conv2d(64,64,3,padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        self.layer7 = nn.Sequential(nn.Conv2d(64,64,3,padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())
        self.layer8 = nn.Sequential(nn.Conv2d(64,64,3,padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        # self.layer7 = nn.Sequential(nn.Conv2d(64,128,3,padding=1),
        #                             nn.BatchNorm2d(128),
        #                             nn.ReLU())
        
        # self.layer3 = nn.Sequential(nn.Conv2d(64,64,3,padding=1),
        #                             nn.BatchNorm2d(64),
        #                             nn.ReLU())
        # self.layer4 = nn.Sequential(nn.Conv2d(64,64,3,padding=1),
        #                             nn.BatchNorm2d(64),
        #                             nn.ReLU(),
        #                             nn.MaxPool2d(2))
        # self.layer5 = nn.Sequential(nn.Conv2d(64,128,3,padding=1),
        #                             nn.BatchNorm2d(128),
        #                             nn.ReLU())        
        # self.layer6 = nn.Sequential(nn.Conv2d(128,128,3,padding=1),
        #                             nn.BatchNorm2d(128),
        #                             nn.ReLU(),
        #                             nn.MaxPool2d(2))
        # self.layer7 = nn.Sequential(nn.Conv2d(128,128,3,padding=1),
        #                             nn.BatchNorm2d(128),
        #                             nn.ReLU())
        # self.layer8 = nn.Sequential(nn.Conv2d(128,128,3,padding=1),
        #                             nn.BatchNorm2d(128),
        #                             nn.ReLU())
        
        self.fc1 = nn.Linear(14080,1024)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1024,256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 64)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(64, 16)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(16, 1)
        self.relu5 = nn.ReLU()
        
        ######################
        ## Original Network###
        ######################
        
        # self.layer1 = nn.Sequential(nn.Conv2d(2,64,3,padding=1),
        #                             nn.BatchNorm2d(64),
        #                             nn.ReLU())                            
        # self.layer2 = nn.Sequential(nn.Conv2d(64,64,3,padding=1),
        #                             nn.BatchNorm2d(64),
        #                             nn.ReLU(),
        #                             nn.MaxPool2d(2))
        # self.layer3 = nn.Sequential(nn.Conv2d(64,64,3,padding=1),
        #                             nn.BatchNorm2d(64),
        #                             nn.ReLU())
        # self.layer4 = nn.Sequential(nn.Conv2d(64,64,3,padding=1),
        #                             nn.BatchNorm2d(64),
        #                             nn.ReLU(),
        #                             nn.MaxPool2d(2))
        # self.layer5 = nn.Sequential(nn.Conv2d(64,128,3,padding=1),
        #                             nn.BatchNorm2d(128),
        #                             nn.ReLU())        
        # self.layer6 = nn.Sequential(nn.Conv2d(128,128,3,padding=1),
        #                             nn.BatchNorm2d(128),
        #                             nn.ReLU(),
        #                             nn.MaxPool2d(2))
        # self.layer7 = nn.Sequential(nn.Conv2d(128,128,3,padding=1),
        #                             nn.BatchNorm2d(128),
        #                             nn.ReLU())
        # self.layer8 = nn.Sequential(nn.Conv2d(128,128,3,padding=1),
        #                             nn.BatchNorm2d(128),
        #                             nn.ReLU())
        # self.fc1 = nn.Linear(3*128*131*240,1024)
        # self.relu1 = nn.ReLU()
        # self.fc2 = nn.Linear(1024,8)
        # self.relu2 = nn.ReLU()
        # self.fc3 = nn.Linear(8, 1)
        # self.relu3 = nn.ReLU()

    def forward(self,x):
        
        ###################
        ## Test Network ### # Shallow network for testing
        ###################
        print("Hi")
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        # print(out.shape)
        out = out.view(-1,14080)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc5(out)
        out = self.relu5(out)
        return out
        
        ######################
        ## Original Network###
        ######################
        
        # out = self.layer1(x)
        # out = self.layer2(out)
        # out = self.layer3(out)
        # out = self.layer4(out)
        # out = self.layer5(out)
        # out = self.layer6(out)
        # out = self.layer7(out)
        # out = self.layer8(out)
        # # print(out.shape)
        # out = out.view(-1,3*128*131*240)
        # out = self.fc1(out)
        # out = self.relu1(out)
        # out = self.fc2(out)
        # out = self.relu2(out)
        # out = self.fc3(out)
        # out = self.relu3(out)
        # return out

# datasets using visual dataset class from Dataset.py
batch_size = 3
TrainingData = VisualDataset(64) # generate 1024 images (set to 64 for testing)
ValidationData = VisualDataset(64) # generate 64 images
TestData = VisualDataset(64) # generate 64 images

# add dataloaders to datasets
TrainLoader = DataLoader(TrainingData, batch_size)
ValidationLoader = DataLoader(ValidationData, batch_size)
TestLoader = DataLoader(TestData, batch_size)

# custom loss function,
# the my_outputs is a tensor matrix with only 1 column and it represents the phi predictions from our NN
# deltas_and_times is a tensor matrix with 2 columns: the first one is double integral of acceleration, the second are the time values 
def custom_loss(my_outputs, deltas_and_times): # my_outputs are the phi output approximations, auxillary_info are the time and delta info
    print('Loss function input checking')
    print(deltas_and_times.shape)
    my_outputs.reshape(my_outputs.size()[0], 1)
    deltas = deltas_and_times[:,0]
    deltas = deltas.reshape(deltas.size()[0], 1) # delta is a single column of values
    times = deltas_and_times[:,1]
    times = times.reshape(times.size()[0], 1) # times is a single column of values

    phi_and_time = torch.cat((torch.sub(my_outputs, 1.0), torch.multiply(times, -1)), 1) # make a matrix where first column is phi-1, second column is -time

    # solve the least squares for Z(0) and Z'(0)
    transpose = torch.transpose(phi_and_time, 0, 1)
    product = torch.matmul(transpose, phi_and_time) # 2 by 2 matrix
    inverse = torch.inverse(product)
    Z_and_Z_vel = torch.matmul(torch.matmul(inverse, transpose), deltas) # first entry is estimated Z(0), second is estimated Z'(0)
    
    residues = torch.sub(torch.matmul(phi_and_time, Z_and_Z_vel), deltas) # difference between predicted delta values and true delta values

    return torch.norm(residues)**2 # returns the norm of the residue vector (ie square all the terms and add them together)


# # example of the format for the tensors for the loss function:
# sample_phi_outputs = torch.tensor([[0], [1], [2], [15], [10]], dtype = float)
# sample_deltas_and_times = torch.tensor([[1,2], [5,4], [4,6], [13, 20], [20, 10]], dtype = float)
# print("sample loss output from sample loss inputs: " + str(custom_loss(sample_phi_outputs, sample_deltas_and_times)))


# actual training portion
epochs = 10
criterion = custom_loss
model = Model().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

for epoch in range(epochs):
    print("epoch: " + str(epoch))
    for i, data in enumerate(TrainLoader):
        # print(i)
        img_batches, accel_batches, time_batches, delta_accel_batches = data
        # print(img_batches.shape)
        # print(accel_batches.shape)
        for j in range(batch_size):
            # for each data set in the batch
            img_arr = img_batches[j]
            accel_arr = accel_batches[j]
            time_arr = time_batches[j]
            delta_accel_arr = delta_accel_batches[j]

            # print(img_arr.shape)
            # print(accel_arr.shape)
            # print(time_arr.shape)
            # print(delta_accel_arr.shape)

            delta_accel_arr_reshape = torch.reshape(delta_accel_arr, (-1,1))
            time_arr_reshape = torch.reshape(time_arr, (-1,1))

            deltas_and_time = torch.cat((delta_accel_arr_reshape, time_arr_reshape), 1)
            # print(delats_and_time.shape)
            print("i: " + str(i))
            optimizer.zero_grad()
            img_arr = img_arr.to(device)
            # print(img_arr.shape)
            # print(accel_arr.shape)
            # print(time_arr.shape)
            # print(delta_accel_arr.shape)

            accel_arr = accel_arr.to(device)
            time_arr = time_arr.to(device)
            delta_accel_arr = delta_accel_arr.to(device)
            # delats_and_time = delats_and_time.to(device)
            phi_estimates = []
            # print(img_arr.shape)
    #         break
    #     break
    # break 
            a = 0
            start_time = time.time()
            for k in range(1, len(img_arr)):
                if a == 10:
                    break
                print("a = " + str(a))
            #     # print(img_arr[j].shape)
                
                
                double_img = torch.cat((img_arr[0], img_arr[k]),2)
                
                double_img = torch.reshape(double_img,(1,double_img.shape[2],double_img.shape[0], double_img.shape[1]))
                
                # double_img = double_img.permute(3,0,1,2)
                
                # print(double_img.float())
                output = model(double_img.float())
                # end_time = time.time()
                # print("Time to run the code" +" : " + str(end_time-start_time))
                break     
                # phi_estimates.append(output)
                
                a += 1
                
            # phi_estimates = torch.tensor(phi_estimates)
            # loss = criterion(phi_estimates, deltas_and_time[1:])
            break
        break
    break

            # loss.backward()
            # optimizer.step()
            # if (i+1) % len(TrainLoader) == 0:
            #     print('Train Epoch: [{}/{}] [{}/{} ({:.0f}%)]\Mean Squared Error: {:.6f}'.format(
            #         epoch+1,epochs, i , len(TrainLoader),
            #         100. * i / len(TrainLoader), loss))

