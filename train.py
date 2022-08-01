from cgi import print_arguments
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
# from __future__ import print_function
from torch.autograd import Variable
# import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print(torch.cuda.is_available())
print(device)

x = []
y = []

# actual model:
class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        
        ###################
        ## Test Network ### #Shallow network for testing
        ###################
        
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
        
        self.fc1 = nn.Linear(4096,1024)
        self.relu1 = nn.Tanh()
        self.fc2 = nn.Linear(1024,64)
        self.relu2 = nn.Tanh()
        self.fc3 = nn.Linear(64, 1)
        self.relu3 = nn.Tanh()
        # self.fc4 = nn.Linear(64, 16)
        # self.relu4 = nn.ReLU()
        # self.fc5 = nn.Linear(16, 1)
        # self.relu5 = nn.ReLU()
        
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
        # print("Hi")
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        # print(out.shape)
        # out = out.view(-1,4096)
        out = self.fc1(out.flatten())
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        # out = self.relu3(out)
        # out = self.fc4(out)
        # out = self.relu4(out)
        # out = self.fc5(out)
        # out = self.relu5(out)
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
batch_size = 1
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
    
    # my_outputs.requires_grad = True
    # return my_outputs[0]
    
    ## Taking average of output 
    # avg = torch.mean(my_outputs)
    # avg.requires_grad = True
    
    # print('Loss function input checking')
    # print(deltas_and_times.shape)
    my_outputs.reshape(my_outputs.size()[0], 1)
    deltas = deltas_and_times[:,0]
    deltas = deltas.reshape(deltas.size()[0], 1) # delta is a single column of values
    times = deltas_and_times[:,1]
    times = times.reshape(times.shape[0],1)
 
    # times = times.reshape(times.size()[0], 1) # times is a single column of values

    # ## Taking average of output 
    # avg = torch.mean(my_outputs)
    # avg.requires_grad = True
    
    # print(torch.sub(my_outputs, 1.0).shape)
    # print(torch.multiply(times, -1).shape)
    phi_and_time = torch.cat((torch.sub(my_outputs, 1.0), torch.multiply(times, -1)), 1) # make a matrix where first column is phi-1, second column is -time

    # solve the least squares for Z(0) and Z'(0)
    transpose = torch.transpose(phi_and_time, 0, 1)
    product = torch.matmul(transpose, phi_and_time) # 2 by 2 matrix
    inverse = torch.inverse(product)
    Z_and_Z_vel = torch.matmul(torch.matmul(inverse, transpose), deltas) # first entry is estimated Z(0), second is estimated Z'(0)
    
    residues = torch.sub(torch.matmul(phi_and_time, Z_and_Z_vel), deltas) # difference between predicted delta values and true delta values
    residues = torch.norm(residues)**2 
    # residues.requires_grad = True
    # residues = [residues]
    return residues # returns the norm of the residue vector (ie square all the terms and add them together)
    # return avg
    # my_outputs.requires_grad = True
    # return my_outputs[0]
    

## loss function without convering into tensor

# custom loss function,
# the my_outputs is a tensor matrix with only 1 column and it represents the phi predictions from our NN
# deltas_and_times is a tensor matrix with 2 columns: the first one is double integral of acceleration, the second are the time values 
def loss_function(my_outputs, deltas_and_times): # my_outputs are the phi output approximations, auxillary_info are the time and delta info
    
    # return my_outputs[0]
    
    # print('Loss function input checking')
    # print(deltas_and_times.shape)
    # my_outputs.reshape(my_outputs.size()[0], 1)
    deltas = deltas_and_times[:,0]
    deltas = deltas.reshape(deltas.size()[0], 1) # delta is a single column of values
    times = deltas_and_times[:,1]
    times = times.reshape(times.shape[0],1)
 
    # times = times.reshape(times.size()[0], 1) # times is a single column of values

    # ## Taking average of output 
    # my_outputs.detach().numpy()
    # avg = np.mean(my_outputs)
    # avg.requires_grad = True
    
    # print(torch.sub(my_outputs, 1.0).shape)
    # print(torch.multiply(times, -1).shape)
    phi_and_time = torch.cat((torch.sub(my_outputs, 1.0), torch.multiply(times, -1)), 1) # make a matrix where first column is phi-1, second column is -time

    # solve the least squares for Z(0) and Z'(0)
    transpose = torch.transpose(phi_and_time, 0, 1)
    product = torch.matmul(transpose, phi_and_time) # 2 by 2 matrix
    inverse = torch.inverse(product)
    Z_and_Z_vel = torch.matmul(torch.matmul(inverse, transpose), deltas) # first entry is estimated Z(0), second is estimated Z'(0)
    
    residues = torch.sub(torch.matmul(phi_and_time, Z_and_Z_vel), deltas) # difference between predicted delta values and true delta values
    residues = torch.norm(residues)**2 
    # residues.requires_grad = True
    # residues = [residues]
    return residues # returns the norm of the residue vector (ie square all the terms and add them together)
    # return avg
    # my_outputs.requires_grad = True
    # return my_outputs[0]

# # example of the format for the tensors for the loss function:
sample_phi_outputs = torch.tensor([[0], [1], [2], [15], [10]], dtype = float)
sample_deltas_and_times = torch.tensor([[1,2], [5,4], [4,6], [13, 20], [20, 10]], dtype = float)
print("sample loss output from sample loss inputs: " + str(custom_loss(sample_phi_outputs, sample_deltas_and_times)))


# actual training portion
epochs = 200
criterion = custom_loss
model = Model().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.0000001, momentum=0.99)
# for params in model.parameters():
#     print(params.requires_grad)
loss = torch.zeros(1)
loss.requires_grad = True
a = 0
for epoch in range(epochs):
    print("epoch: " + str(epoch))
    for i, data in enumerate(TrainLoader):
        optimizer.zero_grad()

        img_batches, accel_batches, time_batches, delta_accel_batches = data
        # print(batch_size)
        for j in range(batch_size):

            img_arr = img_batches[j]
            accel_arr = accel_batches[j]
            time_arr = time_batches[j]
            delta_accel_arr = delta_accel_batches[j]

            delta_accel_arr_reshape = torch.reshape(delta_accel_arr, (-1,1))
            time_arr_reshape = torch.reshape(time_arr, (-1,1))

            deltas_and_time = torch.cat((delta_accel_arr_reshape, time_arr_reshape), 1)

            # print("i: " + str(i))
            # optimizer.zero_grad()
            # img_arr = img_arr.to(device)

            # accel_arr = accel_arr.to(device)
            # time_arr = time_arr.to(device)
            # delta_accel_arr = delta_accel_arr.to(device)
            # delats_and_time = delats_and_time.to(device)
            phi_estimates = []

            start_time = time.time()
            i = 0
            for k in range(1, len(img_arr)):  
                
                double_img = torch.cat((img_arr[0], img_arr[k]),2)
                
                double_img = torch.reshape(double_img,(1,double_img.shape[2],double_img.shape[0], double_img.shape[1]))
                # double_img = Variable(double_img, requires_grad = True)
                output = model(double_img.float())
                end_time = time.time()
                # print("Time to run the code frame " + str(k) + " : " + str(end_time-start_time))
                
                phi_estimates.append(output)
                
                # backPropogating first element of the output
                
                # if i == 0:
                #     print('Hi')
                #     print(output)
                 
                # i += 1
            # phi_estimates[0] = abs(phi_estimates[0])
            # phi_estimates[0].backward()
            # print(len(phi_estimates))
            # print(phi_estimates)
            phi_estimates = torch.reshape(torch.cat(phi_estimates),(len(phi_estimates),1))
            # print(phi_estimates)
            # phi_estimates[0] = abs(phi_estimates[0])
            # phi_estimates.requires_grad = True
            # # phi_estimates[0] = abs(phi_estimates[0])
            # phi_estimates[0].backward()
            
            # print("Phi")
            # print(phi_estimates)
            loss = custom_loss(phi_estimates, deltas_and_time[1:])
            print("training number: " + str(a) + "  "+ 'loss: ' + str(loss))
            # print(model.)
            # loss.retain_grad()
            loss.backward()
            # print(loss)
            # print('hi')       
            # loss.register_hook(lambda grad: print(grad))
            # print(model.fc3.weight)
            # print_loss = copy.deepcopy(loss)
            # print_loss.numpy()
            optimizer.step()
            # x.append(i)
            # y.append(print_loss)
            # plt.plot(x, y, c = 'green')
            # plt.xlabel('No of times NN tranined with same video')
            # plt.ylabel('Loss')
            # plt.title("loss vs No of time NN trained")
            # plt.show()      
            a += 1
            # print(model.fc3.weight)

            # loss.register_hook(lambda grad: print(grad))
            
            # if (i+1) % len(TrainLoader) == 0:
            #     print('Train Epoch: [{}/{}] [{}/{} ({:.0f}%)]\Mean Squared Error: {:.6f}'.format(
            #         epoch+1,epochs, i , len(TrainLoader),
            #         100. * i / len(TrainLoader), loss))

