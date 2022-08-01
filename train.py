from glob import glob
from matplotlib import pyplot as plt
import cv2
import random
import numpy as np
from numpy.linalg import inv
import torch
from torch import Tensor
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
import warnings

from data import SplineDataset, load_data_set
from model import Model
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt


import sys
import os
import math
import numpy as np
import random
sys.path.append(os.path.abspath(''))
import torch.utils.tensorboard
from torch.utils.tensorboard import SummaryWriter
from sklearn.linear_model import LinearRegression
import datetime
import os

# access cuda
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
print(device)


# custom loss function,
# the my_outputs is a tensor matrix with only 1 column and it represents the phi predictions from our NN
# deltas_and_times is a tensor matrix with 2 columns: the first one is double integral of acceleration, the second are the time values 
def custom_loss(phi_hat, deltas_and_times): # my_outputs are the phi output approximations, auxillary_info are the time and delta info
    # print("hi")
    # print(my_outputs.size())
    phi_hat.reshape(phi_hat.size()[0], 1)
    deltas = deltas_and_times[:,0]
    deltas = deltas.reshape(deltas.size()[0], 1) # delta is a single column of values
    times = deltas_and_times[:,1]
    times = times.reshape(times.shape[0], 1)
    
    # times = times.reshape(times.size()[0], 1) # times is a single column of values
  
    phi_and_time = torch.cat((torch.sub(phi_hat, 1), torch.multiply(times, -1)), 1) # make a matrix where first column is phi-1, second column is -time

    # solve the least squares for Z(0) and Z'(0)
    transpose = torch.transpose(phi_and_time, 0, 1)
    # print("transpose: ")
    # print(transpose)
    product = torch.matmul(transpose, phi_and_time) # 2 by 2 matrix
    # print("product: ")
    # print(product)   
    inverse = torch.inverse(product)
    # print("inverse")
    # print(inverse)
    Z_and_Z_vel = torch.matmul(torch.matmul(inverse, transpose), deltas) # first entry is estimated Z(0), second is estimated Z'(0)
    # print("Z_and_Z_vel")
    # print(Z_and_Z_vel)
    # print("Hi")
    # print(Z_and_Z_vel)

    # Z_and_Z_vel_actual = torch.tensor([[np.double(3.0)],[np.double(0.0)]]).to(device)
    # Z_vel = torch.tensor(Z_and_Z_vel[1])

    delta_accel_from_phi = torch.matmul(phi_and_time, Z_and_Z_vel)
    residues = torch.sub(delta_accel_from_phi, deltas) # difference between predicted delta values and true delta values
    residues = torch.norm(residues)**2 
    
    # print(Z_and_Z_vel)
    return residues*1000, delta_accel_from_phi, deltas, Z_and_Z_vel[0], Z_and_Z_vel[1]  # returns the norm of the residue vector (ie square all the terms and add them together)

def least_squares(A:torch.tensor, b:torch.tensor) -> torch.tensor:
    if len(A.shape) == 1:
        A = A.reshape((A.shape[0], 1))
    if len(b.shape) == 1:
        b = b.reshape((b.shape[0], 1))
    x:torch.tensor = torch.matmul(torch.inverse((torch.matmul(torch.transpose(A, 0, 1), A))), torch.matmul(torch.transpose(A, 0, 1), b))

    return x




def train(Training_Video_Num = 1000, Learning_rate = 1e-3, Frames = 100, \
    Epochs = 250, TrainingData = None, ValidationData = None, batch_size_train = 2, \
        batch_size_val = 2):
    
    try:
        # Training dataloader
        if TrainingData is None:
            TrainingData = SplineDataset(Training_Video_Num, frames = Frames)
        TrainLoader = DataLoader(TrainingData, batch_size_train)

        # actual training portion
        model = Model().to(device)
        optimizer = optim.Adam(model.parameters(),lr=Learning_rate)

        # Validation dataloader
        if ValidationData is None:
            ValidationData = load_data_set('/home/tau/Video_Datasets/250Videos100Frames') # generate videos for validation
        ValidationLoader = DataLoader(ValidationData, batch_size_val)

        # set up directory for the run
        path = '../deep_tau_runs/' + str(datetime.datetime.now()).replace(' ', '_') + '/'
        os.mkdir(path)
        os.mkdir(path + 'validation_set')
        os.mkdir(path + 'training_set')
        os.mkdir(path + 'weights')
        os.mkdir(path + 'tensorboard_dir')

        # set up tensorboard writer
        writer = SummaryWriter(log_dir = path + 'tensorboard_dir')
        writer.add_text("training params:", 'lr: ' + str(Learning_rate) + '\nframes: ' + str(Frames) +'\nVideos: ' + str(Training_Video_Num)+'\nBatch size: ' + str(batch_size_train))
        print('Command to view tensorboard is:\ntensorboard --logdir ' + writer.log_dir)
        counter= 0

        validation_file = open(path + "validation.txt","w")
        training_file = open(path + "training.txt","w")


        # training loop
        for epoch in range(Epochs):
            def run_batch_train(img_batches, accel_batches, time_batches, delta_accel_batches, z_zero, z_dot_zero) -> torch.tensor:
                batches_num = img_batches.shape[0]
                batch_loss = torch.tensor([0.0]).to(device)

                # make a tensor in the shape (batches * (frames-1), 144, 144)
                img_pairs = torch.cat(tuple(img_batches[k] for k in range(batches_num)), dim = 0).float().to(device)
                
                # run all image pairs into the model
                phi_batch = model(img_pairs)

                # reshape tensor into shape (batches, frames-1)
                phi_batch = torch.cat(tuple(phi_batch[j * phi_batch.shape[0] // \
                    batches_num: (j + 1) * phi_batch.shape[0] // batches_num] for \
                        j in range(batches_num)), dim = 1).permute((1,0)).to(device)
                
                for j in range(batches_num): # for each batch in our batches
                    accel_arr = accel_batches[j].to(device)
                    time_arr = time_batches[j].to(device)
                    delta_accel_arr = delta_accel_batches[j].to(device)
                    deltas_and_time = torch.cat((torch.reshape(delta_accel_arr, (-1,1)), torch.reshape(time_arr, (-1,1))), 1).to(device)
                    
                    loss,delta_accel_from_phi,delta_accel_actual,predicted_depth, predicted_velocity \
                        = custom_loss((phi_batch[j])[:,None], deltas_and_time[1:])
                    batch_loss += loss
                return batch_loss
            
            def run_batch_val(img_batches, accel_batches, time_batches, delta_accel_batches, z_zero, z_dot_zero) -> torch.tensor:
                with torch.no_grad():
                    batches_num = img_batches.shape[0]
                    batch_loss = torch.tensor([0.0]).to(device)

                    # make a tensor in the shape (batches * (frames-1), 144, 144)
                    img_pairs = torch.cat(tuple(img_batches[k] for k in range(batches_num)), dim = 0).float().to(device)
                    
                    # run all image pairs into the model
                    phi_batch = model(img_pairs)

                    # reshape tensor into shape (batches, frames-1)
                    phi_batch = torch.cat(tuple(phi_batch[j * phi_batch.shape[0] // \
                        batches_num: (j + 1) * phi_batch.shape[0] // batches_num] for \
                            j in range(batches_num)), dim = 1).permute((1,0)).to(device)
                    
                    for j in range(batches_num): # for each batch in our batches
                        accel_arr = accel_batches[j].to(device)
                        time_arr = time_batches[j].to(device)
                        delta_accel_arr = delta_accel_batches[j].to(device)
                        deltas_and_time = torch.cat((torch.reshape(delta_accel_arr, (-1,1)), \
                            torch.reshape(time_arr, (-1,1))), 1).to(device)
                        
                        loss,delta_accel_from_phi,delta_accel_actual,predicted_depth, predicted_velocity \
                            = custom_loss((phi_batch[j])[:,None], deltas_and_time[1:])
                        batch_loss += loss

                return batch_loss

            
            sum_of_train_loss = 0
            sum_of_val_loss = 0
            
            
            # One epoch of training loop
            for i, data in enumerate(TrainLoader):

                img_batches, accel_batches, time_batches, delta_accel_batches, z_zero, z_dot_zero = data
                
                batch_loss = run_batch_train(img_batches, accel_batches, time_batches, delta_accel_batches, z_zero, z_dot_zero)
                batch_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                sum_of_train_loss += batch_loss.item()

            # One epoch of validation loop
            for i, data in enumerate(ValidationLoader):
                
                img_batches, accel_batches, time_batches, delta_accel_batches, z_zero, z_dot_zero = data
                
                batch_loss = run_batch_val(img_batches, accel_batches, time_batches, delta_accel_batches, z_zero, z_dot_zero)

                sum_of_val_loss += batch_loss.item()

            print('epoch: {} avg_train_loss: {:.4f} avg_val_loss: {:.4f}'\
                .format(epoch, sum_of_train_loss / (len(TrainLoader) * batch_size_train), \
                    sum_of_val_loss / (len(ValidationLoader) * batch_size_val)))
            writer.add_scalars('Losses during training', {'avg training loss':sum_of_train_loss / (len(TrainLoader) * batch_size_train),
                                        'avg validation loss':sum_of_val_loss / (len(ValidationLoader) * batch_size_val)}, epoch)
            sum_of_train_loss = 0
            sum_of_val_loss = 0

    except RuntimeError as e:
        print(e)
        # python3 myscript.py 2>&1 | tee output.txt
        torch.cuda.empty_cache()
    
    torch.cuda.empty_cache()
    writer.close()
    validation_file.close()
    training_file.close()

if __name__ == '__main__':
    
    # train(TrainingData = load_data_set("../Video_Datasets/1000Videos200Frames/"),batch_size_train= 1)
    train(TrainingData = load_data_set( "../Video_Datasets/1000Videos150Frames/"), batch_size_train=1)
    train(TrainingData = load_data_set( "../Video_Datasets/750Videos150Frames/"), batch_size_train=1)
    train(TrainingData = load_data_set( "../Video_Datasets/500Videos150Frames/"), batch_size_train=1)
    train(TrainingData = load_data_set( "../Video_Datasets/250Videos150Frames/"), batch_size_train=1)
    train(TrainingData = load_data_set( "../Video_Datasets/100Videos150Frames/"), batch_size_train=1)
    train(TrainingData = load_data_set ("../Video_Datasets/50Videos150Frames/"), batch_size_train=1)
    train(TrainingData = load_data_set ("../Video_Datasets/20Videos150Frames/"), batch_size_train=1)

    train(TrainingData = load_data_set ("../Video_Datasets/1000Videos100Frames/"), batch_size_train=2)
    train(TrainingData = load_data_set( "../Video_Datasets/750Videos100Frames/"), batch_size_train=2)
    train(TrainingData = load_data_set( "../Video_Datasets/500Videos100Frames/"), batch_size_train=2)
    train(TrainingData = load_data_set( "../Video_Datasets/250Videos100Frames/"), batch_size_train=2)
    train(TrainingData = load_data_set( "../Video_Datasets/100Videos100Frames/"), batch_size_train=2)
    train(TrainingData = load_data_set( "../Video_Datasets/50Videos100Frames/"), batch_size_train=2)
    train(TrainingData = load_data_set( "../Video_Datasets/20Videos100Frames/"), batch_size_train=2)

    # train(TrainingData = load_data_set( "../Video_Datasets/1000Videos50Frames/"), batch_size_train=4)
    # train(TrainingData = load_data_set( "../Video_Datasets/750Videos50Frames/"), batch_size_train=4)
    # train(TrainingData = load_data_set( "../Video_Datasets/500Videos50Frames/"), batch_size_train=4)
    # train(TrainingData = load_data_set( "../Video_Datasets/250Videos50Frames/"), batch_size_train=4)
    # train(TrainingData = load_data_set( "../Video_Datasets/100Videos50Frames/"), batch_size_train=4)
    # train(TrainingData = load_data_set("../Video_Datasets/50Videos50Frames/"), batch_size_train=4)
    # train(TrainingData = load_data_set("../Video_Datasets/20Videos50Frames/"), batch_size_train=4)

    # train(TrainingData = load_data_set( "../Video_Datasets/1000Videos20Frames/"), batch_size_train=8)
    # train(TrainingData = load_data_set( "../Video_Datasets/750Videos20Frames/"), batch_size_train=8)
    # train(TrainingData = load_data_set( "../Video_Datasets/500Videos20Frames/"), batch_size_train=8)
    # train(TrainingData = load_data_set( "../Video_Datasets/250Videos20Frames/"), batch_size_train=8)
    # train(TrainingData = load_data_set( "../Video_Datasets/100Videos20Frames/"), batch_size_train=8)
    # train(TrainingData = load_data_set("../Video_Datasets/50Videos20Frames/"), batch_size_train=8)
    # train(TrainingData = load_data_set("../Video_Datasets/20Videos20Frames/"), batch_size_train=8)

    # train(TrainingData = load_data_set( "../Video_Datasets/1000Videos10Frames/"), batch_size_train=16)
    # train(TrainingData = load_data_set( "../Video_Datasets/750Videos10Frames/"), batch_size_train=16)
    # train(TrainingData = load_data_set( "../Video_Datasets/500Videos10Frames/"), batch_size_train=16)
    # train(TrainingData = load_data_set( "../Video_Datasets/250Videos10Frames/"), batch_size_train=16)
    # train(TrainingData = load_data_set( "../Video_Datasets/100Videos10Frames/"), batch_size_train=16)
    # train(TrainingData = load_data_set("../Video_Datasets/50Videos10Frames/"), batch_size_train=16)
    # train(TrainingData = load_data_set("../Video_Datasets/20Videos10Frames/"), batch_size_train=16)    