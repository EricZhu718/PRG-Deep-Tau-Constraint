from stringprep import c22_specials
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from draw_bounding_box import calculate_phi,get_bounding_box
import os
import sys
import torch
import warnings
from data import SplineDataset, load_data_set
from model import Model
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from data_bounding_box_pred import image_dataset
import sys
import os
import cv2 as cv
import numpy as np
sys.path.append(os.path.abspath(''))
import torch.utils.tensorboard
from torch.utils.tensorboard import SummaryWriter
# from sklearn.linear_model import LinearRegression
import datetime
import os
import matplotlib
matplotlib.use('agg')

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
def custom_loss(phi_hat, deltas_and_times,image1, image2): # my_outputs are the phi output approximations, auxillary_info are the time and delta info
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

    product = torch.matmul(transpose, phi_and_time) # 2 by 2 matrix
  
    inverse = torch.inverse(product)
 
    Z_and_Z_vel = torch.matmul(torch.matmul(inverse, transpose), deltas) # first entry is estimated Z(0), second is estimated Z'(0)

    delta_accel_from_phi = torch.matmul(phi_and_time, Z_and_Z_vel)
    residues = torch.sub(delta_accel_from_phi, deltas) # difference between predicted delta values and true delta values
    residues = 1000*torch.norm(residues)**2 
    
    # print(Z_and_Z_vel)
    return residues, delta_accel_from_phi, deltas, Z_and_Z_vel[0], Z_and_Z_vel[1]  # returns the norm of the residue vector (ie square all the terms and add them together)

def least_squares(A:torch.tensor, b:torch.tensor) -> torch.tensor:
    if len(A.shape) == 1:
        A = A.reshape((A.shape[0], 1))
    if len(b.shape) == 1:
        b = b.reshape((b.shape[0], 1))
    x:torch.tensor = torch.matmul(torch.inverse((torch.matmul(torch.transpose(A, 0, 1), A))), torch.matmul(torch.transpose(A, 0, 1), b))

    return x

count = 0
global first_image
first_image = True
global c1, c2
c1, c2 = 0, 0


def train(Learning_rate = 1e-3,Epochs = 200, TrainingData = None, batch_size_train = 1):

    try:
        # Training dataloader
        if TrainingData is None:
            TrainingData = image_dataset()
        TrainLoader = DataLoader(TrainingData, batch_size_train)

        # actual training portion
        model = Model()
        # model.load_state_dict(torch.load('/home/tau/deep_tau_runs/2022-08-08_23:21:42.001165/weights/model_weight95.hdf5'))
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(),lr=Learning_rate)

        # # training loop

        for epoch in range(Epochs):

            def run_batch_train(img_batches, accel_batches, time_batches, delta_accel_batches,i,epoch) -> torch.tensor:

                batches_num = img_batches.shape[0]
                batch_loss = torch.tensor([0.0]).to(device)

                if first_image:

                    os.system('python3 yolov7/detect2.py \
                                --weights object_detector_weights/yolov7.pt \
                                --source ' + '/home/tau/Desktop/monocular_data/dataset-corridor1_512_16/processed_images/undistorted_images/'  + ('0' * (5-len(str(0)))) + str(0) + '.png')
                    
                    first_image = False

                    det = torch.load('~/Desktop/Tau_constaint/monocular_data/det_tensor.pt')

                    for *xyxy, conf, cls in reversed(det):

                        c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
            
                        crop_images1 = img_batches[0][c1[1]:c2[1],c1[0]:c2[0]]
                        crop_images2 = img_batches[1][c1[1]:c2[1],c1[0]:c2[0]]
                        crop_images = cv.vconcat([crop_images1,crop_images2])
                        # cv.imwrite(path + '/processed_images/cropped_images/' + ('0' * (5-len(str(i)))) + str(i) + '.png', crop_images)

                # make a tensor in the shape (batches * (frames-1), 144, 144)
                # img_pairs = torch.cat(tuple(img_batches[k] for k in range(batches_num)), dim = 0).float().to(device)
                img_pairs = torch.from_numpy(crop_images).to(device)
                # run all image pairs into the model
                phi_batch = model(img_pairs)
                # print(phi_batch.shape)
                # reshape tensor into shape (batches, frames-1)
                # phi_batch = torch.cat(tuple(phi_batch[j * phi_batch.shape[0] // batches_num: (j + 1) * phi_batch.shape[0] // batches_num] for \
                #         j in range(batches_num)), dim = 1).permute((1,0)).to(device)
                # print(phi_batch.shape)
                for j in range(batches_num): # for each batch in our batches
                    accel_arr = accel_batches[j].to(device)
                    time_arr = time_batches[j].to(device)
                    delta_accel_arr = delta_accel_batches[j].to(device)
                    deltas_and_time = torch.cat((torch.reshape(delta_accel_arr, (-1,1)), torch.reshape(time_arr, (-1,1))), 1).to(device)
                    
                    loss,delta_accel_from_phi,delta_accel_actual,predicted_depth, predicted_velocity \
                        = custom_loss((phi_batch[j])[:,None], deltas_and_time[1:],crop_images1,crop_images2)
                    global count
                    count += 1

                    # actual_depth  = z_zero[j].cpu().numpy()
                    content = '{}: {}: loss {:.4f} z_gt {:.4f} z_predicted {:.4f}'.format(epoch, batches_num*i+j, loss.item()/1000, actual_depth, predicted_depth.data[0])
                        
                    batch_loss += loss
                # counter = 0 

                return batch_loss/batches_num

            sum_of_train_loss = 0

            global count 
            count = 0
            # One epoch of training loop
            for i, data in enumerate(TrainLoader):
                
                img_batches, accel_batches, time_batches, delta_accel_batches = data
                # print(img_batches.shape)
                batch_loss = run_batch_train(img_batches, accel_batches, time_batches, delta_accel_batches,i,epoch)* batch_size_train
                batch_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                sum_of_train_loss += batch_loss.item()

            count = 0

            if epoch%5 == 0:
                torch.save(model.state_dict(), path + "weights/model_weight" + str(epoch) + ".hdf5")

            sum_of_train_loss = 0
            sum_of_val_loss = 0

    except RuntimeError as e:
        print(e)
        # python3 myscript.py 2>&1 | tee output.txt
        torch.cuda.empty_cache()

if __name__ == '__main__':

    path = '/home/tau/deep_tau_runs/' + str(datetime.datetime.now()).replace(' ', '_') + '/'
    os.mkdir(path)
    writer = SummaryWriter(path + '/'+ 'tensorboard_dir')
    # train(TrainingData = load_data_set("/home/tau/Video_Datasets/1000Videos200Frames/"),batch_size_train= 1, writer = writer, path=path, Frames = 200, Training_Video_Num=1000, Epochs=100)
    train(batch_size_train= 1, writer = writer, path=path, Frames = 200, Training_Video_Num=1000, Epochs=100)