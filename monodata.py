class SplineDataset(Dataset):
    def __init__(self, data_size, frames = 100):
      global random_values_from_0_to_1
      global random_time_index
      self.size = data_size
      self.data = []
      
      random_values_from_0_to_1 = np.random.rand(data_size * 20)
      random_time_index = 0
      
      for i in range(data_size): 
        if i % 250 == 0:
          print('generated ' + str(i) + ' videos')
        img_arr, accel_arr, time_arr, delta_accel_arr, z_zero, z_dot_zero \
            = (get_spline_video(num=8, frames = frames))
        img_arr = torch.tensor(img_arr)
        arr_of_doubles = []

        for k in range(1, len(img_arr)):
            double_img = torch.cat((img_arr[0], img_arr[k]),2)
            double_img = torch.reshape(double_img,(1,double_img.shape[2],double_img.shape[0], double_img.shape[1]))
            arr_of_doubles.append(double_img)

        tensor_of_doubles = torch.cat(tuple(double for double in arr_of_doubles), 0)

        self.data.append((tensor_of_doubles, accel_arr, time_arr, delta_accel_arr, z_zero, z_dot_zero))

    def __len__(self):
        return self.size
    
    # Note: if you want the same video every time, might be best to put
    # the code in the __init__ func so it doesn't regenerate the 
    # video turning every call of __getitem__
    
    def __getitem__(self, index):
        return self.data[index]
