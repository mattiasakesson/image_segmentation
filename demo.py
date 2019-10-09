from u_net import Unet_model
import pickle
import numpy as np


input_image = pickle.load(open('ultrasound_nerve_segmentation_data/input_image.p', "rb"))
mask = pickle.load(open('ultrasound_nerve_segmentation_data/mask.p', "rb"))

input_image = np.expand_dims(input_image,3)
mask = np.expand_dims(mask,3)

print("input image shape: ", input_image.shape)
print("mask shape: ", mask.shape)


model = Unet_model(input_shape=(420,580,1), con_layers=[25, 50, 100])

model.train(input_image,mask,verbose=1,batch_size=8)

