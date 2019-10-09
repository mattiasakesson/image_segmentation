from u_net import Unet_model
import pickle
import numpy as np


input_image = pickle.load(open('ultrasound_nerve_segmentation_data/input_image.p', "rb"))
mask = pickle.load(open('ultrasound_nerve_segmentation_data/mask.p', "rb"))

imput_image = np.expand_dims(input_image,3)
mask = np.expand_dims(mask,3)

print("imput image shape: ", imput_image.shape)
print("mask shape: ", mask.shape)


model = Unet_model(input_shape=(420,580,1), con_layers=[25, 50, 100])

model.train(input_image,mask)

