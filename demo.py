from u_net import Unet_model
import pickle
import numpy as np


input_image = pickle.load(open('ultrasound_nerve_segmentation_data/input_image.p', "rb"))
mask = pickle.load(open('ultrasound_nerve_segmentation_data/mask.p', "rb"))

input_image = np.expand_dims(input_image,3)
mask = np.expand_dims(mask,3)

print("input image shape: ", input_image.shape)
print("mask shape: ", mask.shape)

mask_fraction = np.sum(mask)/mask.size

print("mask fraction: ", mask_fraction)


train_input = input_image[:5000]
train_output = mask[:5000]
val_input = input_image[5000:]
val_output = mask[5000:]


model = Unet_model(input_shape=(420,580,1), con_layers=[32,64,128,248])

model.train(train_input,train_output,val_input,val_output, verbose=1,batch_size=8)

