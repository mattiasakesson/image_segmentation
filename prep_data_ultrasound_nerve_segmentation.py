import numpy as np
import pickle
import time
from PIL import Image
import os
from matplotlib import pylab as plt

path = "ultrasound_nerve_segmentation_data/train"

im = []
mask = []
for f in os.listdir(path):

    f2 = f.split("_")
    if f2[-1] == 'mask.tif':
        mask_ = np.asarray(Image.open(os.path.join(path, f)))
        mask.append(mask_)
        im_name = f2[0] + "_" + f2[1] + ".tif"
        im_ = np.asarray(Image.open(os.path.join(path, im_name)))
        im.append(im_)




im = np.array(im)
mask = np.array(np.array(mask, dtype=np.bool_), dtype=np.int32)


print("im shape: ", im.shape)
print("mask shape: ", mask.shape)

pickle.dump(im, open('input_image.p', "wb"))
pickle.dump(mask, open('mask.p', "wb"))

# mask_ind = np.argsort(mask_names)
# print("mask_ind: ", mask_ind)
# mask_names = np.asarray(mask_names)[mask_ind]
#
# im_ind = np.argsort(im_names)
# im_names = np.asarray(im_names)[im_ind]
# print("im_names: ", im_names)
#
# for im_n, mask_n in zip(im_names,mask_names):
#     print(im_n, " - ", mask_n)
#
# print("im[0] type", type(im[0]))
# print("numpy? ", im.shape)
# print("mask_names: ", mask_names)

