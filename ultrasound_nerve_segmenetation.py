import numpy as np
import pickle
import time
from PIL import Image
import os
from matplotlib import pylab as plt

path = "C:/Users/ma10s/Documents/project/ultrasound-nerve-segmentation/train"

im = []
im_names = []
mask = []
mask_names = []
i = 0
for f in os.listdir(path):
    if i<100:
        f2 = f.split("_")
        # print("f2: ", f2)
        if f2[-1] == 'mask.tif':
            mask.append(Image.open(os.path.join(path, f)))
            im_name = f2[0] + "_" + f2[1] + ".tif"
            im.append(Image.open(os.path.join(path, im_name)))


    i+=1

im = np.array([np.asarray(image)for image in im])
mask = np.array(np.array([np.asarray(image)for image in mask],dtype=np.bool_),dtype=np.int32)

print("im shape: ", im.shape)
print("mask shape: ", mask.shape)

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

index=4
mask_0 = np.argwhere(mask[index]!=0)
print("mask_0 shape: ", mask_0.shape)
f, ax = plt.subplots(2,2,figsize=(20,20))
ax[0,0].imshow(im[index], cmap='Greys')
ax[0,1].imshow(mask[index])
ax[1,1].imshow(im[index])
ax[1,1].scatter(mask_0[:,1],mask_0[:,0], c='r', alpha=0.01)


plt.show()
print("mask: ", np.unique(mask[0]))

