import imageio 
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    test_mask = '001_cutout.png'
    img = plt.imread(test_mask)

    img[:,:,3] = 1.0-img[:,:,0]
    plt.figure()
    plt.imshow(img[:,:,:])
    plt.show()

    plt.imsave(test_mask, img)