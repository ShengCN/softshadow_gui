import matplotlib.pyplot as plt
import cv2
import os
from os.path import join
import numpy as np

def crop_img(img):
    """ crop img by finding the bb
    """
    print(img.shape)
    if img.shape[2] == 3:    
        mask  = (img[:,:,0] + img[:,:,1] + img[:,:,2])/3.0
    else:
        mask = img[:,:,3]
    
    coord = np.where(mask!=0)
    x0, y0, x1, y1 = np.min(coord[1]), np.min(coord[0]), np.max(coord[1]), np.max(coord[0])
    
    h,w = img.shape[0], img.shape[1]
    ret = np.zeros((y1-y0,x1-x0,4))
    ret[:,:,:3] = img[y0:y1, x0:x1, :3]
    ret[:,:,3] = mask[y0:y1, x0:x1]
    return ret

def add_padding(img, final_h, final_w):
    h, w, c = img.shape
    
    if final_h < h or final_w < w:
        print('padding size error')
        return img
    
    ret = np.zeros((final_h, final_w, c))
    pad_y, pad_x = (final_h - h)//2, (final_w-w)//2
    ret[pad_y:pad_y + h, pad_x:pad_x + w, :] = img
    return ret

def old_to_net(img, center_size=85):
    cropped = crop_img(img)
    
    h, w = cropped.shape[0], cropped.shape[1]
    if h > w:
        new_w = int(center_size / h * w)
        resized = cv2.resize(cropped, (new_w, center_size), cv2.INTER_AREA)
    else:
        new_h = int(center_size / w * h)
        resized = cv2.resize(cropped, (center_size, new_h), cv2.INTER_AREA)
    return add_padding(resized, 256, 256)