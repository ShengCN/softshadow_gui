import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from tqdm import tqdm
import os
from os.path import join

def crop_img(img):
    """ crop img by finding the bb
    """
#     print(img.shape)
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
    return ret, [x0,y0]

def add_padding(img, final_h, final_w):
    h, w, c = img.shape
    
    if final_h < h or final_w < w:
        print('padding size error')
        return img
    
    ret = np.zeros((final_h, final_w, c))
    pad_y, pad_x = (final_h - h)//2, (final_w-w)//2
    ret[pad_y:pad_y + h, pad_x:pad_x + w, :] = img
    return ret, [pad_y, pad_x]

def add_new_padding(img, final_h, final_w):
    h, w, c = img.shape
    
    if final_h < h or final_w < w:
        print('padding size error')
        return img
    
    ret = np.zeros((final_h, final_w, c))
    pad_y, pad_x = 5, (final_w-w)//2
    ret[pad_y:pad_y + h, pad_x:pad_x + w, :] = img
    return ret, [pad_y, pad_x]

def to_net_mask(ori_input):
    cropped, anchor = crop_img(ori_input)
    cropped = cropped[:,:,:3]

#     print('cropped size: ', cropped.shape)
    h,w,c = cropped.shape
    
    fix_size = 80
    if h > w:
        fix_h = fix_size
        fact = fix_h/h

        resize_crop = cv2.resize(cropped, (int(w * fact), int(h * fact)))
    else:
        fix_w = fix_size
        fact = fix_w/w

        resize_crop = cv2.resize(cropped, (int(w * fact), int(h * fact)))
        
    padded, padding_offset = add_new_padding(resize_crop, 256, 256)
    
    ret = np.zeros((256,256,4))
    ret[:,:,-1] = padded[:,:,0]
    ret[:,:,:3] = 1.0-padded
    return ret, fact, padding_offset

def composite(ori_input, shadow_img, bg_img,out_fname, norm=False):
    img = ori_input
    _, fact, padding_offset = to_net_mask(ori_input)

    new_shadow_size = int(256 /fact)
    new_shaodow_img = cv2.resize(shadow_img, (new_shadow_size, new_shadow_size))

    shadow_offset_h, shadow_offset_w = int(padding_offset[0] / fact), int(padding_offset[1] / fact)
#     show(img)
#     show(new_shaodow_img)

    # composite results
    cropped, anchor = crop_img(ori_input)
    h,w,c = cropped.shape

    big_size = max(img.shape[0], img.shape[1])

    cropped = cropped[:,:,-1:]
    cropped = np.repeat(cropped[:,:, -1:], 3, axis=2)
#     show(cropped)

    big_size = max(new_shaodow_img.shape[0], new_shaodow_img.shape[1])
    final_img = cv2.resize(bg_img, (big_size * 2, big_size * 2))

    final_anchor_size = max(shadow_offset_h,shadow_offset_w)
    final_anchor = [final_anchor_size,final_anchor_size]
    
    final_img[final_anchor[0]-shadow_offset_h:final_anchor[0]-shadow_offset_h + new_shadow_size, final_anchor[1]-shadow_offset_w:final_anchor[1]-shadow_offset_w+new_shadow_size,:3] *= new_shaodow_img[:,:,:3] 
    final_img[final_anchor[0]:final_anchor[0]+h,final_anchor[1]:final_anchor[1]+w, :3] = cropped * img[anchor[1]:anchor[1] + h, anchor[0]:anchor[0] + w,:3] + (1.0-cropped) * final_img[final_anchor[0]:final_anchor[0]+h,final_anchor[1]:final_anchor[1]+w,:3]
    
    final_cropped = final_img[final_anchor[0]-shadow_offset_h:final_anchor[0]-shadow_offset_h + new_shadow_size, final_anchor[1]-shadow_offset_w:final_anchor[1]-shadow_offset_w+new_shadow_size,:3]
    h = final_cropped.shape[0]
    offset = h //4
    final_cropped = final_img[final_anchor[0]-shadow_offset_h-offset:final_anchor[0]-shadow_offset_h -offset + new_shadow_size, final_anchor[1]-shadow_offset_w:final_anchor[1]-shadow_offset_w+new_shadow_size,:3]

    h,w,c = final_cropped.shape
    padding = int(0.2 * h)
    final_cropped = final_cropped[padding:h-padding,padding:w-padding,:]
#     final_cropped[:,:,-1] = 1.0 - final_cropped[:,:,-1]
#     alpha = final_cropped[:,:,-1]
#     alpha[np.where(alpha>1e-3)] = 1.0
    
    plt.imsave(out_fname, final_cropped)
    return final_cropped