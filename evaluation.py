import sys
sys.path.append("..")

import os 
from os.path import join
import argparse
import time
from tqdm import tqdm
from ssn.ssn import Relight_SSN
from ssn.ssn_dataset import ToTensor
import pickle
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import subprocess
import imageio
import shutil 
from net_mask_crop import old_to_net

parser = argparse.ArgumentParser(description='evaluatoin pipeline')
parser.add_argument('-f', '--file', type=str, help='input model file')
parser.add_argument('-m', '--mask', type=str, help='mask file')
parser.add_argument('-i', '--ibl', type=str, help='ibl file')
parser.add_argument('-o', '--output', type=str, help='output folder')
parser.add_argument('-w', '--weight', type=str, help='weight of current model',
                    default='weights/baseline_human_20-May-11-37-AM.pt')
# parser.add_argument('-w', '--weight', type=str, help='weight of current model',
#                     default='weights/human_baseline_21-September-04-14-PM.pt')
parser.add_argument('-v', '--verbose', action='store_true', help='output file name')

options = parser.parse_args()
print('options: ', options)

# device = torch.device("cpu")
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
model = Relight_SSN(1,1)
weight_file = options.weight
checkpoint = torch.load(weight_file, map_location=device)    
model.to(device)
model.load_state_dict(checkpoint['model_state_dict'])

def get_files(folder):
    return [join(folder,f) for f in os.listdir(folder) if os.path.isfile(join(folder, f))]

def read_img(img_path):
    img = plt.imread(img_path)
    if img.dtype == np.uint8:
        img = img/255.0
    return img

def net_render_np(mask_np, ibl_np):
    """
    input:
        mask_np shape: b x c x h x w
        ibl_np shape: 1 x 16 x 32
    output:
        shadow_predict shape: b x c x h x w
    """
    s = time.time()
    if mask_np.dtype == np.uint8:
        mask_np = mask_np/255.0

    mask, ibl = torch.Tensor(mask_np), torch.Tensor(ibl_np)
    with torch.no_grad():
        I_s, L_t = mask.to(device), ibl.to(device)
        # print('I_s: {}, L_t: {}'.format(I_s.shape, L_t.shape))
        predicted_img, predicted_src_light = model(I_s, L_t)

    print('net predict finished, time: {}s'.format(time.time() -s))
    
    return predicted_img.detach().cpu().numpy()

def rotate_ibl(light_np, offset, is_x=True):
    tmp = light_np.copy()
    if is_x: 
        light_np[:,:offset, :] = tmp[:,-offset:,:] 
        light_np[:,offset:,:] = tmp[:,:-offset,:]
    else:
        light_np[:offset,:, :] = tmp[-offset:,:,:] 
        light_np[offset:,:,:] = tmp[:-offset,:,:]
    return light_np

def batch_predict(light_file, mask_folder, out_folder):
    light_np = read_img(light_file)
    light_np = rotate_ibl(light_np, -6, True)
    light_np = rotate_ibl(light_np, -2, False)
    light_np = cv2.resize(light_np, (32,16))
    light_np = light_np[:8,:,0:1]
    light_np = cv2.flip(light_np, 0)
    light_np = cv2.resize(light_np, (32, 16))
    light_np = light_np[np.newaxis, :, :]
    ibl_np = light_np[np.newaxis,:,:,:]
    
    files = get_files(mask_folder)
    for i, f in enumerate(tqdm(files)):
        mask_np = read_img(f)
        mask_np = old_to_net(mask_np)
        mask_np = mask_np[:,:,-1]
        mask_np = cv2.resize(mask_np, (256,256))
        mask_np = mask_np[:,:,np.newaxis]
        mask_np = np.transpose(mask_np, (2,0, 1))
        mask_np = np.clip(mask_np, 0.0,1.0)
        mask_np = mask_np[np.newaxis, :,:,:]
        # import pdb; pdb.set_trace()

        batch_predicted = np.transpose(net_render_np(mask_np, ibl_np)[0], (1,2,0))

        batch_predicted = np.repeat(batch_predicted, 3, axis=2)
        cv2.normalize(batch_predicted, batch_predicted,0.0, 1.0, cv2.NORM_MINMAX) 

        basename = os.path.basename(f)
        out_fname = join(out_folder, basename)
        plt.imsave(out_fname, batch_predicted) 

if __name__ == '__main__':
    test_light = '/home/ysheng/Documents/paper_project/adobe/OGL_EXP/light_right.png'
    root = '/home/ysheng/Documents/paper_project/adobe/pifu_exp'
    folder = join(root, 'original_mask')
    out_folder = join(root, 'net_out')
    os.makedirs(out_folder, exist_ok=True)

    batch_predict(test_light, folder, out_folder) 
    
    plt.show()
