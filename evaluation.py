import sys
sys.path.append("..")

import os 
from os.path import join
import argparse
import time
from tqdm import tqdm
from ssn.ssn import Relight_SSN
from ssn.ssn_touch import SSN_Touch
from ssn.ssn_dataset import ToTensor
import pickle
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import subprocess
import imageio
import shutil 
import glob
from tqdm import tqdm
from compsite import composite

parser = argparse.ArgumentParser(description='evaluatoin pipeline')
parser.add_argument('-f', '--file', type=str, help='input model file')
parser.add_argument('-m', '--mask', type=str, help='mask file')
parser.add_argument('-i', '--ibl', type=str, help='ibl file')
parser.add_argument('-o', '--output', type=str, help='output folder')
parser.add_argument('-w', '--weight', type=str, help='weight of current model',
                    default='weights/human_touch.pt')
# parser.add_argument('-w', '--weight', type=str, help='weight of current model',
#                     default='weights/general_tbaseline_13-November-02-23-AM.pt')
# parser.add_argument('-w', '--weight', type=str, help='weight of current model',
#                       default='weights/new_arch_touch_100%.pt')
parser.add_argument('-v', '--verbose', action='store_true', help='output file name')

options = parser.parse_args()
print('options: ', options)

# device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
model = Relight_SSN(2,1)
weight_file = options.weight
checkpoint = torch.load(weight_file, map_location=device)    
model.to(device)
model.load_state_dict(checkpoint['model_state_dict'])

ao_model = Relight_SSN(1,1)
weight_file = 'weights/human_pred_touch.pt'
cp = torch.load(weight_file, map_location=device)
ao_model.to(device)
ao_model.load_state_dict(cp['model_state_dict'])

def net_pred_touch(mask_np, ibl_np):
    mask, ibl = torch.Tensor(mask_np), torch.Tensor(ibl_np)
    with torch.no_grad():
        I_s, L_t = mask.to(device), ibl.to(device)
        predicted_img, predicted_src_light = ao_model(I_s, L_t)

    return predicted_img.detach().cpu().numpy()

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

if __name__ == '__main__':
    name = 'woman'
    mask_path = '/home/ysheng/Documents/paper_project/adobe/soft_shadow/paper_demo/supplementary/videos/mask/{}.png'.format(name)
    ao_path = '/home/ysheng/Documents/paper_project/adobe/soft_shadow/paper_demo/supplementary/videos/ao/{}.png'.format(name)

    light_folder = '/home/ysheng/Documents/paper_project/adobe/soft_shadow/paper_demo/supplementary/videos/lights'
    out_folder = '/home/ysheng/Documents/paper_project/adobe/soft_shadow/paper_demo/supplementary/videos/shadow'
    os.makedirs(out_folder, exist_ok=True)
    
    light_files = glob.glob(join(light_folder, '*'))
    for l in tqdm(light_files):
        ibl_np = plt.imread(l)[:,:,:1]
        ibl_np = cv2.resize(ibl_np, (32, 16))
        ibl_np = cv2.flip(ibl_np, 0)
        ibl_np = np.transpose(np.expand_dims(cv2.resize(ibl_np, (32, 16), cv2.INTER_LINEAR), axis=2), (2,0,1))
        if np.sum(ibl_np) > 1e-3:
            ibl_np = ibl_np * 30.0 / np.sum(ibl_np)
        ibl_np = np.repeat(ibl_np[np.newaxis,:,:,:], 1, axis=0)

        mask_img = plt.imread(mask_path)[:,:,-1:]
        mask_img = cv2.resize(mask_img, (256,256))
        mask_input = np.transpose(mask_img[:,:,np.newaxis], (2,0,1))
        mask_input = mask_input[np.newaxis, :,:,:]
        
        ao_img = plt.imread(ao_path)[:,:,:1]
        ao_img = np.transpose(ao_img, (2,0,1))
        ao_img = ao_img[np.newaxis, :,:,:]

        # ao_pred = net_pred_touch(mask_input, ibl_np)
        inputs = np.concatenate((mask_input, ao_img), axis=1)
        shadow_pred = net_render_np(inputs, ibl_np)[0]
        shadow_pred = np.transpose(shadow_pred, (1,2,0))
        shadow_pred = np.squeeze(shadow_pred)
        shadow_pred = np.ascontiguousarray(shadow_pred, dtype=np.float32)
        cv2.normalize(shadow_pred, shadow_pred, 0.0,1.0, cv2.NORM_MINMAX)
        
        shadow_pred = np.repeat(shadow_pred[:,:,np.newaxis], 3, axis=2)

        out_fname = join(out_folder, os.path.basename(l))
        plt.imsave(out_fname, shadow_pred)
            
    
    ofolder = '/home/ysheng/Documents/paper_project/adobe/soft_shadow/paper_demo/supplementary/videos'
    shadows = glob.glob(join(ofolder, 'shadow/*'))
    ori_input = plt.imread(join(ofolder,'segmentation/{}.png'.format(name)))
    # bg_img = plt.imread(join(ofolder, 'bg.png'))
    bg_img = np.ones((512,512,4))

    for s in tqdm(shadows):
        shadow_img = 1.0 - plt.imread(s)[:,:,:3]
        outfname = join(ofolder, 'comp', '{}_'.format(name) + os.path.basename(s))
        composite(ori_input, shadow_img, bg_img, outfname, False)