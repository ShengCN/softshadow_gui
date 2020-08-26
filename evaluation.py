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

parser = argparse.ArgumentParser(description='evaluatoin pipeline')
parser.add_argument('-f', '--file', type=str, help='input model file')
parser.add_argument('-m', '--mask', type=str, help='mask file')
parser.add_argument('-i', '--ibl', type=str, help='ibl file')
parser.add_argument('-o', '--output', type=str, help='output folder')
parser.add_argument('-w', '--weight', type=str, help='weight of current model',
                    default='weights/group_norm_15-May-07-45-PM.pt')
parser.add_argument('-v', '--verbose', action='store_true', help='output file name')

options = parser.parse_args()
print('options: ', options)

# device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
model = Relight_SSN(1,1)
weight_file = options.weight
checkpoint = torch.load(weight_file, map_location=device)    
model.to(device)
model.load_state_dict(checkpoint['model_state_dict'])

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

