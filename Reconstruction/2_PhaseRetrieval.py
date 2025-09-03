import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import sys
import numpy as np
import torch
import cv2
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm
from phase_retrieval import compute_paganin_batch
from reco_utils import recon_batch, setup_device

load_path = '../ProcessedData/projFiltered/' #path to preprocessed projections
save_path = '../ProcessedData/projPR/' #path to save filtered projections

PR_parameters = {"delta":0.8e-8, 
                 "beta":1e-11, 
                 "z":5000, #in mm 
                 "pixel_size":0.009, #in mm
                 "energy":70, #in keV
                 "batch_size":2}

def load_projections(load_path, exptime, pos):
    file_path = os.path.join(load_path + f'projFiltered_{exptime}ms_pos{pos}.npy')
    proj_stitched = np.load(file_path, mmap_mode='c')
    proj_stitched = torch.from_numpy(proj_stitched).to(torch.float32)
    print(f'Loaded projections: {proj_stitched.shape}')
    return proj_stitched

def process_projections(proj_stitched, device, PR_parameters):
    return compute_paganin_batch(
        proj_stitched,
        device=device,
        **PR_parameters)
        
def save_projection(projs, exptime, pos, save_path):
    save_file = os.path.join(save_path, f'projPR_{exptime}ms_pos{pos}.npy')
    np.save(save_file, projs.numpy().astype("float16"))
    print(f'Saved reconstruction: {save_file}')

def main():
    print('set device')
    device = setup_device()
    print(device)
    os.makedirs(save_path, exist_ok=True)
    
    exp_list = [25, 33, 50, 67, 100, 200]  # Adjust as needed
    pos_list = [1, 2, 3, 4, 5, 7] # Adjust as needed

    exp_list = [67, 200]  # Adjust as needed
    pos_list = [2] # Adjust as needed

    for pos in pos_list:
        for exptime in exp_list:
            proj_stitched = load_projections(load_path, exptime, pos)
            print(proj_stitched.shape)
            proj_pr = process_projections(proj_stitched, device, PR_parameters)
            save_projection(proj_pr, exptime, pos, save_path)
            del proj_stitched            
            del proj_pr
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
