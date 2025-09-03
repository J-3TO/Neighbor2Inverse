import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2" #limit to one GPU, adjust as needed
import sys
import numpy as np
import torch
import cv2
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm
from phase_retrieval import compute_paganin_batch
from reco_utils import recon_batch, setup_device
from utilRingArtifactRemoval import * 
from tqdm import tqdm
import gc

load_path = '../ProcessedData/projStitched/' #path to preprocessed projections
save_path = '../ProcessedData/projFiltered/' #path to save filtered projections
batch_size = 4

def load_projections(load_path, exptime, pos):
    file_path = os.path.join(load_path + f'projStitched_{exptime}ms_pos{pos}.npy')
    proj_stitched = np.load(file_path, mmap_mode='c')
    proj_stitched = torch.from_numpy(proj_stitched).to(torch.float32)
    print(f'Loaded projections: {proj_stitched.shape}')
    return proj_stitched

def save_projection(projs, exptime, pos, save_path, name=None):
    if name==None:
        save_file = os.path.join(save_path, f'pos{pos}Exptime{exptime}msProjPR.npy')
    else:
        save_file = os.path.join(save_path, name)
    np.save(save_file, projs.numpy().astype("float16"))
    print(f'Saved projection: {save_file}')

def main():
    print('set device')
    device = setup_device()
    print(device)
    os.makedirs(save_path, exist_ok=True)
    
    exp_list = [15, 25, 33, 50, 67, 100, 200]  # Adjust as needed
    pos_list = [1, 2, 3, 4, 5, 7] # Adjust as needed

    exp_list = [67, 200]  # Testing
    pos_list = [2] # Testing


    for pos in pos_list:
        for exptime in exp_list:
            proj_stitched = load_projections(load_path, exptime, pos)
            print(proj_stitched.shape)
            sinograms = proj_stitched.swapaxes(0, 2)
            print(sinograms.shape)
            
            # 3. Apply the pipeline function
            processed_sinograms = sorted_filter_batch(sinograms.squeeze(), sort_dim=1, kernel_size=127, batch_size=batch_size)
            
            filtered_projs = processed_sinograms.swapaxes(0, 2)
            print('filtered projs', filtered_projs.shape)

            name = f'projFiltered_{exptime}ms_pos{pos}.npy'

            save_projection(filtered_projs, exptime, pos, save_path, name=name)
            del proj_stitched, sinograms, processed_sinograms, filtered_projs
            torch.cuda.empty_cache()
            gc.collect()

if __name__ == "__main__":
    main()
