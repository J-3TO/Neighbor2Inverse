import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2" #limit to one GPU, adjust as needed
import numpy as np
import torch
import cv2
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm
from phase_retrieval import compute_paganin_batch
from reco_utils import recon_batch, setup_device
from utilRingArtifactRemoval import sorted_filter_batch

load_path = '../ProcessedData/projPR/' #path to preprocessed projections if you wanna do phase retrieval and ring removal, adjust accordingly
save_path = '../ProcessedData/recos/'
phase_retrieval = False
ring_removal = False


def load_projections(load_path, exptime, pos, name=None):
    if name == None:
        name = f'projPR_{exptime}ms_pos{pos}.npy'
    file_path = os.path.join(load_path + name)
    proj_stitched = np.load(file_path, mmap_mode='c')
    proj_stitched = torch.from_numpy(proj_stitched)
    print(f'Loaded projections: {proj_stitched.shape}')
    return proj_stitched

def process_projections(proj_stitched, device):
    return compute_paganin_batch(
        proj_stitched,
        delta=0.8e-8, beta=1e-11, z=5000, 
        pixel_size=0.009, energy=70, 
        batch_size=2, device=device, clip=(None, None)
    )

def reconstruct(proj, device):
    with torch.no_grad():
        sin_stack = proj.swapaxes(0, 2)  # Adjust axis for reconstruction
        recon_batch_out = recon_batch(sin_stack, batch_size=3, size_rivers=7, device=device, dtype_out=torch.float16)
        torch.cuda.empty_cache()  # Free GPU memory
    return recon_batch_out

def save_reconstructions(reco, exptime, pos, save_path, name=None):
    if name == None:
        name = f'reco_{exptime}ms_pos{pos}.npy'
    save_file = os.path.join(save_path, name)
    np.save(save_file, reco.numpy().astype("float16"))
    print(f'Saved reconstruction: {save_file}')
        

def main():
    device = setup_device()
    print(device)
    os.makedirs(save_path, exist_ok=True)
    batch_size = 4

    #nr of slices is reduced
    exp_list = [200, 100, 67, 50, 33, 25, 15]  # Adjust as needed
    pos_list = [1, 2, 3, 4, 5, 7] # Adjust as needed

     # Set to True to perform phase retrieval
    
    for pos in pos_list:
        for exptime in exp_list:
            
            if ring_removal:    
                print("performing ring removal")
                proj_stitched = load_projections(load_path, exptime, pos, name=f'projStitched_{exptime}ms_pos{pos}.npy')
                sinograms = proj_stitched.swapaxes(0, 2)
                print(sinograms.shape)
                processed_sinograms = sorted_filter_batch(sinograms.squeeze(), sort_dim=1, kernel_size=127, batch_size=batch_size)
                filtered_projs = processed_sinograms.swapaxes(0, 2)

            if phase_retrieval and ring_removal:
                print("performing phase retrieval")
                proj_pr = process_projections(filtered_projs, device)
                del proj_stitched
                torch.cuda.empty_cache()

            if phase_retrieval and not ring_removal:
                print("performing phase retrieval")
                filtered_projs = load_projections(load_path, exptime, pos, name=f'projFiltered_{exptime}ms_pos{pos}.npy')
                print(proj_stitched.shape, proj_stitched.mean(), proj_stitched.std())
                proj_pr = process_projections(filtered_projs, device)
                del proj_stitched
                torch.cuda.empty_cache()

            if not phase_retrieval and not ring_removal:
                proj_pr = load_projections(load_path, exptime, pos, name=None)
            
            recon = reconstruct(proj_pr, device)
            print('recon shape', recon.shape)
            print('saving recon')
            save_reconstructions(recon, exptime, pos, save_path, name=None)
            del proj_pr, recon
            torch.cuda.empty_cache()
            
if __name__ == "__main__":
    main()
