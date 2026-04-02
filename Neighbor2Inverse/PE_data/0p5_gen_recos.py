import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
import sys
sys.path.append("../")
import lightning as pl
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset import ClinicalDataset
from modelLightning import *
import yaml
import sys
from copy import deepcopy
import argparse
from network import UNet
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from utils_callback import SavePredictionCallback, SavePredictionCallbackSlice, SaveHyperparametersCallback
from eval_utils import plot_images_with_zoom
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from tqdm import tqdm
torch.set_float32_matmul_precision('medium')

with open('trainparamsClinicalSupervised.yml', 'r') as f:
    trainparams = yaml.safe_load(f)

trainparams["dataset_val"]["skip"] = 1
base_network = UNet(**trainparams['base_network']['params'])
    
# ----- init model -----
litmodel = LitmodelSupervised(network=base_network, 
                 **trainparams["lightning_params"],
                 optimizer_algo =  trainparams["optimizer_algo"],
                 scheduler_algo =  trainparams["scheduler_algo"],
                 optimizer_params = trainparams["optimizer_params"],
                 scheduler_params = trainparams["scheduler_params"],
                   )
litmodel = litmodel.cuda()

pat_list_noisy = []
pat_list_clean = []

save_path_target = "/data-pool/data_no_backup/ga63cun/PE/fanbeam_reconstructed_target/"
save_path_noisy = "/data-pool/data_no_backup/ga63cun/PE/fanbeam_reconstructed_noisy/"


dataset_train = ClinicalDataset(**trainparams['dataset'], **trainparams['dataset_train'])
dataset_val = ClinicalDataset(**trainparams['dataset'], **trainparams['dataset_val'])
dataset_test = ClinicalDataset(**trainparams['dataset'], **trainparams['dataset_test'])

def gen_dataset(dataset, batch_size=16):
    prev_pat_name = None
    batch_buffer = []
    
    def process_batch(batch_data):
        """Process a batch of slices together for better GPU utilization"""
        proj_cleans, proj_noisys, angles_list = [], [], []
        
        for proj_stack, proj_stack_noisy, _, _, _ in batch_data:
            proj_clean = torch.from_numpy(proj_stack).unsqueeze(0)
            proj_noisy = torch.from_numpy(proj_stack_noisy).unsqueeze(0)
            proj_cleans.append(proj_clean)
            proj_noisys.append(proj_noisy)
        
        # Stack all slices in batch
        proj_cleans = torch.cat(proj_cleans, dim=0)
        proj_noisys = torch.cat(proj_noisys, dim=0)
        n_angles = int(proj_noisys.shape[1])
        angles = torch.linspace(0, 2*np.pi, n_angles, device='cuda')
        
        # Combine clean and noisy
        proj_combined = torch.cat((proj_cleans, proj_noisys), dim=0)
        sin_stack = proj_combined.swapaxes(1, 2)
        
        # Single GPU call for entire batch
        with torch.no_grad():
            reco = litmodel.reconstruct(sin_stack.cuda(), angles=angles, image_size=512, source_distance=2000).detach().cpu()
        
        batch_len = len(batch_data)
        reco_clean = reco[:batch_len].numpy()
        reco_noisy = reco[batch_len:].numpy()
        
        return reco_clean, reco_noisy
    
    def flush_batch():
        """Process accumulated batch"""
        if not batch_buffer:
            return
        
        reco_cleans, reco_noisys = process_batch(batch_buffer)
        
        for i in range(len(reco_cleans)):
            pat_list_clean.append(reco_cleans[i])
            pat_list_noisy.append(reco_noisys[i])
        
        batch_buffer.clear()
        torch.cuda.empty_cache()
    
    for idx in tqdm(range(dataset.__len__())):
        proj_stack, proj_stack_noisy, pat_name, filename, slice_index = dataset.__getitem__(idx)
        
        # Save and clear when we encounter a new patient
        if prev_pat_name is not None and pat_name != prev_pat_name:
            # Flush any remaining batch before saving
            flush_batch()
            
            pat_clean = np.stack(pat_list_clean).astype("float16")[:, 0]
            pat_noisy = np.stack(pat_list_noisy).astype("float16")[:, 0]
            np.save(f"{save_path_target}/{prev_pat_name}.npy", pat_clean)
            np.save(f"{save_path_noisy}/{prev_pat_name}.npy", pat_noisy)
            pat_list_noisy.clear()
            pat_list_clean.clear()
        
        prev_pat_name = pat_name
        
        # Add to batch buffer
        batch_buffer.append((proj_stack, proj_stack_noisy, pat_name, filename, slice_index))
        
        # Process when batch is full
        if len(batch_buffer) >= batch_size:
            flush_batch()
    
    # Flush final batch and save last patient
    flush_batch()
    if prev_pat_name is not None:
        pat_clean = np.stack(pat_list_clean).astype("float16")[:, 0]
        pat_noisy = np.stack(pat_list_noisy).astype("float16")[:, 0]
        np.save(f"{save_path_target}/{prev_pat_name}.npy", pat_clean)
        np.save(f"{save_path_noisy}/{prev_pat_name}.npy", pat_noisy)
        pat_list_noisy.clear()
        pat_list_clean.clear()

gen_dataset(dataset_train)
gen_dataset(dataset_val)
gen_dataset(dataset_test)