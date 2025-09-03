import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2" #limit to one GPU, adjust as needed
import matplotlib
import logging
import traceback
import datetime
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [7, 7]
import numpy as np
import torch
import cv2
from tqdm import tqdm
print(torch.__version__)
print(torch.cuda.get_device_name(0))
from torch_radon import Radon, RadonFanbeam
from phase_retrieval import *
from reco_utils import *
import h5py
from stitching import *
import pandas as pd

source_path = '../Measurements/' #path to raw data
save_path = '../ProcessedData/projStitched/' #path to save preprocessed projections

device = setup_device()
torch.set_float32_matmul_precision('medium')
dfOverlap_path = save_path + '/overlap.csv'


if not os.path.exists(save_path):
    os.makedirs(save_path)

def load_flat(path):
    ar = np.array(h5py.File(path)['entry']['data']['data']).mean(axis=0)[10:] #cut first 10 pixels which are faulty
    return torch.from_numpy(ar[None, None])

# Set up logging
log_file = os.path.join(os.path.dirname(__file__), 'processing_errors.log')
logging.basicConfig(
    filename=log_file,
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s' 
)
exp_list = [15, 25, 33, 50, 67, 100, 200]  # Adjust as needed
pos_list = [1, 2, 3, 4, 5, 7] # Adjust as needed

try:
    for pos in pos_list:     
        for exptime in exp_list: 
            try:
                print(f"start processing pos {pos}, exposure time {exptime}")
                path = source_path  + f'/Calf_31_pos{pos}_{exptime}ms/'
                #load data
                print("loading data")
                f1 = h5py.File(path + '/SAMPLE.hdf')
                proj_stack = np.array(f1['entry']['data']['data'])[:, 10:] #cut first 10 pixels which are faulty
                
                print("loading flats and darks")
                #load flats and darks
                img_dark_before = load_flat(path + '/DF_BEFORE.hdf')
                try:
                    img_dark_after = load_flat(path + '/DF_AFTER.hdf')
                except:
                    img_dark_after = img_dark_before
                img_flat_before = load_flat(path + '/BG_BEFORE.hdf')
                
                try:
                    img_flat_after = load_flat(path + '/BG_AFTER.hdf')
                except:
                    img_flat_after = img_flat_before
                
                #do flatffielding
                images = torch.from_numpy(proj_stack[:, None])
                with torch.no_grad():
                    proj_stack_corrected = flatfield_correction(images.float(), img_dark_before, img_dark_after, img_flat_before, img_flat_after,batch_size=4, device=device)
                    torch.cuda.empty_cache()
                
                #save images
                #do flip and stitch to double the field-of-view
                proj_stack_clip = proj_stack_corrected[20:3620] #measured too many projections (364 angles) so we have to throw out some
                proj_left, proj_right = proj_stack_clip[:1800],  proj_stack_clip[1800:] 
                proj_right = torch.flip(proj_right, [-1])
                assert proj_right.shape==proj_left.shape, f"right shape: {proj_right.shape}, left shape:{proj_left.shape}"
                
                if exptime == 200:
                    #find out best overlap for stitching once per position
                    
                    if os.path.isfile(dfOverlap_path):
                        df_overlap = pd.read_csv(dfOverlap_path)
                    else:
                        df_overlap = pd.DataFrame()
                        df_overlap['label'] = ['overlap', 'mean', 'min', 'max', 'std']
                        
                    overlap_list = []
                    print("finding best overlap")
                    for nr in tqdm(range(0, 1800, 10)):
                        img_left, img_right = proj_left[nr:nr+1], proj_right[nr:nr+1]
                        best_overlap = find_best_overlap(img_left, img_right, min_overlap=400, max_overlap=500)
                        overlap_list.append(best_overlap)
                    
                    overlap_list = np.array(overlap_list)
                    overlap = int(overlap_list.mean())
                    overlap_stats = [overlap, overlap_list.mean(), overlap_list.min(), overlap_list.max(), overlap_list.std()]
                    df_overlap[f'overlap pos: {pos}'] = overlap_stats
                    df_overlap.to_csv(dfOverlap_path)
                    
                else:    
                    df_overlap = pd.read_csv(dfOverlap_path)
                    overlap = int(df_overlap[f'overlap pos: {pos}'][0])
                    
                print("start stitching with overlap ", overlap)
                
                proj_stitched = stitch_images(proj_left, proj_right, overlap=overlap)
                assert not math.isnan(proj_stitched.min()), 'found NaN'
                assert not proj_stitched.min() <= 0, f'found number 0 or below: {proj_stitched.min()}'
                print("start despeckling")
                proj_stitched = despeckle_batched_optimized(proj_stitched, batch_size=16, th=0.2, device=device)
                #save
                np.save(save_path + f'projStitched_{exptime}ms_pos{pos}.npy', np.array(proj_stitched).astype("float16"))
                
            except KeyboardInterrupt:
                print("\nKeyboard interrupt detected. Exiting gracefully...")
                # Optional: log that the process was manually interrupted
                logging.info(f"Process interrupted by user at pos {pos}, exposure time {exptime}")
                sys.exit(0)
            except Exception as e:
                # Log the error with position and exposure time information
                error_message = f"Error processing pos {pos}, exposure time {exptime}: {str(e)}"
                error_details = traceback.format_exc()
                
                # Write to log file
                logging.error(error_message)
                logging.error(error_details)
                
                # Also print to console
                print(f"ERROR: {error_message}")
                print(f"Continuing to next iteration...")
                
                # Continue to the next iteration
                continue
except KeyboardInterrupt:
    print("\nKeyboard interrupt detected. Exiting gracefully...")
    # Clean up resources if needed
    try:
        torch.cuda.empty_cache()  # Free GPU memory
    except:
        pass
    sys.exit(0)
