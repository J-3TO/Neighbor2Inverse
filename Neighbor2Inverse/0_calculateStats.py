import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "50"
import pandas as pd
import sys
import numpy as np
import torch
from tqdm import tqdm


def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("medium")
    return device

def main():
    stats_path = './reco_stats.csv'
    
    # Check if stats file already exists
    if os.path.exists(stats_path):
        print(f"Loading existing stats from {stats_path}")
        df_stats = pd.read_csv(stats_path, index_col=0)
    else:
        print("Generating new stats file...")
        df_stats = pd.DataFrame()
    device = setup_device()
    print(device)
    load_path = '../ProcessedData/recos/'
    
    #nr of slices is reduced
    exp_list = [15, 200]  # Adjust as needed
    pos_list = [1, 2, 3, 4, 5, 7] # Adjust as needed

    exp_list = [15, 200]  # Adjust as needed #ToDo
    pos_list = [3] # Adjust as needed #ToDO
    
    for pos in pos_list:
        for exptime in exp_list:
            df_ = pd.DataFrame()

            print('loading reco')
            reco = np.load(load_path + f'/reco_{exptime}ms_pos{pos}.npy', mmap_mode='c').astype('float32')
            
            print('reco shape', reco.shape)
            mean, std, mn, mx = float(reco.mean()), float(reco.std()), float(reco.min()), float(reco.max())
            n_slices, _, shape_y, shape_x = reco.shape
        
            df_['filename'] = [f'reco_{exptime}ms_pos{pos}']
            df_['mean'] = [mean]
            df_['std'] = [std]
            df_['min'] = [mn]
            df_['max'] = [mx]
            df_['n_slices'] = [n_slices]
            df_['shape_x'] = [shape_x]
            df_['shape_y'] = [shape_y]
            
            df_stats = pd.concat((df_stats, df_), ignore_index=True)
            del reco
            torch.cuda.empty_cache()
    
    # Save the stats
    df_stats.to_csv(stats_path)
    print(f"Saved new stats to {stats_path}")

if __name__ == "__main__":
    main()
