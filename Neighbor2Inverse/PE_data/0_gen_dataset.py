import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import torch
import pydicom
import numpy as np
import torch_radon
from tqdm import tqdm
import pandas as pd
import random
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

# Configuration
IMAGE_SIZE = 512
N_ANGLES = 2048
DATA_BASE_PATH = "/vault3/machine_learning/datasets/public/RSNA_PE/rsna-str-pulmonary-embolism-detection/train"
OUTPUT_BASE_PATH = "/data-pool/data_no_backup/ga63cun/PE/fanbeam_sinograms"
NUM_SAMPLES = 100
NUM_WORKERS = 4  # Number of threads for I/O operations
RANDOM_SEED = 42  # Fixed seed for reproducibility

# Set random seeds for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

# Initialize radon transform globally (once for GPU)
angles = torch.linspace(0, 2 * np.pi, N_ANGLES)
radon = torch_radon.RadonFanbeam(
    IMAGE_SIZE,
    angles,
    source_distance=2000,
    det_spacing=1,
    clip_to_circle=False,
    det_count=int(2 * IMAGE_SIZE)
)


def load_dicom_volume(inst_names, pat_name, serie_name, data_base_path):
    """Load DICOM files and create volume (CPU-only, can be parallelized)"""
    # Read first DICOM to get RescaleIntercept
    first_dicom = pydicom.dcmread(f"{data_base_path}/{pat_name}/{serie_name}/{inst_names[0]}")
    rescale_intercept = first_dicom.RescaleIntercept
    
    volume = np.stack([
        pydicom.dcmread(
            f"{data_base_path}/{pat_name}/{serie_name}/{inst_name}"
        ).pixel_array
        for inst_name in inst_names
    ], axis=0) 

    # Apply intercept (either -1024 or 0) to make sure scaling is correct
    volume = (volume + rescale_intercept + 1024) / 4095.
    return volume


def process_patient(row_data, data_base_path, output_base_path, label):
    """
    Process a single patient: load DICOM files, create sinogram, and save.
    
    Args:
        row_data: tuple of (inst_names, pat_name, serie_name)
        data_base_path: base path for DICOM files
        output_base_path: base path for output sinograms
        label: 'positive' or 'negative' label for the patient
    
    Returns:
        dict: patient info including name, array size, label, success status, and error
    """
    inst_names, pat_name, serie_name = row_data
    
    try:
        # Load volume (this part can be threaded for I/O)
        volume = load_dicom_volume(inst_names, pat_name, serie_name, data_base_path)
        
        # Convert to tensor and compute sinogram on GPU (must be sequential)
        volume_batch = torch.from_numpy(volume[:, None]).float()
        
        with torch.no_grad():
            sinogram = radon.forward(volume_batch.cuda()).cpu().numpy().astype("float32")
        
        # Save result
        output_path = Path(output_base_path) / f"{pat_name}.npy"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, sinogram)
        
        # Get array size
        array_size = sinogram.shape
        
        # Clean up
        del volume_batch, sinogram
        torch.cuda.empty_cache()
        
        return {
            'patient_name': pat_name,
            'array_shape': str(array_size),
            'label': label,
            'success': True,
            'error': None
        }
    
    except Exception as e:
        return {
            'patient_name': pat_name,
            'array_shape': None,
            'label': label,
            'success': False,
            'error': str(e)
        }


def process_dataframe(df, num_samples, data_base_path, output_base_path, label, desc="Processing"):
    """
    Process a dataframe of patients.
    Uses GPU sequentially to avoid CUDA context issues.
    
    Args:
        df: DataFrame with patient information
        num_samples: number of samples to process
        data_base_path: base path for DICOM files
        output_base_path: base path for output sinograms
        label: 'positive' or 'negative' label for the patients
        desc: description for progress bar
    
    Returns:
        list: list of dictionaries containing patient information
    """
    # Prepare data
    indices = list(range(len(df)))
    random.shuffle(indices)
    indices = indices[:num_samples]
    
    # Extract row data
    rows_data = [
        (eval(df.iloc[i]["sorted_instance_names"]),
         df.iloc[i]["pat_name"],
         df.iloc[i]["serie_name"])
        for i in indices
    ]
    
    # Create partial function with fixed arguments
    process_func = partial(
        process_patient,
        data_base_path=data_base_path,
        output_base_path=output_base_path,
        label=label
    )
    
    # Process sequentially (GPU operations can't be parallelized safely)
    results = []
    for row_data in tqdm(rows_data, desc=desc):
        result = process_func(row_data)
        results.append(result)
    
    # Report results
    success_count = sum(1 for r in results if r['success'])
    error_count = sum(1 for r in results if not r['success'])
    
    for result in results:
        if not result['success']:
            print(f"Error processing {result['patient_name']}: {result['error']}")
    
    print(f"\n{desc} - Success: {success_count}, Errors: {error_count}")
    return results


if __name__ == "__main__":
    # Ensure output directory exists
    Path(OUTPUT_BASE_PATH).mkdir(parents=True, exist_ok=True)
    
    # List to collect all patient information
    all_patient_info = []
    
    print("Processing positive samples...")
    df_mapPos = pd.read_csv("./mapPos.csv")
    pos_results = process_dataframe(
        df_mapPos,
        NUM_SAMPLES,
        DATA_BASE_PATH,
        OUTPUT_BASE_PATH,
        label="positive",
        desc="Positive samples"
    )
    all_patient_info.extend(pos_results)
    torch.cuda.empty_cache()
    
    print("\nProcessing negative samples...")
    df_mapNeg = pd.read_csv("./mapNeg.csv")
    neg_results = process_dataframe(
        df_mapNeg,
        NUM_SAMPLES,
        DATA_BASE_PATH,
        OUTPUT_BASE_PATH,
        label="negative",
        desc="Negative samples"
    )
    all_patient_info.extend(neg_results)
    
    # Create and save DataFrame with patient information
    df_output = pd.DataFrame(all_patient_info)
    df_output = df_output[['patient_name', 'array_shape', 'label', 'success', 'error']]
    
    output_csv_path = Path(OUTPUT_BASE_PATH) / "dataset_info.csv"
    df_output.to_csv(output_csv_path, index=False)
    print(f"\nDataset information saved to: {output_csv_path}")
    print(f"Total patients processed: {len(df_output)}")
    print(f"Successfully processed: {df_output['success'].sum()}")
    print(f"Failed: {(~df_output['success']).sum()}")