import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import torch
import pydicom
import numpy as np
import torch_radon
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
import pandas as pd
import pydicom
import random

df_mapNeg = pd.read_csv("/users/Thalhammer/3DCNN_ArtifactReduction/mapNeg.csv")
df_mapPos = pd.read_csv("/users/Thalhammer/3DCNN_ArtifactReduction/mapPos.csv")
dataset_info = pd.read_csv("/data-pool/data_no_backup/ga63cun/PE/fanbeam_sinograms/dataset_info.csv")

def get_identifier_list(start_nr, stop_nr, df, dataset_info):
    identifier_list = []

    for k in range(start_nr, stop_nr):
        sl_nr = eval(dataset_info["array_shape"].values[k])[0]
        pat_name = dataset_info["patient_name"].values[k]
        #print(sl_nr, pat_name)
        #print(pat_name)
        sorted_instance_names = eval(df[df["pat_name"]==pat_name]["sorted_instance_names"].values[0])
        
        _identifier_list = [f"{pat_name}_{sorted_instance_names[j]}_{j}" for j in range(sl_nr)]
        identifier_list.extend(_identifier_list)

    return identifier_list

#train list
dataset_info_pos = dataset_info[dataset_info["label"]=="positive"]
identifier_list_pos_train = get_identifier_list(0, 60, df_mapPos, dataset_info_pos)

dataset_info_neg = dataset_info[dataset_info["label"]=="negative"]
identifier_list_neg_train = get_identifier_list(0, 60, df_mapNeg, dataset_info_neg)

identifier_list_train = identifier_list_pos_train + identifier_list_neg_train
df = pd.DataFrame()
df["identifier"] = identifier_list_train
df.to_csv("train.csv")

#val list
dataset_info_pos = dataset_info[dataset_info["label"]=="positive"]
identifier_list_pos_val = get_identifier_list(60, 80, df_mapPos, dataset_info_pos)

dataset_info_neg = dataset_info[dataset_info["label"]=="negative"]
identifier_list_neg_val = get_identifier_list(60, 80, df_mapNeg, dataset_info_neg)

identifier_list_val = identifier_list_pos_val + identifier_list_neg_val
df = pd.DataFrame()
df["identifier"] = identifier_list_val
df.to_csv("val.csv")

#test list
dataset_info_pos = dataset_info[dataset_info["label"]=="positive"]
identifier_list_pos_test = get_identifier_list(80, 100, df_mapPos, dataset_info_pos)

dataset_info_neg = dataset_info[dataset_info["label"]=="negative"]
identifier_list_neg_test = get_identifier_list(80, 100, df_mapNeg, dataset_info_neg)

identifier_list_test = identifier_list_pos_test + identifier_list_neg_test
df = pd.DataFrame()
df["identifier"] = identifier_list_test
df.to_csv("test.csv")
