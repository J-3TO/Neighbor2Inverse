import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
import sys
sys.path.append("../")
import lightning as pl
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset import ProjDatasetSlice, ProjDataset, ClinicalDataset
from modelLightning import *
import yaml
import sys
from copy import deepcopy
import argparse
from network import UNet
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from utils_callback import SavePredictionCallback, SavePredictionCallbackSlice, SaveHyperparametersCallback, SavePredictionPE

torch.set_float32_matmul_precision('medium')

def main(trainparams):
    pl.seed_everything(42)

    print("Let's go!", chr(sum(range(ord(min(str(not())))))))
    print(trainparams)
    
    # ----- initialize dataset -----
    #initialize the datasets & dataloader
    dataset_train = ClinicalDataset(**trainparams['dataset'], **trainparams['dataset_train'])
    dataset_val = ClinicalDataset(**trainparams['dataset'], **trainparams['dataset_val'])

    # initialize the dataloaders
    dataloader_train = DataLoader(dataset_train, **trainparams['train_loader'])
    dataloader_val = DataLoader(dataset_val, **trainparams['val_loader'])
    base_network = UNet(**trainparams['base_network']['params'])
        
    # ----- init model -----
    base_network = UNet(**trainparams['base_network']['params'])
        
    # ----- init model -----
    litmodel = Neighbor2InverseClinical(network=base_network, 
                    **trainparams["lightning_params"],
                    optimizer_algo =  trainparams["optimizer_algo"],
                    scheduler_algo =  trainparams["scheduler_algo"],
                    optimizer_params = trainparams["optimizer_params"],
                    scheduler_params = trainparams["scheduler_params"],
                    )
    
    # -----  define callbacks/loggers -----
    save_path = os.path.abspath(trainparams['save_path'] + f"/{trainparams['name']}/")

    lr_monitor = pl.pytorch.callbacks.LearningRateMonitor(logging_interval='epoch') 
    tblogger = pl.pytorch.loggers.TensorBoardLogger(save_path) 
    csvlogger = pl.pytorch.loggers.CSVLogger(save_path, version=tblogger.version) 
    checkpoint = pl.pytorch.callbacks.ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=3, filename='{epoch}_valL{val_loss:.4f}', save_last=True)

    if trainparams["earlyStopping"] == True:
        early_stopping = EarlyStopping(**trainparams["earlyStopping_params"]) #stops trainig process if performance does not improve anymore
    else:
        early_stopping = pl.pytorch.callbacks.EarlyStopping(monitor="val_loss", patience=10000) #dummy callback, will never trigger

    #init image logger
    prediction_callback = SavePredictionPE(input_file="/users/Thalhammer/australienSynchrotron/GITHUB/Neighbor2Inverse/Neighbor2Inverse/PE_data/testReco.npy", output_dir=save_path + f"/lightning_logs/version_{tblogger.version}/predictions/") #saves an image of the current model prediction of the val set

    savetrainparams_callback  = SaveHyperparametersCallback(output_dir=save_path + f"/lightning_logs/version_{tblogger.version}/", file=trainparams) #saves a copy of the trainparams.yaml file in the same folder as the checkpoints
    
    # -----  init trainer and start training -----
    trainer = pl.Trainer(logger=[csvlogger, tblogger], 
                        callbacks=[lr_monitor, checkpoint, savetrainparams_callback, early_stopping, prediction_callback],
                        max_epochs=trainparams['lightning_params']['n_epoch'],
                        accelerator='gpu',
                        devices=trainparams['gpus'], 
                        **trainparams['trainer_params'],
                        #overfit_batches=3, #for debugging
                        )
    
    trainer.fit(litmodel, dataloader_train, dataloader_val)

if __name__ == '__main__':
    #option to provide some parameters via command line, overriding the values in the trainparams yaml file
    parser = argparse.ArgumentParser()
    # Add all your arguments...
    parser.add_argument("--trainparams", type=str, default="./trainparamsNeighbor2InverseProjSub.yml")
    parser.add_argument("--gpu", type=int, default=None)
    args = parser.parse_args()
    with open(args.trainparams, 'r') as f:
        trainparams = yaml.safe_load(f)

    # Override parameters if provided
    if args.gpu is not None:
        trainparams['gpus'] = [int(args.gpu)]
        print('trainparams gpu:', trainparams['gpus'])
        
    main(trainparams)