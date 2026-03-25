import os
import time
import torch
import argparse
import itertools
import pandas as pd
import scanpy as sc
import data_utils
import train_model
import fine_tuning

from copy import deepcopy
from log import setup_logger

pd.options.mode.chained_assignment = None


def main(args, update_params_dict, logger):
    ccle_data = sc.read_h5ad('processed_data/bulk_ccle_data.h5ad')

    args.input_dim = ccle_data.shape[-1]
    args.dropout = update_params_dict['dropout']

    param_str = data_utils.dict_to_str(update_params_dict)
    args.model_save_dir = os.path.join('model_save', param_str)


    task_save_folder = os.path.join(args.model_save_dir, args.measurement, args.drug)
    data_utils.safe_make_dir(task_save_folder)
    
    # load pre-trained encoder model
    encoder = train_model.load_model(args)
    labeled_dataloader_generator = data_utils.get_labeled_dataloader_generator(args, ccle_data)

    for train_labeled_ccle, test_labeled_ccle, labeled_sc in labeled_dataloader_generator:
        ft_encoder = deepcopy(encoder)
        sc_results = fine_tuning.fine_tune_encoder(args, 
                                                   ft_encoder, 
                                                   train_labeled_ccle, 
                                                   test_labeled_ccle,
                                                   labeled_sc,
                                                   task_save_folder, 
                                                   logger,
                                                   param_str)
        
        if args.task == 'cell_line':
            print("single cell results-->", "AUROC", sc_results['auroc'], "AUPRC")
            
        elif args.task == 'patient':
            print("single cell results-->", "AUROC", sc_results['auroc'], "AUPRC", sc_results['auprc'])
        
        return sc_results

if __name__ == '__main__':
    # cell line
    # drug_list = ['Dasatinib', 'Docetaxel', 'Erlotinib', 'PLX-4720']
    # GSE_num =   ['GSE158457', 'GSE140440', 'GSE149214', 'GSE108383_451Lu']

    # patient sample
    # GSE_num =   ["GSE169246"]  # GSE169246 GSE189460
    # drug_list = ["Paclitaxel"]  # Paclitaxel, ("Bortezomib", "Melphalan", "Prednisolone")

    parser = argparse.ArgumentParser('scTAPE')

    # data selecting parameters setting
    parser.add_argument('--task', type=str, default='cell_line', choices=['cell_line', 'patient'])
    parser.add_argument('--drug', type=str, default='PLX-4720')
    parser.add_argument('--sc_data', type=str, default="GSE108383_451Lu")

    # pre-training paremeters setting
    parser.add_argument('--input_dim', type=int)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--norm_flag', type=bool, default=True)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--gap_num', dest='gap_num', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--dim_feedforward', type=int, default=1024) 
    parser.add_argument('--alpha', type=float, default=1.0)

    # fine-tuning parameters setting
    parser.add_argument('--ft_epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--metric', default='bce_loss')
    parser.add_argument('--measurement', default='Z_SCORE')
    parser.add_argument('--threshold', type=float, default=0)
    parser.add_argument('--decay_coefficient', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', default='3', type=int)
    parser.add_argument('--classifier_dims', type=list, default=[128, 64])

    # common parameters setting
    parser.add_argument('--seed', default=2002, type=int)
    parser.add_argument('--model_save_dir', type=str)
    parser.add_argument('--logs_path', type=str)

    args = parser.parse_args()
    args.device = torch.device(args.device) if args.device >= 0 else torch.device('cpu')
    
    params_grid = {
        "pretrain_num_epochs": [10, 20, 30, 40, 50],
        "adv_num_epochs": [50, 100, 150, 200, 300, 400],
        "dropout": [0.0, 0.1]
    }

    keys, values = zip(*params_grid.items())
    update_params_dict_list = [dict(zip(keys, v)) for v in itertools.product(*values)]

    args.logs_path = 'log_test/'
    start_time = time.time()

    logger = setup_logger(log_dir=args.logs_path + args.sc_data, log_filename_prefix=args.drug)
    logger.info('Starting training process')

    for param_dict in update_params_dict_list:
        data_utils.set_seed(args.seed)      
        logger.info(f"Currently model parameters are: %s", param_dict)
        sc_results = main(args, param_dict, logger)
        print(f"""AUROC: {sc_results['auroc']:.4f}, 
                  AUPRC: {sc_results['auprc']:.4f}""")

