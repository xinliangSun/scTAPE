import os
import time
import torch
import argparse
import itertools
import data_utils
import train_model
import scanpy as sc

from log import setup_logger


def main(para, update_params_dict, logger):
    para.pretrain_num_epochs = update_params_dict['pretrain_num_epochs']
    para.adv_num_epochs = update_params_dict['adv_num_epochs']
    para.dropout = update_params_dict['dropout']
    
    param_str = data_utils.dict_to_str(update_params_dict)
    # para.model_save_dir = os.path.join('ablation_model/train_batch_' + str(args.batch_size) + '_5K', param_str)
    para.model_save_dir = os.path.join('ablation_model/no_adv', param_str)
    data_utils.safe_make_dir(para.model_save_dir)

    sc_data = sc.read_h5ad('processed_data/sc_ccle_data_5K.h5ad')
    ccle_data = sc.read_h5ad('processed_data/bulk_ccle_data_5K.h5ad')
    para.input_dim = ccle_data.shape[-1]

    logger.info(f"Currently pre-training model parameters are: %s", para)

    s_dataloaders, t_dataloaders = data_utils.get_unlabeled_dataloaders(para, ccle_data, sc_data)
    
    # start unlabeled training
    pretrain_model = train_model.train
    pretrain_model(s_dataloaders, t_dataloaders, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('scModel')
    train_group = parser.add_mutually_exclusive_group(required=False)
    train_group.add_argument('--train', dest='retrain_flag', action='store_true')
    train_group.add_argument('--no-train', dest='retrain_flag', action='store_false')
    parser.set_defaults(retrain_flag=True)

    # parser.add_argument('--retrain_flag', type=bool, default=False)
    # pre-training paremeters setting
    parser.add_argument('--pretrain_num_epochs', type=int)
    parser.add_argument('--adv_num_epochs', type=int)
    parser.add_argument('--gap_num', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=128)  # 128
    parser.add_argument('--lr', type=float, default=0.0001) # 0.0001
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--device', default='2', type=int)

    # model paremeters setting
    parser.add_argument('--input_dim', type=int)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--classifier_hidden_dims', type=list, default=[256, 64])
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--d_model', type=int, default=512)  # 512
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--dim_feedforward', type=int, default=1024) 
    parser.add_argument('--norm_flag', type=bool, default=True)
    parser.add_argument('--alpha', type=float, default=1.0)

    # common parameters setting
    parser.add_argument('--es_flag', type=bool, default=False)
    parser.add_argument('--model_save_dir', type=str)
    parser.add_argument('--seed', default=2002, type=int)

    args = parser.parse_args()
    args.device = torch.device(args.device) if args.device >= 0 else torch.device('cpu')
    print(f'Current config is {args}')
    
    params_grid = {
        "pretrain_num_epochs": [50],  # 0, 10, 20, 30, 40, 50
        "adv_num_epochs": [300],  # 50, 100, 150, 200, 300, 400   
        "dropout": [0.1]  # 0.1
    }

    keys, values = zip(*params_grid.items())
    update_params_dict_list = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # logger = setup_logger(log_dir='logs/train_batch_' + str(args.batch_size) + '_5K/', log_filename_prefix="pre_training_parameters")
    logger = setup_logger(log_dir='ablation_model/no_adv/', log_filename_prefix="pre_training_parameters")

    start_time = time.time()
    for param_dict in update_params_dict_list:
        data_utils.set_seed(args.seed)
        main(args, param_dict, logger)

    end_time = time.time()
    elapsed_time = end_time - start_time

    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    # 输出结果
    print(f"模型运行时间: {hours}小时 {minutes}分钟 {seconds}秒")