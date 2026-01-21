import torch.distributed as dist

from tqdm import tqdm
from itertools import cycle
from itertools import chain
from train_eval_utils import *
from model import Enformer, MLP, DSNFormer
from collections import defaultdict
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


def train(s_dataloaders, t_dataloaders, args):

    source_train = s_dataloaders[0]
    source_test = s_dataloaders[1]

    target_train = t_dataloaders[0]
    target_test = t_dataloaders[1]
    
    shared_encoder = Enformer(num_layers=args.num_layers, d_model=args.d_model, 
                              nhead=args.nhead, dim_feedforward=args.dim_feedforward, 
                              dop=args.dropout).to(args.device)

    shared_decoder = MLP(input_dim=args.d_model * 2,
                         output_dim=args.input_dim,
                         hidden_dims=[1024],
                         dop=args.dropout).to(args.device)
    
    s_private_encoder = Enformer(num_layers=args.num_layers, d_model=args.d_model, 
                                nhead=args.nhead, dim_feedforward=args.dim_feedforward, 
                                dop=args.dropout).to(args.device)
    
    t_private_encoder = Enformer(num_layers=args.num_layers, d_model=args.d_model, 
                                nhead=args.nhead, dim_feedforward=args.dim_feedforward, 
                                dop=args.dropout).to(args.device)

    s_dsnformer = DSNFormer(shared_encoder=shared_encoder,
                            decoder=shared_decoder,
                            p_encoder=s_private_encoder,
                            alpha=args.alpha,
                            norm_flag=args.norm_flag).to(args.device)

    t_dsnformer = DSNFormer(shared_encoder=shared_encoder,
                            decoder=shared_decoder,
                            p_encoder=t_private_encoder,
                            alpha=args.alpha,
                            norm_flag=args.norm_flag).to(args.device)

    confounding_classifier = MLP(input_dim=args.d_model * 2,
                                 output_dim=1,
                                 hidden_dims=args.classifier_hidden_dims,
                                 dop=args.dropout).to(args.device)
    
    ae_params = [t_dsnformer.private_encoder.parameters(),
                 s_dsnformer.private_encoder.parameters(),
                 shared_encoder.parameters(),
                 shared_decoder.parameters()]
    
    t_ae_params = [t_dsnformer.private_encoder.parameters(),
                   s_dsnformer.private_encoder.parameters(),
                   shared_encoder.parameters(),
                   shared_decoder.parameters()]

    ae_optimizer = torch.optim.AdamW(chain(*ae_params), lr=args.lr)
    classifier_optimizer = torch.optim.RMSprop(confounding_classifier.parameters(), lr=args.lr)
    t_ae_optimizer = torch.optim.RMSprop(chain(*t_ae_params), lr=args.lr)

    if args.retrain_flag:
        print(f'\nStart Model Pre-training...')
        pbar = tqdm(total=args.pretrain_num_epochs, desc="Processing")
        for _ in range(int(args.pretrain_num_epochs)):
            train_history = defaultdict(list)
            val_history = defaultdict(list)
                
            for step, (s_batch, t_batch) in enumerate(zip(cycle(source_train), target_train)):       
                s_batch = [s_batch[0][0].to(args.device), s_batch[1][0].to(args.device)]
                t_batch = [t_batch[0][0].to(args.device), t_batch[1][0].to(args.device)]

                train_history = train_step(s_model=s_dsnformer,
                                           t_model=t_dsnformer,
                                           s_batch=s_batch,
                                           t_batch=t_batch,
                                           optimizer=ae_optimizer,
                                           history=train_history)
                
            mean_loss = {k: np.mean(v) if isinstance(v, list) else v for k, v in train_history.items()}
    
            val_history = eval_epoch(model=s_dsnformer,
                                     data_loader=source_test,
                                     device=args.device,
                                     history=val_history)
            
            val_history = eval_epoch(model=t_dsnformer,
                                     data_loader=target_test,
                                     device=args.device,
                                     history=val_history)
            
            for k in val_history:
                if k != 'best_index':
                    val_history[k][-2] += val_history[k][-1]
                    val_history[k].pop()

            # pbar.set_postfix({"Loss": f"{mean_loss['loss']:.4f}", "s_loss": f"{mean_loss['s_recons_loss']:.4f}", 
            #                 "t_loss": f"{mean_loss['t_recons_loss']:.4f}", "diff_loss": f"{mean_loss['diff_loss']:.4f}"})
            # 
            pbar.set_postfix({"Loss": f"{val_history['loss'][0]:.4f}", "recons_loss": f"{val_history['recons_loss'][0]:.4f}", 
                            "diff_loss": f"{val_history['diff_loss'][0]:.4f}"})
            pbar.update(1)

            if args.es_flag:
                save_flag, stop_flag = model_save_check(val_history, metric_name='loss', tolerance_count=20)
                if save_flag:
                    torch.save(s_dsnformer.state_dict(), os.path.join(args.model_save_dir, 'a_s_dsnae.pt'))
                    torch.save(t_dsnformer.state_dict(), os.path.join(args.model_save_dir, 'a_t_dsnae.pt'))
                if stop_flag:
                    break

        pbar.close()  

        # testing
        torch.save(s_dsnformer.state_dict(), os.path.join(args.model_save_dir, 'a_s_dsnae.pt'))
        torch.save(t_dsnformer.state_dict(), os.path.join(args.model_save_dir, 'a_t_dsnae.pt'))

        if args.es_flag:
            s_dsnformer.load_state_dict(torch.load(os.path.join(args.model_save_dir, 'a_s_dsnae.pt')))
            t_dsnformer.load_state_dict(torch.load(os.path.join(args.model_save_dir, 'a_t_dsnae.pt')))
        
        print(f'\nStart WGAN training...')
        for _ in tqdm(range(int(args.adv_num_epochs))):
            critic_train_history = defaultdict(list)
            gen_train_history = defaultdict(list)

            for step, (s_batch, t_batch) in enumerate(zip(cycle(source_train), target_train)):   
                s_batch = [s_batch[0][0].to(args.device), s_batch[1][0].to(args.device)]
                t_batch = [t_batch[0][0].to(args.device), t_batch[1][0].to(args.device)]
                          
                critic_train_history = critic_train_step(critic=confounding_classifier,
                                                         s_model=s_dsnformer,
                                                         t_model=t_dsnformer,
                                                         s_batch=s_batch,
                                                         t_batch=t_batch,
                                                         device=args.device,
                                                         optimizer=classifier_optimizer,
                                                         history=critic_train_history,
                                                         gp=10.0)
                if (step + 1) % 5 == 0:
                    gen_train_history = gan_gen_train_step(critic=confounding_classifier,
                                                           s_model=s_dsnformer,
                                                           t_model=t_dsnformer,
                                                           s_batch=s_batch,
                                                           t_batch=t_batch,
                                                           optimizer=t_ae_optimizer,
                                                           alpha=1.0,
                                                           history=gen_train_history)

        torch.save(s_dsnformer.state_dict(), os.path.join(args.model_save_dir, 'a_s_dsnae.pt'))
        torch.save(t_dsnformer.state_dict(), os.path.join(args.model_save_dir, 'a_t_dsnae.pt'))

    else:
        try:
            t_dsnformer.load_state_dict(torch.load(os.path.join(args.model_save_dir, 'a_t_dsnae.pt')))

        except FileNotFoundError:
            raise Exception("No pre-trained encoder")

    return t_dsnformer.shared_encoder


def load_model(args):
    try:
        shared_encoder = Enformer(num_layers=args.num_layers, d_model=args.d_model, 
                                  nhead=args.nhead, dim_feedforward=args.dim_feedforward, 
                                  dop=args.dropout).to(args.device)
        
        shared_decoder = MLP(input_dim=args.d_model * 2,
                             output_dim=args.input_dim,
                             hidden_dims=[1024],
                             dop=args.dropout).to(args.device)
        
        t_private_encoder = Enformer(num_layers=args.num_layers, d_model=args.d_model, 
                                     nhead=args.nhead, dim_feedforward=args.dim_feedforward, 
                                     dop=args.dropout).to(args.device)
        
        t_dsnae = DSNFormer(shared_encoder=shared_encoder,
                            decoder=shared_decoder,
                            p_encoder=t_private_encoder,
                            alpha=args.alpha,
                            norm_flag=args.norm_flag).to(args.device)
        
        # device = next(t_dsnae.parameters()).device

        # if all(param.device == device for param in t_dsnae.parameters()):
        #     print(f"All parameters are on device: {device}")
        # else:
        #     print("Model has parameters on different devices.")

        pretrained_dict = torch.load(os.path.join(args.model_save_dir, 'a_t_dsnae.pt'), map_location=args.device)
        t_dsnae.load_state_dict(pretrained_dict)  

        return t_dsnae.shared_encoder
    
    except FileNotFoundError:
            raise Exception("No pre-trained encoder!")


def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)  # 使用 NCCL 后端
    torch.cuda.set_device(rank)
    torch.backends.cudnn.benchmark = True


def cleanup():
    dist.destroy_process_group()


def train_DDP(rank, world_size, s_dataloaders, t_dataloaders, args):
    print(args)
    setup(rank, world_size)
    source_train = s_dataloaders[0]
    source_test = s_dataloaders[1]

    target_train = t_dataloaders[0]
    target_test = t_dataloaders[1]

    s_train_sampler = DistributedSampler(source_train, num_replicas=world_size, rank=rank, shuffle=True)
    t_train_sampler = DistributedSampler(target_train, num_replicas=world_size, rank=rank, shuffle=True)
    s_train_dataloader = DataLoader(source_train, batch_size=args.batch_size, sampler=s_train_sampler)
    t_train_dataloader = DataLoader(target_train, batch_size=args.batch_size, sampler=t_train_sampler)

    shared_encoder = Enformer(num_layers=args.num_layers, d_model=args.d_model, 
                              nhead=args.nhead, dim_feedforward=args.dim_feedforward, 
                              dop=args.dropout).to(rank)

    shared_decoder = MLP(input_dim=args.d_model * 2,
                         output_dim=args.input_dim,
                         hidden_dims=[1024],
                         dop=args.dropout).to(rank)
    
    s_private_encoder = Enformer(num_layers=args.num_layers, d_model=args.d_model, 
                                nhead=args.nhead, dim_feedforward=args.dim_feedforward, 
                                dop=args.dropout).to(rank)
    
    t_private_encoder = Enformer(num_layers=args.num_layers, d_model=args.d_model, 
                                nhead=args.nhead, dim_feedforward=args.dim_feedforward, 
                                dop=args.dropout).to(rank)
    
    shared_encoder = DDP(shared_encoder, device_ids=[rank])
    shared_decoder = DDP(shared_decoder, device_ids=[rank])

    s_dsnformer = DSNFormer(shared_encoder=shared_encoder,
                            decoder=shared_decoder,
                            p_encoder=s_private_encoder,
                            alpha=args.alpha,
                            norm_flag=args.norm_flag).to(rank)

    t_dsnformer = DSNFormer(shared_encoder=shared_encoder,
                            decoder=shared_decoder,
                            p_encoder=t_private_encoder,
                            alpha=args.alpha,
                            norm_flag=args.norm_flag).to(rank)

    s_dsnformer = DDP(s_dsnformer, device_ids=[rank])
    t_dsnformer = DDP(t_dsnformer, device_ids=[rank])

    confounding_classifier = MLP(input_dim=args.d_model * 2,
                                 output_dim=1,
                                 hidden_dims=args.classifier_hidden_dims,
                                 dop=args.dropout).to(rank)
    
    confounding_classifier = DDP(confounding_classifier, device_ids=[rank])

    ae_params = [t_dsnformer.module.private_encoder.parameters(),
                 s_dsnformer.module.private_encoder.parameters(),
                 shared_encoder.module.parameters(),
                 shared_decoder.module.parameters()]
    
    t_ae_params = [t_dsnformer.module.private_encoder.parameters(),
                   s_dsnformer.module.private_encoder.parameters(),
                   shared_encoder.module.parameters(),
                   shared_decoder.module.parameters()]

    ae_optimizer = torch.optim.AdamW(chain(*ae_params), lr=args.lr)
    classifier_optimizer = torch.optim.RMSprop(confounding_classifier.parameters(), lr=args.lr)
    t_ae_optimizer = torch.optim.RMSprop(chain(*t_ae_params), lr=args.lr)

    if args.retrain_flag:
        print(f'\nStart Model Pre-training...')
        if rank == 0:
            pbar = tqdm(total=args.pretrain_num_epochs, desc="Processing")
        for epoch in range(int(args.pretrain_num_epochs)):
            train_history = defaultdict(list)
            val_history = defaultdict(list)

            s_train_sampler.set_epoch(epoch)
            t_train_sampler.set_epoch(epoch)

            for step, (s_batch, t_batch) in enumerate(zip(cycle(s_train_dataloader), t_train_dataloader)):
 
                s_batch = [s_batch[0][0].to(rank), s_batch[1][0].to(rank)]
                t_batch = [t_batch[0][0].to(rank), t_batch[1][0].to(rank)]

                train_history = train_step(s_model=s_dsnformer,
                                           t_model=t_dsnformer,
                                           s_batch=s_batch,
                                           t_batch=t_batch,
                                           optimizer=ae_optimizer,
                                           history=train_history)
                
            mean_loss = {k: np.mean(v) if isinstance(v, list) else v for k, v in train_history.items()}
    
            val_history = eval_epoch_ddp(model=s_dsnformer,
                                     data_loader=source_test,
                                     device=args.device,
                                     history=val_history,
                                     rank=rank)
            
            val_history = eval_epoch_ddp(model=t_dsnformer,
                                     data_loader=target_test,
                                     device=args.device,
                                     history=val_history,
                                     rank=rank)
            
            for k in val_history:
                if k != 'best_index':
                    val_history[k][-2] += val_history[k][-1]
                    val_history[k].pop()

            # pbar.set_postfix({"Loss": f"{mean_loss['loss']:.4f}", "s_loss": f"{mean_loss['s_recons_loss']:.4f}", 
            #                 "t_loss": f"{mean_loss['t_recons_loss']:.4f}", "diff_loss": f"{mean_loss['diff_loss']:.4f}"})
            if rank == 0:
                pbar.set_postfix({"Loss": f"{val_history['loss'][0]:.4f}", "recons_loss": f"{val_history['recons_loss'][0]:.4f}", 
                             "diff_loss": f"{val_history['diff_loss'][0]:.4f}"})
                pbar.update(1)

            if args.es_flag:
                save_flag, stop_flag = model_save_check(val_history, metric_name='loss', tolerance_count=20)
                if save_flag:
                    torch.save(s_dsnformer.state_dict(), os.path.join(args.model_save_dir, 'a_s_dsnae.pt'))
                    torch.save(t_dsnformer.state_dict(), os.path.join(args.model_save_dir, 'a_t_dsnae.pt'))
                if stop_flag:
                    break

        if rank == 0:
            pbar.close()  

        if args.es_flag:
            s_dsnformer.load_state_dict(torch.load(os.path.join(args.model_save_dir, 'a_s_dsnae.pt')))
            t_dsnformer.load_state_dict(torch.load(os.path.join(args.model_save_dir, 'a_t_dsnae.pt')))
        
        print(f'\nStart GAN training...')
        for _ in tqdm(range(int(args.adv_num_epochs))):
            critic_train_history = defaultdict(list)
            gen_train_history = defaultdict(list)

            for step, (s_batch, t_batch) in enumerate(zip(cycle(source_train), target_train)):   
                s_batch = [s_batch[0][0].to(rank), s_batch[1][0].to(rank)]
                t_batch = [t_batch[0][0].to(rank), t_batch[1][0].to(rank)]
                          
                critic_train_history = critic_train_step(critic=confounding_classifier,
                                                         s_model=s_dsnformer,
                                                         t_model=t_dsnformer,
                                                         s_batch=s_batch,
                                                         t_batch=t_batch,
                                                         device=args.device,
                                                         optimizer=classifier_optimizer,
                                                         history=critic_train_history,
                                                         gp=10.0)
                if (step + 1) % 5 == 0:
                    gen_train_history = gan_gen_train_step(critic=confounding_classifier,
                                                           s_model=s_dsnformer,
                                                           t_model=t_dsnformer,
                                                           s_batch=s_batch,
                                                           t_batch=t_batch,
                                                           device=args.device,
                                                           optimizer=t_ae_optimizer,
                                                           alpha=1.0,
                                                           history=gen_train_history)

        if range == 0:
            torch.save(s_dsnformer.state_dict(), os.path.join(args.model_save_dir, 'a_s_dsnae.pt'))
            torch.save(t_dsnformer.state_dict(), os.path.join(args.model_save_dir, 'a_t_dsnae.pt'))
        
        cleanup()
    else:
        try:
            t_dsnformer.load_state_dict(torch.load(os.path.join(args.model_save_dir, 'a_t_dsnae.pt')))

        except FileNotFoundError:
            raise Exception("No pre-trained encoder")

    # return t_dsnformer.shared_encoder


# import torch
# import torch.nn as nn
# import torch.distributed as dist
# import torch.multiprocessing as mp
# from torch.nn.parallel import DistributedDataParallel as DDP
# from tqdm import tqdm
# import os
# from collections import defaultdict
# import numpy as np
# from itertools import chain
# from torch.utils.data import DataLoader
# from torch.utils.data.distributed import DistributedSampler


# def setup(rank, world_size, backend="nccl"):
#     """ Initialize the distributed environment. """
#     dist.init_process_group(
#         backend=backend,
#         init_method='env://',  # You can use file:// or tcp:// for different setups
#         world_size=world_size,
#         rank=rank
#     )
#     torch.cuda.set_device(rank)  # Set the device for this rank


# def cleanup():
#     """ Clean up the distributed environment. """
#     dist.destroy_process_group()


# def train_model_DDP(rank, world_size, s_dataloaders, t_dataloaders, args):
#     setup(rank, world_size)

#     # Initialize data loaders with DistributedSampler
#     s_train_dataloader_v = DataLoader(s_dataloaders[0][0], sampler=DistributedSampler(s_dataloaders[0][0], num_replicas=world_size, rank=rank))
#     s_test_dataloader_v = DataLoader(s_dataloaders[0][1], sampler=DistributedSampler(s_dataloaders[0][1], num_replicas=world_size, rank=rank))

#     s_train_dataloader_i = DataLoader(s_dataloaders[1][0], sampler=DistributedSampler(s_dataloaders[1][0], num_replicas=world_size, rank=rank))
#     s_test_dataloader_i = DataLoader(s_dataloaders[1][1], sampler=DistributedSampler(s_dataloaders[1][1], num_replicas=world_size, rank=rank))

#     t_train_dataloader_v = DataLoader(t_dataloaders[0][0], sampler=DistributedSampler(t_dataloaders[0][0], num_replicas=world_size, rank=rank))
#     t_test_dataloader_v = DataLoader(t_dataloaders[0][1], sampler=DistributedSampler(t_dataloaders[0][1], num_replicas=world_size, rank=rank))

#     t_train_dataloader_i = DataLoader(t_dataloaders[1][0], sampler=DistributedSampler(t_dataloaders[1][0], num_replicas=world_size, rank=rank))
#     t_test_dataloader_i = DataLoader(t_dataloaders[1][1], sampler=DistributedSampler(t_dataloaders[1][1], num_replicas=world_size, rank=rank))

#     # Initialize models
#     shared_encoder = Enformer(num_layers=args.num_layers, d_model=args.d_model, 
#                               nhead=args.nhead, dim_feedforward=args.dim_feedforward, 
#                               dop=args.dropout).to(rank)

#     shared_decoder = MLP(input_dim=args.d_model * 2,
#                          output_dim=args.input_dim,
#                          hidden_dims=[1024],
#                          dop=args.dropout).to(rank)

#     s_dsnae = DSNFormer(shared_encoder=shared_encoder,
#                         decoder=shared_decoder,
#                         alpha=args.alpha,
#                         input_dim=args.d_model,
#                         dim_feedforward=args.dim_feedforward,
#                         nhead=args.nhead,
#                         num_layers=args.num_layers,
#                         dop=args.dropout,
#                         norm_flag=args.norm_flag).to(rank)

#     t_dsnae = DSNFormer(shared_encoder=shared_encoder,
#                         decoder=shared_decoder,
#                         alpha=args.alpha,
#                         input_dim=args.d_model,
#                         dim_feedforward=args.dim_feedforward,
#                         nhead=args.nhead,
#                         num_layers=args.num_layers,
#                         dop=args.dropout,
#                         norm_flag=args.norm_flag).to(rank)

#     confounding_classifier = MLP(input_dim=args.d_model * 2,
#                                  output_dim=1,
#                                  hidden_dims=args.classifier_hidden_dims,
#                                  dop=args.dropout).to(rank)

#     # Wrap models with DDP
#     shared_encoder = DDP(shared_encoder, device_ids=[rank])
#     shared_decoder = DDP(shared_decoder, device_ids=[rank])
#     s_dsnae = DDP(s_dsnae, device_ids=[rank])
#     t_dsnae = DDP(t_dsnae, device_ids=[rank])
#     confounding_classifier = DDP(confounding_classifier, device_ids=[rank])

#     ae_params = [t_dsnae.private_encoder.parameters(),
#                  s_dsnae.private_encoder.parameters(),
#                  shared_encoder.parameters(),
#                  shared_decoder.parameters()]

#     t_ae_params = [t_dsnae.private_encoder.parameters(),
#                    s_dsnae.private_encoder.parameters(),
#                    shared_encoder.parameters(),
#                    shared_decoder.parameters()]

#     ae_optimizer = torch.optim.AdamW(chain(*ae_params), lr=args.lr)
#     classifier_optimizer = torch.optim.RMSprop(confounding_classifier.parameters(), lr=args.lr)
#     t_ae_optimizer = torch.optim.RMSprop(chain(*t_ae_params), lr=args.lr)

#     dsnae_train_history = defaultdict(list)
#     dsnae_val_history = defaultdict(list)
#     critic_train_history = defaultdict(list)
#     gen_train_history = defaultdict(list)

#     # Start model pre-training
#     if args.retrain_flag:
#         print(f'\n\tStart Model Pre-training...')
#         pbar = tqdm(total=args.pretrain_num_epochs, desc="Processing")
#         for _ in range(int(args.pretrain_num_epochs)):
#             s_iter_v = iter(s_train_dataloader_v)
#             s_iter_i = iter(s_train_dataloader_i)
#             t_iter_i = iter(t_train_dataloader_i)

#             for step, t_batch_v in enumerate(t_train_dataloader_v):
#                 try:
#                     s_batch_v = next(s_iter_v)
#                     s_batch_i = next(s_iter_i)
#                     t_batch_i = next(t_iter_i)
#                 except StopIteration:
#                     s_iter_v = iter(s_train_dataloader_v)
#                     s_iter_i = iter(s_train_dataloader_i)
#                     # t_iter_i = iter(t_train_dataloader_i)
#                     s_batch_v = next(s_iter_v)
#                     s_batch_i = next(s_iter_i)
#                     t_batch_i = next(t_iter_i)

#                 s_batch = [s_batch_v, s_batch_i]
#                 t_batch = [t_batch_v, t_batch_i]
#                 dsnae_train_history = dsnae_train_step(s_dsnae=s_dsnae,
#                                                        t_dsnae=t_dsnae,
#                                                        s_batch=s_batch,
#                                                        t_batch=t_batch,
#                                                        device=rank,
#                                                        optimizer=ae_optimizer,
#                                                        history=dsnae_train_history)
                
#             mean_loss = {k: np.mean(v) if isinstance(v, list) else v for k, v in dsnae_train_history.items()}

#             # Evaluate the model on validation data
#             s_test_dataloader = [s_test_dataloader_v, s_test_dataloader_i]
#             dsnae_val_history = eval_dsnae_epoch(model=s_dsnae,
#                                                  data_loader=s_test_dataloader,
#                                                  device=rank,
#                                                  history=dsnae_val_history)
            
#             t_test_dataloader = [t_test_dataloader_v, t_test_dataloader_i]
#             dsnae_val_history = eval_dsnae_epoch(model=t_dsnae,
#                                                  data_loader=t_test_dataloader,
#                                                  device=rank,
#                                                  history=dsnae_val_history)
            
#             pbar.set_postfix({"Loss": f"{mean_loss['loss']:.4f}", "s_loss": f"{mean_loss['s_recons_loss']:.4f}", 
#                               "t_loss": f"{mean_loss['t_recons_loss']:.4f}", "diff_loss": f"{mean_loss['diff_loss']:.4f}"})
#             pbar.update(1)

#             for k in dsnae_val_history:
#                 if k != 'best_index':
#                     dsnae_val_history[k][-2] += dsnae_val_history[k][-1]
#                     dsnae_val_history[k].pop()

#             if args.es_flag:
#                 save_flag, stop_flag = model_save_check(dsnae_val_history, metric_name='loss', tolerance_count=20)
#                 if save_flag:
#                     torch.save(s_dsnae.state_dict(), os.path.join(args.model_save_dir, 'a_s_dsnae.pt'))
#                     torch.save(t_dsnae.state_dict(), os.path.join(args.model_save_dir, 'a_t_dsnae.pt'))
#                 if stop_flag:
#                     break

#         pbar.close()

#         if args.es_flag:
#             s_dsnae.load_state_dict(torch.load(os.path.join(args.model_save_dir, 'a_s_dsnae.pt')))
#             t_dsnae.load_state_dict(torch.load(os.path.join(args.model_save_dir, 'a_t_dsnae.pt')))

#         print(f'\n\tStart GAN training...')
#         for _ in tqdm(range(int(args.adv_num_epochs))):
#             s_iter_v = iter(s_train_dataloader_v)
#             s_iter_i = iter(s_train_dataloader_i)
#             t_iter_i = iter(t_train_dataloader_i)

#             for step, t_batch_v in enumerate(t_train_dataloader_v):
#                 try:
#                     s_batch_v = next(s_iter_v)
#                     s_batch_i = next(s_iter_i)
#                     t_batch_i = next(t_iter_i)
#                 except StopIteration:
#                     s_iter_v = iter(s_train_dataloader_v)
#                     s_iter_i = iter(s_train_dataloader_i)
#                     # t_iter_i = iter(t_train_dataloader_i)
#                     s_batch_v = next(s_iter_v)
#                     s_batch_i = next(s_iter_i)
#                     t_batch_i = next(t_iter_i) 

#                 s_batch = [s_batch_v, s_batch_i]
#                 t_batch = [t_batch_v, t_batch_i]
#                 critic_train_history = critic_dsn_train_step(critic=confounding_classifier,
#                                                              s_dsnae=s_dsnae,
#                                                              t_dsnae=t_dsnae,
#                                                              s_batch=s_batch,
#                                                              t_batch=t_batch,
#                                                              device=rank,
#                                                              optimizer=classifier_optimizer,
#                                                              history=critic_train_history,
#                                                              gp=10.0)

#                 if (step + 1) % 5 == 0:
#                     gen_train_history = gan_dsn_gen_train_step(critic=confounding_classifier,
#                                                                s_dsnae=s_dsnae,
#                                                                t_dsnae=t_dsnae,
#                                                                s_batch=s_batch,
#                                                                t_batch=t_batch,
#                                                                device=rank,
#                                                                optimizer=t_ae_optimizer,
#                                                                alpha=1.0,
#                                                                history=gen_train_history)

#         # Save the final models after GAN training
#         torch.save(s_dsnae.state_dict(), os.path.join(args.model_save_dir, 'a_s_dsnae.pt'))
#         torch.save(t_dsnae.state_dict(), os.path.join(args.model_save_dir, 'a_t_dsnae.pt'))

#     else:
#         try:
#             t_dsnae.load_state_dict(torch.load(os.path.join(args.model_save_dir, 'a_t_dsnae.pt')))

#         except FileNotFoundError:
#             raise Exception("No pre-trained encoder")

#     cleanup()

#     return t_dsnae.shared_encoder, (dsnae_train_history, dsnae_val_history, critic_train_history, gen_train_history)


# def train_model_muliGPU(s_dataloaders, t_dataloaders, args):
#     s_train_dataloader_v = s_dataloaders[0][0]
#     s_test_dataloader_v = s_dataloaders[0][1]

#     s_train_dataloader_i = s_dataloaders[1][0]
#     s_test_dataloader_i = s_dataloaders[1][1]

#     t_train_dataloader_v = t_dataloaders[0][0]
#     t_test_dataloader_v = t_dataloaders[0][1]

#     t_train_dataloader_i = t_dataloaders[1][0]
#     t_test_dataloader_i = t_dataloaders[1][1]
    

#     shared_encoder = Enformer(num_layers=args.num_layers, d_model=args.d_model, 
#                               nhead=args.nhead, dim_feedforward=args.dim_feedforward, 
#                               dop=args.dropout)

#     shared_decoder = MLP(input_dim=args.d_model * 2,
#                          output_dim=args.input_dim,
#                          hidden_dims=[1024],
#                          dop=args.dropout)

#     s_dsnae = DSNFormer(shared_encoder=shared_encoder,
#                         decoder=shared_decoder,
#                         alpha=args.alpha,
#                         input_dim=args.d_model,
#                         dim_feedforward=args.dim_feedforward,
#                         nhead=args.nhead,
#                         num_layers=args.num_layers,
#                         dop=args.dropout,
#                         norm_flag=args.norm_flag)

#     t_dsnae = DSNFormer(shared_encoder=shared_encoder,
#                         decoder=shared_decoder,
#                         alpha=args.alpha,
#                         input_dim=args.d_model,
#                         dim_feedforward=args.dim_feedforward,
#                         nhead=args.nhead,
#                         num_layers=args.num_layers,
#                         dop=args.dropout,
#                         norm_flag=args.norm_flag)

#     confounding_classifier = MLP(input_dim=args.d_model * 2,
#                                  output_dim=1,
#                                  hidden_dims=args.classifier_hidden_dims,
#                                  dop=args.dropout)

#     # 将模型移动到指定设备（例如 GPU 0）
#     device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
#     shared_encoder = shared_encoder.to(device)
#     shared_decoder = shared_decoder.to(device)
#     s_dsnae = s_dsnae.to(device)
#     t_dsnae = t_dsnae.to(device)
#     confounding_classifier = confounding_classifier.to(device)

#     # 使用 nn.DataParallel 将模型包装，以支持多 GPU 训练
#     if torch.cuda.device_count() > 1:
#         device_ids = list(range(torch.cuda.device_count()))  # 使用所有可用的 GPU
#         shared_encoder = nn.DataParallel(shared_encoder, device_ids=device_ids)
#         shared_decoder = nn.DataParallel(shared_decoder, device_ids=device_ids)
#         s_dsnae = nn.DataParallel(s_dsnae, device_ids=device_ids)
#         t_dsnae = nn.DataParallel(t_dsnae, device_ids=device_ids)
#         confounding_classifier = nn.DataParallel(confounding_classifier, device_ids=device_ids)

#     # 设置优化器
#     ae_params = [t_dsnae.module.private_encoder.parameters() if isinstance(t_dsnae, nn.DataParallel) else t_dsnae.private_encoder.parameters(),
#                  s_dsnae.module.private_encoder.parameters() if isinstance(s_dsnae, nn.DataParallel) else s_dsnae.private_encoder.parameters(),
#                  shared_encoder.module.parameters() if isinstance(shared_encoder, nn.DataParallel) else shared_encoder.parameters(),
#                  shared_decoder.module.parameters() if isinstance(shared_decoder, nn.DataParallel) else shared_decoder.parameters()]

#     t_ae_params = [t_dsnae.module.private_encoder.parameters() if isinstance(t_dsnae, nn.DataParallel) else t_dsnae.private_encoder.parameters(),
#                    s_dsnae.module.private_encoder.parameters() if isinstance(s_dsnae, nn.DataParallel) else s_dsnae.private_encoder.parameters(),
#                    shared_encoder.module.parameters() if isinstance(shared_encoder, nn.DataParallel) else shared_encoder.parameters(),
#                    shared_decoder.module.parameters() if isinstance(shared_decoder, nn.DataParallel) else shared_decoder.parameters()]

#     ae_optimizer = torch.optim.AdamW(chain(*ae_params), lr=args.lr)
#     classifier_optimizer = torch.optim.RMSprop(confounding_classifier.parameters(), lr=args.lr)
#     t_ae_optimizer = torch.optim.RMSprop(chain(*t_ae_params), lr=args.lr)

#     dsnae_train_history = defaultdict(list)
#     dsnae_val_history = defaultdict(list)
#     critic_train_history = defaultdict(list)
#     gen_train_history = defaultdict(list)

#     if args.retrain_flag:
#         print(f'\n\tStart Model Pre-training...')
#         pbar = tqdm(total=args.pretrain_num_epochs, desc="Processing")
#         for _ in range(int(args.pretrain_num_epochs)):
#             s_iter_v = iter(s_train_dataloader_v)
#             s_iter_i = iter(s_train_dataloader_i)
#             t_iter_i = iter(t_train_dataloader_i)

#             for step, t_batch_v in enumerate(t_train_dataloader_v):
#                 try:
#                     s_batch_v = next(s_iter_v)
#                     s_batch_i = next(s_iter_i)
#                     t_batch_i = next(t_iter_i)
#                 except StopIteration:
#                     s_iter_v = iter(s_train_dataloader_v)
#                     s_iter_i = iter(s_train_dataloader_i)
#                     t_iter_i = iter(t_train_dataloader_i)
#                     s_batch_v = next(s_iter_v)
#                     s_batch_i = next(s_iter_i)
#                     t_batch_i = next(t_iter_i)
                
#                 s_batch = [s_batch_v, s_batch_i]
#                 t_batch = [t_batch_v, t_batch_i]
#                 dsnae_train_history = dsnae_train_step(s_dsnae=s_dsnae,
#                                                        t_dsnae=t_dsnae,
#                                                        s_batch=s_batch,
#                                                        t_batch=t_batch,
#                                                        device=device,
#                                                        optimizer=ae_optimizer,
#                                                        history=dsnae_train_history)
                
#             mean_loss = {k: np.mean(v) if isinstance(v, list) else v for k, v in dsnae_train_history.items()}

#             s_test_dataloader = [s_test_dataloader_v, s_test_dataloader_i]
#             dsnae_val_history = eval_dsnae_epoch(model=s_dsnae,
#                                                  data_loader=s_test_dataloader,
#                                                  device=device,
#                                                  history=dsnae_val_history)
            
#             t_test_dataloader = [t_test_dataloader_v, t_test_dataloader_i]
#             dsnae_val_history = eval_dsnae_epoch(model=t_dsnae,
#                                                  data_loader=t_test_dataloader,
#                                                  device=device,
#                                                  history=dsnae_val_history)
            
#             pbar.set_postfix({"Loss": f"{mean_loss['loss']:.4f}", "s_loss": f"{mean_loss['s_recons_loss']:.4f}", 
#                               "t_loss": f"{mean_loss['t_recons_loss']:.4f}", "diff_loss": f"{mean_loss['diff_loss']:.4f}"})
#             pbar.update(1)

#             if args.es_flag:
#                 save_flag, stop_flag = model_save_check(dsnae_val_history, metric_name='loss', tolerance_count=20)
#                 if save_flag:
#                     torch.save(s_dsnae.module.state_dict(), os.path.join(args.model_save_dir, 'a_s_dsnae.pt'))
#                     torch.save(t_dsnae.module.state_dict(), os.path.join(args.model_save_dir, 'a_t_dsnae.pt'))
#                 if stop_flag:
#                     break

#         pbar.close()

#         # Load best model if early stopping is enabled
#         if args.es_flag:
#             s_dsnae.load_state_dict(torch.load(os.path.join(args.model_save_dir, 'a_s_dsnae.pt')))
#             t_dsnae.load_state_dict(torch.load(os.path.join(args.model_save_dir, 'a_t_dsnae.pt')))

#         # 开始 GAN 训练阶段
#         print(f'\n\tStart GAN training...')
#         for _ in tqdm(range(int(args.adv_num_epochs))):
#             s_iter_v = iter(s_train_dataloader_v)
#             s_iter_i = iter(s_train_dataloader_i)
#             t_iter_i = iter(t_train_dataloader_i)

#             for step, t_batch_v in enumerate(t_train_dataloader_v):
#                 try:
#                     s_batch_v = next(s_iter_v)
#                     s_batch_i = next(s_iter_i)
#                     t_batch_i = next(t_iter_i)
#                 except StopIteration:
#                     s_iter_v = iter(s_train_dataloader_v)
#                     s_iter_i = iter(s_train_dataloader_i)
#                     t_iter_i = iter(t_train_dataloader_i)
#                     s_batch_v = next(s_iter_v)
#                     s_batch_i = next(s_iter_i)
#                     t_batch_i = next(t_iter_i)                
                
#                 s_batch = [s_batch_v, s_batch_i]
#                 t_batch = [t_batch_v, t_batch_i]
#                 critic_train_history = critic_dsn_train_step(critic=confounding_classifier,
#                                                              s_dsnae=s_dsnae,
#                                                              t_dsnae=t_dsnae,
#                                                              s_batch=s_batch,
#                                                              t_batch=t_batch,
#                                                              device=device,
#                                                              optimizer=classifier_optimizer,
#                                                              history=critic_train_history,
#                                                              gp=10.0)
#                 if (step + 1) % 5 == 0:
#                     gen_train_history = gan_dsn_gen_train_step(critic=confounding_classifier,
#                                                                s_dsnae=s_dsnae,
#                                                                t_dsnae=t_dsnae,
#                                                                s_batch=s_batch,
#                                                                t_batch=t_batch,
#                                                                device=device,
#                                                                optimizer=t_ae_optimizer,
#                                                                alpha=1.0,
#                                                                history=gen_train_history)

#         torch.save(s_dsnae.module.state_dict(), os.path.join(args.model_save_dir, 'a_s_dsnae.pt'))
#         torch.save(t_dsnae.module.state_dict(), os.path.join(args.model_save_dir, 'a_t_dsnae.pt'))

#     else:
#         try:
#             t_dsnae.load_state_dict(torch.load(os.path.join(args.model_save_dir, 'a_t_dsnae.pt')))
#         except FileNotFoundError:
#             raise Exception("No pre-trained encoder")

#     return t_dsnae.shared_encoder, (dsnae_train_history, dsnae_val_history, critic_train_history, gen_train_history)

