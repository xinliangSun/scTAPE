from tqdm import tqdm
from itertools import cycle
from itertools import chain
from train_eval_utils import *
from model import Enformer, MLP, DSNFormer
from collections import defaultdict


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

        pretrained_dict = torch.load(os.path.join(args.model_save_dir, 'a_t_dsnae.pt'), map_location=args.device)
        t_dsnae.load_state_dict(pretrained_dict)  

        return t_dsnae.shared_encoder
    
    except FileNotFoundError:
            raise Exception("No pre-trained encoder!")



