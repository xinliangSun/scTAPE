import os
import torch
import torch.nn as nn

from tqdm import tqdm
from train_eval_utils import evaluate_epoch, model_save_check, evaluate_patient_epoch
from collections import defaultdict
from itertools import chain
from model import EncoderDecoder, Classify, CellClassify


def classification_train_step(args, model, batch, loss_fn, device, optimizer, history, scheduler=None, clip=None):
    model.zero_grad()
    model.train()

    x = batch[0].to(device)
    y = batch[1].to(device)
    loss = loss_fn(model(x), y.double().unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    if clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    history['bce_loss'].append(loss.cpu().detach().item())

    return history


def fine_tune_encoder(args, encoder, train_dataloader, val_dataloader, test_dataloader, 
                      task_save_folder, logger, param_str):
    
    if args.task == 'cell_line':
        target_decoder = CellClassify(input_dim=args.d_model, hidden_dims=args.classifier_dims).to(args.device)
    else:
        target_decoder = Classify(input_dim=args.d_model, hidden_dims=args.classifier_dims).to(args.device)
        
    target_classifier = EncoderDecoder(encoder, target_decoder, args.norm_flag).to(args.device)
    
    cls_loss = nn.BCEWithLogitsLoss()
    train_history = defaultdict(list)
    eval_train_history = defaultdict(list)
    eval_val_history = defaultdict(list)
    eval_test_history = defaultdict(list)
    
    encoder_module_indices = [i for i, module in enumerate(encoder.modules())
                              if isinstance(module, nn.Linear)]

    reset_count = 1
    target_params = [target_classifier.decoder.parameters()]
    target_optimizer = torch.optim.AdamW(chain(*target_params), lr=args.lr)

    pbar = tqdm(total=args.ft_epochs, desc="Processing")
    for epoch in range(args.ft_epochs):
        for _, batch in enumerate(train_dataloader):
            train_history = classification_train_step(args=args,
                                                      model=target_classifier,
                                                      batch=batch,
                                                      loss_fn=cls_loss,
                                                      device=args.device,
                                                      optimizer=target_optimizer,
                                                      history=train_history)
    
        eval_val_history, results = evaluate_epoch(classifier=target_classifier,
                                                   dataloader=val_dataloader,
                                                   device=args.device,
                                                   history=eval_val_history,
                                                   flag='val',
                                                   param_str=param_str,
                                                   args=args)

        save_flag, stop_flag = model_save_check(history=eval_val_history,
                                                metric_name=args.metric,
                                                tolerance_count=10,
                                                reset_count=reset_count)
        if save_flag:
            torch.save(target_classifier.state_dict(),
                       os.path.join(task_save_folder, f'target_classifier.pt'))
            
        if stop_flag:
            try:
                ind = encoder_module_indices.pop()
                print(f'Unfreezing {epoch}')
                target_classifier.load_state_dict(
                    torch.load(os.path.join(task_save_folder, f'target_classifier.pt')))
                
                target_params.append(list(target_classifier.encoder.modules())[ind].parameters())
                lr = args.lr * args.decay_coefficient
                target_optimizer = torch.optim.AdamW(chain(*target_params), lr=lr)
                reset_count += 1
            except IndexError:
                break
        
        pbar.set_postfix({"Loss": f"{results['loss']:.4f}"})
        pbar.update(1) 
    pbar.close()

    target_classifier.load_state_dict(torch.load(os.path.join(task_save_folder, f'target_classifier.pt')))
    
    if args.task == 'patient':
        _, sc_results = evaluate_patient_epoch(classifier=target_classifier,
                                               dataloader=test_dataloader,
                                               device=args.device,
                                               history=eval_test_history,
                                               flag='test',
                                               param_str=param_str,
                                               args=args)
    else:
        _, sc_results = evaluate_epoch(classifier=target_classifier,
                                       dataloader=test_dataloader,
                                       device=args.device,
                                       history=eval_test_history,
                                       flag='test',
                                       param_str=param_str,
                                       args=args)        
    if args.task == 'cell_line':
        logger.info(f"""Epoch {epoch+1}/{args.ft_epochs}, Loss: {sc_results['loss']:.4f}, AUROC: {sc_results['auroc']:.4f}, AUPRC: {sc_results['auprc']:.4f}""")
    else:
        logger.info(f"""Epoch {epoch+1}/{args.ft_epochs}, AUROC: {sc_results['auroc']:.4f}, AUPRC: {sc_results['auprc']:.4f}""")

    return sc_results