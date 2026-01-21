import os
import re
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.autograd as autograd
import sklearn.metrics as metrics

from collections import defaultdict
from captum.attr import IntegratedGradients

def z_score(arr):
    mean = np.mean(arr)
    std = np.std(arr)
    z_score = (arr - mean) / std
    return z_score


def train_step(s_model, t_model, s_batch, t_batch, optimizer, history):

    s_model.train()
    t_model.train()

    s_x_v, s_x_i = s_batch[0], s_batch[1]
    s_x = [s_x_v, s_x_i]

    t_x_v, t_x_i = t_batch[0], t_batch[1]
    t_x = [t_x_v, t_x_i]

    s_loss_dict = s_model.loss_function(*s_model(s_x))
    t_loss_dict = t_model.loss_function(*t_model(t_x))

    # if isinstance(s_model, torch.nn.parallel.DistributedDataParallel):
    #     s_loss_dict = s_model.module.loss_function(*s_model(s_x))
    # else:
    #     s_loss_dict = s_model.loss_function(*s_model(s_x))

    # if isinstance(t_model, torch.nn.parallel.DistributedDataParallel):
    #     t_loss_dict = t_model.module.loss_function(*t_model(s_x))
    # else:
    #     t_loss_dict = t_model.loss_function(*t_model(t_x))    

    loss = s_loss_dict['loss'] + t_loss_dict['loss']

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # if scheduler is not None:
    #     scheduler.step()

    loss_dict = {k: v.cpu().detach().item() + t_loss_dict[k].cpu().detach().item() for k, v in s_loss_dict.items()}
    
    s_recons_loss, s_diff_loss = s_loss_dict['recons_loss'].cpu().detach().item(), s_loss_dict['diff_loss'].cpu().detach().item()
    t_recons_loss, t_diff_loss = t_loss_dict['recons_loss'].cpu().detach().item(), t_loss_dict['diff_loss'].cpu().detach().item()

    loss_dict.update({"s_recons_loss": s_recons_loss, "s_diff_loss": s_diff_loss, 
                      "t_recons_loss": t_recons_loss, "t_diff_loss": t_diff_loss})
    
    # loss_dict.update({"s_recons_loss": s_recons_loss, "t_recons_loss": t_recons_loss})
    
    for k, v in loss_dict.items():
        history[k].append(v)

    return history


def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand((real_samples.shape[0], 1)).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    critic_interpolates = critic(interpolates)
    fakes = torch.ones((real_samples.shape[0], 1)).to(device)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=fakes,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def critic_train_step(critic, s_model, t_model, s_batch, t_batch, device, optimizer, 
                      history, scheduler=None, clip=None, gp=None):
    
    s_model.eval()
    t_model.eval()
    critic.train()

    s_x_v, t_x_v = s_batch[0], t_batch[0]

    s_code = s_model.encode(s_x_v).detach()
    t_code = t_model.encode(t_x_v).detach()

    # if isinstance(s_model, torch.nn.parallel.DistributedDataParallel):
    #     s_code = s_model.module.encode(s_x_v).detach()
    # else:
    #     s_code = s_model.encode(s_x_v).detach()

    # if isinstance(t_model, torch.nn.parallel.DistributedDataParallel):
    #     t_code = t_model.module.encode(t_x_v).detach()
    # else:
    #     t_code = t_model.encode(t_x_v).detach()  

    loss = torch.mean(critic(t_code)) - torch.mean(critic(s_code))

    if gp is not None:
        gradient_penalty = compute_gradient_penalty(critic,
                                                    real_samples=s_code,
                                                    fake_samples=t_code,
                                                    device=device)
        loss = loss + gp * gradient_penalty

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if clip is not None:
        for p in critic.parameters():
            p.data.clamp_(-clip, clip)

    if scheduler is not None:
        scheduler.step()

    history['critic_loss'].append(loss.cpu().detach().item())

    return history


def gan_gen_train_step(critic, s_model, t_model, s_batch, t_batch, 
                       optimizer, alpha, history, scheduler=None):

    critic.eval()
    s_model.train()
    t_model.train()

    for param in critic.parameters():
        param.requires_grad = False

    s_x_v, s_x_i = s_batch[0], s_batch[1]
    s_x = [s_x_v, s_x_i]

    t_x_v, t_x_i = t_batch[0], t_batch[1]
    t_x = [t_x_v, t_x_i]

    t_code = t_model.encode(t_x_v)

    # if isinstance(t_model, torch.nn.parallel.DistributedDataParallel):
    #     t_code = t_model.module.encode(t_x_v)
    # else:
    #     t_code = t_model.encode(t_x_v)     

    gen_loss = -torch.mean(critic(t_code))

    s_loss_dict = s_model.loss_function(*s_model(s_x))
    t_loss_dict = t_model.loss_function(*t_model(t_x))

    # if isinstance(s_model, torch.nn.parallel.DistributedDataParallel):
    #     s_loss_dict = s_model.module.loss_function(*s_model(s_x))
    # else:
    #     s_loss_dict = s_model.loss_function(*s_model(s_x))

    # if isinstance(t_model, torch.nn.parallel.DistributedDataParallel):
    #     t_loss_dict = t_model.module.loss_function(*t_model(t_x))
    # else:
    #     t_loss_dict = t_model.loss_function(*t_model(t_x)) 

    recons_loss = s_loss_dict['loss'] + t_loss_dict['loss']
    loss = recons_loss + alpha * gen_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    for param in critic.parameters():
        param.requires_grad = True

    if scheduler is not None:
        scheduler.step()

    loss_dict = {k: v.cpu().detach().item() + t_loss_dict[k].cpu().detach().item() for k, v in s_loss_dict.items()}

    for k, v in loss_dict.items():
        history[k].append(v)
        
    history['gan_loss'].append(gen_loss.cpu().detach().item())

    return history


def model_save_check(history, metric_name, tolerance_count=5, reset_count=1):

    save_flag, stop_flag = False, False
    if 'best_index' not in history:
        history['best_index'] = 0

    if metric_name.endswith('loss'):
        if history[metric_name][-1] <= history[metric_name][history['best_index']]:
            save_flag = True
            history['best_index'] = len(history[metric_name]) - 1
    else:
        if history[metric_name][-1] >= history[metric_name][history['best_index']]:
            save_flag = True
            history['best_index'] = len(history[metric_name]) - 1

    if len(history[metric_name]) - history['best_index'] > tolerance_count * reset_count and history['best_index'] > 0:
        stop_flag = True

    # print({"best_index": history['best_index'], "metric":history[metric_name][history['best_index']]})
    return save_flag, stop_flag


def eval_epoch(model, data_loader, device, history):

    model.eval()
    avg_loss_dict = defaultdict(float)
    for batch_v, batch_i in data_loader:
        x_batch = [batch_v[0].to(device), batch_i[0].to(device)]
        with torch.no_grad():
            loss_dict = model.loss_function(*(model(x_batch)))
            for k, v in loss_dict.items():
                avg_loss_dict[k] += v.cpu().detach().item() / len(data_loader)

    for k, v in avg_loss_dict.items():
        history[k].append(v)

    return history


def eval_epoch_ddp(model, data_loader, device, history, rank):
    if rank == 0:  # 仅主进程执行测试
        model.eval()
        avg_loss_dict = defaultdict(float)
        for batch_v, batch_i in data_loader:
            x_batch = [batch_v[0].to(device), batch_i[0].to(device)]
            with torch.no_grad():
                loss_dict = model.module.loss_function(*(model(x_batch)))
                for k, v in loss_dict.items():
                    avg_loss_dict[k] += v.cpu().detach().item() / len(data_loader)

        for k, v in avg_loss_dict.items():
            history[k].append(v)

        return history


# fine-tuning step
def evaluate_epoch(classifier, dataloader, device, history, flag, param_str, args):
    y_truths = np.array([])
    y_preds = np.array([])
    classifier.eval()

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        with torch.no_grad():
            y_truths = np.concatenate([y_truths, y_batch.cpu().detach().numpy().ravel()])
            y_pred = torch.sigmoid(classifier(x_batch)).detach()  # torch.sigmoid()
            y_preds = np.concatenate([y_preds, y_pred.cpu().detach().numpy().ravel()])

    # save scores
    if flag == 'val':
        save_path = os.path.join(args.logs_path + args.sc_data, args.drug)
    else:
        save_path = os.path.join(args.logs_path + args.sc_data, args.drug)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        results_df = pd.DataFrame({'y_truths': y_truths, 'y_preds': y_preds})
        results_df.to_csv(save_path + '/' + flag +  '_' + param_str +  '_predict_scores.csv', index=None)

    threshold = 0.5
    binary_preds = (y_preds >= threshold).astype(int)
    
    history['auroc'].append(metrics.roc_auc_score(y_truths, y_preds))
    history['auprc'].append(metrics.average_precision_score(y_truths, y_preds))
    history['acc'].append(metrics.accuracy_score(y_truths, binary_preds))
    history['f1'].append(metrics.f1_score(y_truths, binary_preds, average='weighted'))
    history['precision'].append(metrics.precision_score(y_truths, binary_preds, zero_division=0))
    history['recall'].append(metrics.recall_score(y_truths, binary_preds))
    history['bce_loss'].append(metrics.log_loss(y_truths, y_preds))

    results = {"loss": history['bce_loss'][-1], "auroc": history['auroc'][-1], "auprc": history['auprc'][-1], 
            "acc": history['acc'][-1], "f1": history['f1'][-1], "precision": history['precision'][-1], "recall": history['recall'][-1]}

    return history, results


def evaluate_patient_epoch(classifier, dataloader, device, history, flag, param_str, args):
    data, label = dataloader
    # y_truths = np.array([])
    y_preds = np.array([])
    classifier.eval()

    all_batches = []

    for x_batch in data:
        x_batch = x_batch[0].to(device)
        all_batches.append(x_batch)

        with torch.no_grad():
            y_pred = torch.sigmoid(classifier(x_batch).detach())
            y_preds = np.concatenate([y_preds, y_pred.cpu().detach().numpy().ravel()])

    merged_data = torch.cat(all_batches, dim=0)
    baseline = torch.zeros_like(merged_data)
    ig = IntegratedGradients(classifier)
    attributions, delta = ig.attribute(merged_data, baselines=baseline, return_convergence_delta=True)
    contibutions = attributions.view(attributions.shape[0], -1)[:, 0:512*12].cpu().numpy()

    import pickle
    output_path = './Bortezomib_attributions_processed.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(contibutions, f)

    label.index = label.index.astype(int)
    label['patient_id'] = label['patient_id'].astype(str)
    y_preds = pd.DataFrame(y_preds, columns=['predict_score'])

    patient_score = pd.concat([y_preds, label], axis=1)

    # ++++++++++++++++++++++++++++++++++++++
    # subclone_proportion = pd.read_csv('./data/scdata/sc_patients/GSE189460/GSE189460_subclone_proportion.csv')
    # merged_df = pd.concat([patient_score, subclone_proportion], axis=1)
    # merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
    # merged_df['weight_score'] = merged_df['predict_score'] * merged_df['proportion']
    patient_score_list = list(patient_score.groupby('patient_id')) # patient_score  merged_df
    # save scores 
    save_path = os.path.join(args.logs_path + args.sc_data, args.drug)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    patient_score.to_csv(save_path + '/' + flag +  '_' + param_str +  '_predict_scores_clones.csv', index=None) # patient_score

    patient_score = {}
    for patient_id, score in patient_score_list:
        patient_score[patient_id] = score['predict_score'].min() # weight_score predict_score z_score

    # +++++++++++++++++++++++++++++++++++++++
        
    patient_score_df = pd.DataFrame([patient_score]).T
    patient_score_df.columns = ["predicted_score"]
    patient_score_df['patient_id'] = patient_score_df.index.tolist()
    
    # lung dataset
    # patient_score_df['numeric_part'] = patient_score_df['patient_id'].apply(lambda x: int(re.search(r'(\d+)', x).group()))
    # patient_score_df = patient_score_df.sort_values(by='numeric_part')
    # patient_score_df = patient_score_df.drop(columns=['numeric_part'])

    reponse_file = os.path.join('data/scdata/sc_patients/', args.sc_data, args.sc_data + '_response.csv')
    # y_truths = pd.read_csv(reponse_file)['Label'].values

    patient_info = pd.read_csv(reponse_file, index_col=2)  # index_col=2, 0
    result = pd.merge(patient_score_df, patient_info, left_index=True, right_index=True)
    result.to_csv(save_path + '/' + flag +  '_' + param_str +  '_predict_scores.csv', index=None)

    y_truths = result['label'].tolist()
    predicted_score = result['predicted_score'].tolist()

    threshold = 0.5
    binary_preds = (result['predicted_score'] >= threshold).astype(int).tolist()

    history['auroc'].append(metrics.roc_auc_score(y_truths, predicted_score))
    history['auprc'].append(metrics.average_precision_score(y_truths, predicted_score))
    history['acc'].append(metrics.accuracy_score(y_truths, binary_preds))
    history['f1'].append(metrics.f1_score(y_truths, binary_preds, average='weighted'))

    results = {"auroc": history['auroc'][-1], "auprc": history['auprc'][-1], "acc": history['acc'][-1], "f1": history['f1'][-1]}

    return history, results


def predict_target_classification(classifier, test_df, device):
    y_preds = np.array([])
    classifier.eval()

    for df in [test_df[i:i+64] for i in range(0,test_df.shape[0],64)]:
        x_batch = torch.from_numpy(df.values.astype('float32')).to(device)
        with torch.no_grad():
            y_pred = torch.sigmoid(classifier(x_batch)).detach()
            y_preds = np.concatenate([y_preds, y_pred.cpu().detach().numpy().ravel()])

    output_df = pd.DataFrame(y_preds,index=test_df.index, columns=['score'])

    return output_df