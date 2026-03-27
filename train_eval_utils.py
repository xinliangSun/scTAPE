import os
import torch
import pandas as pd
import numpy as np
import torch.autograd as autograd
import sklearn.metrics as metrics

from collections import defaultdict


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

    loss = s_loss_dict['loss'] + t_loss_dict['loss']

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_dict = {k: v.cpu().detach().item() + t_loss_dict[k].cpu().detach().item() for k, v in s_loss_dict.items()}
    
    s_recons_loss, s_diff_loss = s_loss_dict['recons_loss'].cpu().detach().item(), s_loss_dict['diff_loss'].cpu().detach().item()
    t_recons_loss, t_diff_loss = t_loss_dict['recons_loss'].cpu().detach().item(), t_loss_dict['diff_loss'].cpu().detach().item()

    loss_dict.update({"s_recons_loss": s_recons_loss, "s_diff_loss": s_diff_loss, 
                      "t_recons_loss": t_recons_loss, "t_diff_loss": t_diff_loss})
 
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

    gen_loss = -torch.mean(critic(t_code))

    s_loss_dict = s_model.loss_function(*s_model(s_x))
    t_loss_dict = t_model.loss_function(*t_model(t_x))

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
            y_pred = torch.sigmoid(classifier(x_batch)).detach()
            y_preds = np.concatenate([y_preds, y_pred.cpu().detach().numpy().ravel()])

    # save scores
    if flag == 'val':
        save_path = os.path.join(args.logs_path + args.sc_data, args.drug)
    else:
        save_path = os.path.join(args.logs_path + args.sc_data, args.drug)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    
    history['auroc'].append(metrics.roc_auc_score(y_truths, y_preds))
    history['auprc'].append(metrics.average_precision_score(y_truths, y_preds))

    history['bce_loss'].append(metrics.log_loss(y_truths, y_preds))

    results = {"loss": history['bce_loss'][-1], "auroc": history['auroc'][-1], "auprc": history['auprc'][-1]}

    return history, results


def evaluate_patient_epoch(classifier, dataloader, device, history, args):
    data, label = dataloader
    y_preds = np.array([])
    classifier.eval()

    all_batches = []

    for x_batch in data:
        x_batch = x_batch[0].to(device)
        all_batches.append(x_batch)

        with torch.no_grad():
            y_pred = torch.sigmoid(classifier(x_batch).detach())
            y_preds = np.concatenate([y_preds, y_pred.cpu().detach().numpy().ravel()])


    label.index = label.index.astype(int)
    label['patient_id'] = label['patient_id'].astype(str)
    y_preds = pd.DataFrame(y_preds, columns=['predict_score'])

    patient_score = pd.concat([y_preds, label], axis=1)

    patient_score_list = list(patient_score.groupby('patient_id'))
    # save scores 
    save_path = os.path.join(args.logs_path + args.sc_data, args.drug)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    patient_score = {}
    for patient_id, score in patient_score_list:
        patient_score[patient_id] = score['predict_score'].min()
        
    patient_score_df = pd.DataFrame([patient_score]).T
    patient_score_df.columns = ["predicted_score"]
    patient_score_df['patient_id'] = patient_score_df.index.tolist()

    reponse_file = os.path.join('data/scdata/sc_patients/', args.sc_data, args.sc_data + '_response.csv')
    patient_info = pd.read_csv(reponse_file, index_col=2)  # index_col=2, 0
    result = pd.merge(patient_score_df, patient_info, left_index=True, right_index=True)

    y_truths = result['label'].tolist()
    predicted_score = result['predicted_score'].tolist()

    history['auroc'].append(metrics.roc_auc_score(y_truths, predicted_score))
    history['auprc'].append(metrics.average_precision_score(y_truths, predicted_score))

    results = {"auroc": history['auroc'][-1], "auprc": history['auprc'][-1]}

    return history, results