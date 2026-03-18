import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import os
from shutil import copyfile
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from util import save_checkpoint, save_reg_checkpoint, my_eval_with_dynamic_thresh
from finetune_model import ft_12lead_ECGFounder, ft_1lead_ECGFounder, ft_12lead_UKB
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dataset import UKB_Dataset
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score, confusion_matrix, balanced_accuracy_score, roc_curve

def find_optimal_threshold(gt, pred):
    n_task = gt.shape[1]
    optimal_thresholds = []

    for i in range(n_task):
        best_ba = -1  
        best_thresh = 0.5  
        for thresh in np.linspace(0.01, 0.99, 99):  
            pred_labels = (pred[:, i] > thresh).astype(int)
            ba = balanced_accuracy_score(gt[:, i], pred_labels)  
            if ba > best_ba:
                best_ba = ba
                best_thresh = thresh
        optimal_thresholds.append(best_thresh)

    return optimal_thresholds

def eval_model(gt, pred, thresholds):
    """
    Evaluates the model with dynamically adjusted thresholds for each task.

    Args:
        gt: Ground truth labels (numpy array)
        pred: Prediction probabilities (numpy array)

    Returns:
        - Overall mean of the metrics across all tasks
        - Per-metric mean across all tasks (as a list)
        - All metrics per task in a columnar format
    """
    optimal_thresholds = thresholds
    n_task = gt.shape[1]
    rocaucs = []
    sensitivities = []
    specificities = []
    f1 = []
    auprcs = []  # Step 2: Initialize list for AUPRC

    for i in range(n_task):
        tmp_gt = np.nan_to_num(gt[:, i], nan=0)
        tmp_pred = np.nan_to_num(pred[:, i], nan=0)

        # ROC-AUC
        try:
            rocaucs.append(roc_auc_score(tmp_gt, tmp_pred))
        except:
            rocaucs.append(0.0)

        # AUPRC  # Step 3: Calculate AUPRC
        try:
            auprc = average_precision_score(tmp_gt, tmp_pred)
            auprcs.append(auprc)
        except:
            auprcs.append(0.0)

        # Sensitivity and Specificity
        pred_labels = (tmp_pred > optimal_thresholds[i]).astype(int)
        # pred_labels = (tmp_pred > 0.5).astype(int)
        cm = confusion_matrix(tmp_gt, pred_labels).ravel()
        
        # Handle different sizes of confusion matrix
        if len(cm) == 1:
            # Only one class present in predictions
            if pred_labels.sum() == 0:  # Only negative class predicted
                tn, fp, fn, tp = cm[0], 0, 0, 0
            else:                       # Only positive class predicted
                tn, fp, fn, tp = 0, 0, 0, cm[0]
        else:
            tn, fp, fn, tp = cm

        # Calculate Sensitivity (True Positive Rate)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        sensitivities.append(sensitivity)
        
        # Calculate Specificity (True Negative Rate)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificities.append(specificity)

        f1s = f1_score(tmp_gt, pred_labels)
        f1.append(f1s)
    
    # Convert lists to numpy arrays
    rocaucs = np.array(rocaucs)
    sensitivities = np.array(sensitivities)
    specificities = np.array(specificities)
    f1 = np.array(f1)
    auprcs = np.array(auprcs)  # Step 4: Compute mean AUPRC

    # Calculate means for each metric
    mean_rocauc = np.mean(rocaucs)
    mean_auprc = np.mean(auprcs)  # Step 4: Compute mean AUPRC

    # Step 5: Update return statement
    return mean_rocauc, rocaucs, sensitivities, specificities, f1, auprcs

def data_metrics(loader, step="Validation", predefined_thresh=None):
    progress_bar = tqdm(loader, desc=step, leave=False)
    all_input_labels = []
    all_pred_prob = []
    with torch.no_grad():
        for batch_idx, data_batch in enumerate(progress_bar):
            batch_input_x, batch_input_y = tuple(t.to(device) for t in data_batch)
            logits = model(batch_input_x)
            pred = torch.sigmoid(logits)
            all_pred_prob.append(pred.cpu().data.numpy())
            all_input_labels.append(batch_input_y.cpu().data.numpy())
    all_pred_prob = np.concatenate(all_pred_prob)
    all_input_labels = np.concatenate(all_input_labels)
    all_input_labels = np.array(all_input_labels)
    thresh = find_optimal_threshold(all_input_labels, all_pred_prob) if not predefined_thresh else predefined_thresh
    mean_rocauc, rocaucs, sensitivities, specificities, f1, auprcs = eval_model(all_input_labels, all_pred_prob, thresh)
    return mean_rocauc, rocaucs, sensitivities, specificities, f1, auprcs, thresh
    

num_lead = 12 # 12-lead ECG or 1-lead ECG 

gpu_id = 0
batch_size = 256
lr = 1e-4
weight_decay = 1e-5
early_stop_lr = 1e-5
Epochs = 8
df_label_path = 'cm_var_labels_ecgfounder.tsv'
ecg_path = './' #'/mnt/project/Bulk/Electrocardiogram/Resting/'
tasks = ['has_cm_var']
saved_dir = './'

device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')

n_classes = len(tasks)

ECGdataset = UKB_Dataset
pth = '/app/12_lead_ECGFounder.pth'
model = ft_12lead_UKB(device, pth, n_classes,linear_prob=True)

df_label = pd.read_csv(df_label_path, sep="\t")
# Splitting the dataset into train, validation, and test sets

train_df, test_df = train_test_split(df_label, test_size=0.2, shuffle=False)
val_df, test_df = train_test_split(test_df, test_size=0.5, shuffle=False)

train_dataset = ECGdataset(data_dir=ecg_path,labels_df=train_df)
val_dataset = ECGdataset(data_dir=ecg_path,labels_df=val_df)
test_dataset = ECGdataset(data_dir=ecg_path,labels_df=test_df)

# Example DataLoader usage
trainloader = DataLoader(train_dataset, batch_size=batch_size,num_workers=0, shuffle=True)
valloader = DataLoader(val_dataset, batch_size=batch_size,num_workers=0, shuffle=False)
testloader = DataLoader(test_dataset, batch_size=batch_size,num_workers=0, shuffle=False)

# linear classificaion  ->  linear_prob=True
# full fine-tuning  ->  linear_prob=False

criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, mode='max', verbose=True)

### train model
best_val_auroc = 0.
step = 0
current_lr = lr
all_res = []
pos_neg_counts = {}
total_steps_per_epoch = len(trainloader)
eval_steps = total_steps_per_epoch

model.eval()
model.dense.train()
for epoch in range(Epochs):
    ### train
    for batch in tqdm(trainloader,desc='Training'):
        input_x, input_y = tuple(t.to(device) for t in batch)
        outputs = model(input_x)
        loss = criterion(outputs, input_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1

        if step % eval_steps == 0: # At the end of each epoch, evaluate performance
            model.eval()
            
            # val accuracy
            res_val, res_val_auroc, res_val_sens, res_val_spec, res_val_f1, res_val_auprc, val_thresh = data_metrics(valloader, step="Validation")
            val_auroc = res_val

            # test accuracy
            res_test, res_test_auroc, res_test_sens, res_test_spec, res_test_f1, res_test_auprc, _ = data_metrics(testloader, step="Test", predefined_thresh=val_thresh)
            test_auroc = res_test
        
            # train accuracy
            train_labels = input_y.cpu().data.numpy()
            train_preds = torch.sigmoid(outputs).cpu().data.numpy()
            res_train, res_train_auroc, res_train_sens, res_train_spec, res_train_f1, res_train_auprc = eval_model(train_labels, train_preds, val_thresh)
            train_auroc = res_train
            
            print(f'Epoch {epoch} step {step}, train: {train_auroc} val: {val_auroc} threshold: {val_thresh[0]}')

            ### save model and res
            is_best = bool(val_auroc > best_val_auroc)
            if is_best:
                best_val_auroc = val_auroc
                print('==> Saving a new val best!')
                save_checkpoint({
                    'epoch': epoch,
                    'step': step,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'val_auroc': val_auroc,
                }, saved_dir)
                
            current_lr = optimizer.param_groups[0]['lr']

            for i, task in enumerate(tasks):
                pos_count = test_df[task].sum()
                neg_count = len(test_df) - pos_count
                all_res.append([task, res_val_auroc[i], res_val_sens[i], res_val_spec[i], res_val_f1[i], res_val_auprc[i], val_thresh[i], 
                                res_test_auroc[i], res_test_sens[i], res_test_spec[i], res_test_f1[i], res_test_auprc[i], pos_count, neg_count])

            columns = ['Field_ID', 'val_AUROC', 'val_sensitivity', 'val_specificity', 'val_f1', 'val_auprc', 'val_thresh', 
                       'test_AUROC', 'test_sensitivity', 'test_specificity', 'test_f1', 'test_auprc', 'pos_num', 'neg_num']
            
            df = pd.DataFrame(all_res, columns=columns)

            df.to_csv(os.path.join(saved_dir, f'res.csv'), index=False, float_format='%.5f')
            
            scheduler.step(val_auroc)
            ### early stop
            current_lr = optimizer.param_groups[0]['lr']
            if current_lr < early_stop_lr:
                print("Early stop")
                exit()
                
            model.dense.train() # set back to train
            
