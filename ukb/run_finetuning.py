import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import xmltodict

from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score, confusion_matrix, balanced_accuracy_score, roc_curve
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


from util import filter_bandpass, save_checkpoint, find_optimal_threshold, find_optimal_threshold_f2
from net1d import Net1D

class UKB_Dataset(Dataset):
    def __init__(self, data_dir, labels_df, transform=None):
        """
        Args:
            data_dir (str): Directory path containing the numpy data files. -> "/mnt/project/Bulk/Electrocardiogram/Resting/"
            labels_df (DataFrame): DataFrame containing the annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.labels_df = labels_df
        self.transform = transform
        self.data_dir = data_dir
        self.input_leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        self.fs = 5000 # length of data, 5000 = 500Hz * 10s

    def __len__(self):
        return len(self.labels_df)

    def z_score_normalization(self,signal):
        return (signal - np.mean(signal)) / (np.std(signal) +1e-8) 

    def check_nan_in_array(self, arr):
        contains_nan = np.isnan(arr).any()
        return contains_nan
    
    def extract_waveform_from_xml(self, xml_path):
        """
        Extract ECG waveform from xml and save as numpy array with shape=[5000,12,1] ([time, leads, 1]).
        The voltage unit should be in 1 mv/unit and the sampling rate should be 500/second (total 10 second).
        The leads should be ordered as follow I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6.
        """
        with open(xml_path, 'rb') as fd:
            xml_dict = xmltodict.parse(fd.read().decode('utf8'))
        
        ukb_pt_id = xml_path.split("/")[-1].split(".")[0] #xmlfile.split("/")[-1].split("_")[0]
        xml_dict = xml_dict['CardiologyXML']
        
        #need to instantiate leads in the proper order for the model
        lead_order = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        lead_data =  dict.fromkeys(lead_order)
        for lead_num, lead in enumerate(xml_dict['StripData']['WaveformData']):
            lead_id = lead["@lead"]
            waveform_data = np.array([np.int16(x.strip()) for x in xml_dict['StripData']['WaveformData'][lead_num]["#text"].split(",")])
            lead_data[lead_id] = waveform_data

        # now construct and reshape the array
        # converting the dictionary to an np.array
        temp = []
        for key,value in lead_data.items():
            temp.append(value)

        # Shape is [leads, time]
        ecg_array = np.array(temp).T
        ecg_array = np.expand_dims(ecg_array, axis=0)
        
        # Here is a check to make sure all the model inputs are the right shape
        assert ecg_array.shape == (1, 5000, 12), "ecg_array is shape {} not (1, 5000, 12)".format(ecg_array.shape)
        
        return ecg_array

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        wave_file_path = str(self.labels_df.iloc[idx]["npy_path"])
        labels = self.labels_df.iloc[idx, -1]
        labels = labels.astype(np.float32)
        data = np.load(wave_file_path) #self.extract_waveform_from_xml(self.data_dir + xml_file_name)
        data = np.nan_to_num(data, nan=0)
        data = data.squeeze(0)
        data = np.transpose(data,  (1, 0))
        data = filter_bandpass(data, 500) 
        signal = self.z_score_normalization(data)
        signal = torch.FloatTensor(signal)

        # Convert to torch tensors
        labels = torch.tensor(labels, dtype=torch.float)
        if labels.dim() == 0:  
            labels = labels.unsqueeze(0)
        elif labels.dim() == 1:  
            labels = labels.unsqueeze(1)
        return signal, labels     

def ft_UKB(device, pth, n_classes, linear_prob=False, dropout=False):
    model = Net1D(
        in_channels=12, 
        base_filters=64, 
        ratio=1, 
        filter_list=[64,160,160,400,400,1024,1024],    
        m_blocks_list=[2,2,2,3,3,4,4], 
        kernel_size=16, 
        stride=2, 
        groups_width=16,
        verbose=False, 
        use_bn=False,
        use_do=dropout,
        n_classes=n_classes)

    checkpoint = torch.load(pth, map_location=device)
    state_dict = checkpoint['state_dict']

    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('dense.')} 

    model.load_state_dict(state_dict, strict=False)

    # set shape of classification head
    model.dense = nn.Sequential(nn.Linear(model.dense.in_features, model.dense.in_features//2), nn.Linear(model.dense.in_features//2, model.dense.in_features//4), nn.Linear(model.dense.in_features//4, n_classes)).to(device)

    # freezing model
    if linear_prob == True: 
        for name, param in model.named_parameters():
            if 'dense' not in name:  # no freezing last layer
                param.requires_grad = False

    model.to(device)

    return model

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
    thresh = 0.5 #find_optimal_threshold_f2(all_input_labels, all_pred_prob) if not predefined_thresh else predefined_thresh
    mean_rocauc, rocaucs, sensitivities, specificities, f1, auprcs = eval_model(all_input_labels, all_pred_prob, thresh)
    return mean_rocauc, rocaucs, sensitivities, specificities, f1, auprcs, thresh
    

gpu_id = 0
batch_size = 256
lr = 1e-4
weight_decay = 1e-5
early_stop_lr = 1e-5
epochs = 8
freeze_conv_layers = True

df_label_path = 'cm_var_labels_ecgfounder.tsv'
tasks = ['has_cm_var']
n_classes = len(tasks)

ecg_path = './' #'/mnt/project/Bulk/Electrocardiogram/Resting/'
out_dir = './'

device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')

checkpoint_path = '/app/12_lead_ECGFounder.pth'
model = ft_UKB(device, checkpoint_path, n_classes,linear_prob=freeze_conv_layers, dropout=True)
# linear classificaion  ->  linear_prob=True
# full fine-tuning  ->  linear_prob=False

df_label = pd.read_csv(df_label_path, sep="\t")

# Splitting the dataset into train, validation, and test sets
train_df, test_df = train_test_split(df_label, test_size=0.2, shuffle=False)
val_df, test_df = train_test_split(test_df, test_size=0.5, shuffle=False)

train_dataset = UKB_Dataset(data_dir=ecg_path,labels_df=train_df)
val_dataset = UKB_Dataset(data_dir=ecg_path,labels_df=val_df)
test_dataset = UKB_Dataset(data_dir=ecg_path,labels_df=test_df)

trainloader = DataLoader(train_dataset, batch_size=batch_size,num_workers=0, shuffle=True)
valloader = DataLoader(val_dataset, batch_size=batch_size,num_workers=0, shuffle=False)
testloader = DataLoader(test_dataset, batch_size=batch_size,num_workers=0, shuffle=False)

case_ctrl_ratio = torch.tensor([df_label['has_cm_var'].sum() / (len(df_label) - df_label['has_cm_var'].sum())])
criterion = nn.BCEWithLogitsLoss(pos_weight=case_ctrl_ratio)
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

# Disable dropout for conv layer
if freeze_conv_layers:
    model.eval()
    model.dense.train()
else:
    model.train()

for epoch in range(epochs):
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
            
            print(f'Epoch {epoch} step {step}, train: {train_auroc} val: {val_auroc} test: {test_auroc} threshold: {val_thresh[0]}')

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
                }, out_dir)
                
            current_lr = optimizer.param_groups[0]['lr']

            for i, task in enumerate(tasks):
                pos_count = test_df[task].sum()
                neg_count = len(test_df) - pos_count
                all_res.append([task, res_val_auroc[i], res_val_sens[i], res_val_spec[i], res_val_f1[i], res_val_auprc[i], val_thresh[i], 
                                res_test_auroc[i], res_test_sens[i], res_test_spec[i], res_test_f1[i], res_test_auprc[i], pos_count, neg_count])

            columns = ['Field_ID', 'val_AUROC', 'val_sensitivity', 'val_specificity', 'val_f1', 'val_auprc', 'val_thresh', 
                       'test_AUROC', 'test_sensitivity', 'test_specificity', 'test_f1', 'test_auprc', 'pos_num', 'neg_num']
            
            results_df = pd.DataFrame(all_res, columns=columns)

            results_df.to_csv(os.path.join(out_dir, f'results.csv'), index=False, float_format='%.5f')
            
            scheduler.step(val_auroc)
            ### early stop
            current_lr = optimizer.param_groups[0]['lr']
            if current_lr < early_stop_lr:
                print("Early stop")
                exit()
            
            if freeze_conv_layers:
                model.dense.train() # set back to train
            else:
                model.train()
            
