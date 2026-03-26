import math
import os
import shutil
import h5py
import numpy as np
from time import gmtime, strftime
from matplotlib import pyplot as plt
from collections import Counter, OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score, confusion_matrix, balanced_accuracy_score, roc_curve, fbeta_score
from sklearn.utils import resample
from sklearn.metrics import average_precision_score
from scipy.signal import medfilt, iirnotch, filtfilt, butter, resample
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
warnings.filterwarnings("ignore", message="A single label was found in 'y_true' and 'y_pred'")

from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def filter_bandpass(signal, fs):
    """
    Bandpass filter
    :param signal: 2D numpy array of shape (channels, time)
    :param fs: sampling frequency
    :return: filtered signal
    """
    # Remove power-line interference
    b, a = iirnotch(50, 30, fs)
    filtered_signal = np.zeros_like(signal)
    for c in range(signal.shape[0]):
        filtered_signal[c] = filtfilt(b, a, signal[c])

    # Simple bandpass filter
    b, a = butter(N=4, Wn=[0.67, 40], btype='bandpass', fs=fs)
    for c in range(signal.shape[0]):
        filtered_signal[c] = filtfilt(b, a, filtered_signal[c])

    # Remove baseline wander
    baseline = np.zeros_like(filtered_signal)
    for c in range(filtered_signal.shape[0]):
        kernel_size = int(0.4 * fs) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure kernel size is odd
        baseline[c] = medfilt(filtered_signal[c], kernel_size=kernel_size)
    filter_ecg = filtered_signal - baseline

    return filter_ecg


def bootstrap_ci(
    gt, 
    pred, 
    metric, 
    n_bootstrap=1000, 
    ci_percentile=95
):
    """
    Calculates confidence intervals for a given metric using bootstrapping.

    Args:
        gt: Ground truth labels (numpy array), shape: [N,]
        pred: Prediction probabilities (numpy array), shape: [N,]
        metric: One of ['roc_auc', 'auprc', 'ppv', 'npv', 'sensitivity', 'specificity']
        n_bootstrap: Number of bootstrap samples to generate
        ci_percentile: Percentile for the confidence intervals

    Returns:
        (lower_bound, upper_bound): tuple of floats
    """
    from sklearn.metrics import (roc_auc_score, average_precision_score, 
                                 confusion_matrix, f1_score)

    n = len(gt)
    metrics_list = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(range(n), size=n, replace=True)
        gt_resampled = gt[indices]
        pred_resampled = pred[indices]

        if metric == 'roc_auc':
            try:
                val = roc_auc_score(gt_resampled, pred_resampled)
            except ValueError:
                val = 0.0

        elif metric == 'auprc':
            try:
                val = average_precision_score(gt_resampled, pred_resampled)
            except ValueError:
                val = 0.0

        else:
            pred_labels = (pred_resampled > 0.5).astype(int)
            cm = confusion_matrix(gt_resampled, pred_labels).ravel()
            if len(cm) == 1:
                if pred_labels.sum() == 0:  
                    tn, fp, fn, tp = cm[0], 0, 0, 0
                else:
                    tn, fp, fn, tp = 0, 0, 0, cm[0]
            else:
                tn, fp, fn, tp = cm

            if metric == 'sensitivity':   # recall
                val = tp / (tp + fn) if (tp + fn) > 0 else 0
            elif metric == 'specificity':
                val = tn / (tn + fp) if (tn + fp) > 0 else 0
            elif metric == 'ppv':  # precision
                val = tp / (tp + fp) if (tp + fp) > 0 else 0
            elif metric == 'npv':
                val = tn / (tn + fn) if (tn + fn) > 0 else 0
            else:
                val = 0.0

        metrics_list.append(val)

    alpha = (100 - ci_percentile) / 2
    lower_bound = np.percentile(metrics_list, alpha)
    upper_bound = np.percentile(metrics_list, 100 - alpha)
    return (lower_bound, upper_bound)

def quantile_accuracy(y_true, y_pred, quantiles):
    """
    :param y_true: 
    :param y_pred: 
    :param quantiles: e.g. [0.25, 0.5, 0.75]
    """
    quantile_errors = {}
    for q in quantiles:
        pred_quantile = np.percentile(y_pred, q * 100)
        true_quantile = np.percentile(y_true, q * 100)
        # calculate error
        quantile_errors[q] = abs(pred_quantile - true_quantile)
    
    return quantile_errors

def find_optimal_thresholds(gt, pred):
    optimal_thresholds = []
    for i in range(gt.shape[1]):
        fpr, tpr, thresholds = roc_curve(gt[:, i], pred[:, i])
        optimal_idx = np.argmax(tpr - fpr)  
        optimal_thresholds.append(thresholds[optimal_idx])
    return np.array(optimal_thresholds)

def get_time_str():
    return strftime("%Y%m%d_%H%M%S", gmtime())

def print_and_log(log_name, my_str):
    out = '{}|{}'.format(get_time_str(), my_str)
    print(out)
    with open(log_name, 'a') as f_log:
        print(out, file=f_log)

def save_checkpoint(state, path):
    filename = 'checkpoint_{0}_{1:.4f}.pth'.format(state['step'], state['val_auroc'])
    filename = os.path.join(path, filename)
    torch.save(state, filename)

def save_reg_checkpoint(state, path):
    filename = 'checkpoint_{0}_{1:.4f}.pth'.format(state['step'], state['mae'])
    filename = os.path.join(path, filename)
    torch.save(state, filename)


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

def find_optimal_threshold_f2(gt, pred):
    n_task = gt.shape[1]
    optimal_thresholds = []

    for i in range(n_task):
        best_f2 = -1  
        best_thresh = 0.5  
        for thresh in np.linspace(0.01, 0.99, 99):  
            pred_labels = (pred[:, i] > thresh).astype(int)
            f2score = fbeta_score(gt[:, i], pred_labels, beta=2)  
            if f2score > best_f2:
                best_f2 = f2score
                best_thresh = thresh
        optimal_thresholds.append(best_thresh)

    return optimal_thresholds