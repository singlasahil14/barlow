import os
os.environ['NOTEBOOK_MODE'] = '1'
import cv2
import math

import sys
import torch as ch
import torch.nn.functional as F
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif

from scipy import stats
from collections import defaultdict
import matplotlib.pyplot as plt
from robustness import model_utils
from robustness import datasets as dataset_utils
from robustness.tools.label_maps import CLASS_DICT

# Compute accuracy using logits and labels
def compute_accuracy(logits, labels):
    preds = np.argmax(logits, axis=1)
    return compute_preds_accuracy(preds, labels)

# Compute accuracy using predictions and labels
def compute_preds_accuracy(preds, labels):
    num_correct = np.sum(preds==labels)
    num_total = labels.size
    acc = num_correct/num_total
    return acc

# Compute failure labels (i.e whether prediction was correct or not) 
# using logits and labels
def compute_failure_labels(logits, labels):
    preds = np.argmax(logits, axis=1)
    success = (preds==labels)
    failure = np.logical_not(success)
    return failure

# Compute failure labels (i.e whether prediction was correct or not) 
# using logits and labels
def np_softmax(logits):
    assert logits.ndim==2
    max_logits = np.max(logits, axis=1, keepdims=True)
    logits_n = logits - max_logits
    
    logits_n_exp = np.exp(logits_n)
    logits_n_exp_sum = np.sum(logits_n_exp, axis=1, keepdims=True)
    probs = logits_n_exp/logits_n_exp_sum
    return probs

# Create dictionary that maps a unique element to its count
def create_dict(unique, counts, dtype=int):
    count_dict = defaultdict(dtype)
    for u,c in zip(unique, counts): 
        count_dict[u] = count_dict[u] + c
    return count_dict

# Compute precision and recall using predictions and labels
def compute_precision_recall(preds, labels):
    num_true = np.sum(labels)
    num_preds_true = np.sum(preds)
    num_correct_preds_true = np.sum(np.logical_and(preds, labels))
    
    precision = num_correct_preds_true/num_preds_true
    recall = num_correct_preds_true/num_true
    return precision, recall

# Load features from path 'features_path'
def load_features(model_name, features_path, splitset):
    assert splitset in ['train', 'test']
    if model_name == 'ImageNetNat.pt':
        features = np.load(features_path + 'nonrobust_' + splitset + '_features.npy')
        logits = np.load(features_path + 'nonrobust_' + splitset + '_logits.npy')
        labels = np.load(features_path + splitset + '_labels.npy')
    elif model_name == 'ImageNet_l2_3_0.pt':
        features = np.load(features_path + 'robust_' + splitset + '_features.npy')
        logits = np.load(features_path + 'robust_' + splitset + '_logits.npy')
        labels = np.load(features_path + splitset + '_labels.npy')
    else:
        raise ValueError('Unidentified model name: ' + model_name)
    return features, logits, labels

# Generate dataset using model and data_loader
def generate_dataset(model, data_loader):
    features_all, logits_all, labels_all = [], [], []
    total = 0
    for _, (ims, labels) in enumerate(data_loader):
        ims, labels = ims.cuda(), labels.cuda()
        batch_size = ims.shape[0]
        (logits, features), _ = model(ims, with_latent=True)
        features = features.detach()
        logits = logits.detach()
        total += len(ims)
        features_all.append(features.detach().cpu().numpy())
        logits_all.append(logits.detach().cpu().numpy())
        labels_all.append(labels.cpu().numpy())
        print(total)
        
    features_all = np.concatenate(features_all, axis=0)
    logits_all = np.concatenate(logits_all, axis=0)
    labels_all = np.concatenate(labels_all, axis=0)
    return features_all, logits_all, labels_all

# Generate failure statistics using logits and labels
def failure_statistics(logits, labels):
    preds = np.argmax(logits, axis=1)
    num_classes = logits.shape[1]
    
    pred_failures_arr = np.zeros(num_classes, dtype=np.long)
    label_failures_arr = np.zeros(num_classes, dtype=np.long)
    count_preds_arr = np.zeros(num_classes, dtype=np.long)
    count_labels_arr = np.zeros(num_classes, dtype=np.long)
    for i in range(num_classes):
        pred_failures = np.sum((preds==i) & np.logical_not(preds==labels))
        pred_failures_arr[i] = pred_failures
        label_failures = np.sum((labels==i) & np.logical_not(preds==labels))
        label_failures_arr[i] = label_failures
        count_preds_arr[i] = np.sum(preds==i)
        count_labels_arr[i] = np.sum(labels==i)
        
    dic = {'pred_failures': pred_failures_arr, 
           'label_failures': label_failures_arr,
           'pred_counts': count_preds_arr,
           'label_counts': count_labels_arr}
    return dic

# Print failure statistics for group of images with 
# predicted class index = 'class_idx'
def print_failure_stats_pred(failure_dict, class_idx, class_names):
    pred_failures_arr = failure_dict['pred_failures']
    pred_counts_arr = failure_dict['pred_counts']
    label_counts_arr = failure_dict['label_counts']

    class_name = ', '.join(class_names[class_idx].split(',')[:2])
    num_failures = pred_failures_arr[class_idx]
    num_preds = pred_counts_arr[class_idx]
    num_labels = label_counts_arr[class_idx]

    print('class_idx: {:d}, class_name: {:s}, num_failures: {:d}, num_preds: {:d}, num_labels: {:d}'.
          format(class_idx, class_name, num_failures, num_preds, num_labels))
    return class_name, num_failures, num_preds, num_labels
    
# Print failure statistics for group of images with 
# label class index = 'class_idx'
def print_failure_stats_label(failure_dict, class_idx, class_names):
    label_failures_arr = failure_dict['label_failures']
    pred_counts_arr = failure_dict['pred_counts']
    label_counts_arr = failure_dict['label_counts']

    class_name = ', '.join(class_names[class_idx].split(',')[:2])
    num_failures = label_failures_arr[class_idx]
    num_preds = pred_counts_arr[class_idx]
    num_labels = label_counts_arr[class_idx]
    fraction = num_failures/num_labels

    print('class_idx: {:d}, class_name: {:s}, num_failures: {:d}, num_preds: {:d}, num_labels: {:d}, fraction: {:.4f}'.
          format(class_idx, class_name, num_failures, num_preds, num_labels, fraction))
    return class_name, num_failures, num_preds, num_labels
    
# Return indices where the predicted class is 'class_idx'
def predicted_class_indices(logits, class_idx):
    preds = np.argmax(logits, axis=1)
    indices = np.nonzero(preds==class_idx)[0]
    return indices

# Return features and failure labels for specified indices
def failure_data(indices, features, logits, labels):
    features_indices = features[indices]
    logits_indices = logits[indices]
    preds_indices = np.argmax(logits_indices, axis=1)
    labels_indices = labels[indices]
    
    failure_indices = np.logical_not(preds_indices==labels_indices)    
    return features_indices, failure_indices

# Find indices where predicted or label class (depending on grouping) is class_idx
def failure_indices_class(train_logits, train_labels, class_idx, class_names, 
                          grouping='label', print_stats=True):
    train_failure_dict = failure_statistics(train_logits, train_labels)
    if grouping == 'label':
        if print_stats:
            class_name, num_failures, num_preds, num_labels = print_failure_stats_label(
                train_failure_dict, class_idx, class_names)
        train_indices_class = np.nonzero(train_labels==class_idx)[0]
    elif grouping == 'pred':
        if print_stats:
            class_name, num_failures, num_preds, class_names = print_failure_stats_pred(
                train_failure_dict, class_idx, class_names)
        train_indices_class = predicted_class_indices(train_logits, class_idx)
    return train_indices_class

def mutual_info_select(train_features_class, train_failure_class, num_features=20):
    mi = mutual_info_classif(train_features_class, train_failure_class, 
                             random_state = 0)
    important_features_indices = np.argsort(mi)[-num_features:]
    important_features_values = mi[important_features_indices]
    return important_features_indices, important_features_values

def feature_importance_select(train_features_class, num_features=20):
    fi = np.mean(train_features_class, axis=0)
    important_features_indices = np.argsort(fi)[-num_features:]
    important_features_values = fi[important_features_indices]
    return important_features_indices, important_features_values
    
# Perform feature selection using some prespecified method such as mutual_information
def select_important_features(train_features, train_failure, num_features=20, 
                              method='mutual_info'):
    if method == 'mutual_info':
        important_indices, _ = mutual_info_select(
                                            train_features, 
                                            train_failure, 
                                            num_features=num_features)
    elif method == 'feature_importance':
        important_indices, _ = feature_importance_select(
                                            train_features, 
                                            num_features=num_features)        
    else:
        raise ValueError('Unknown feature selection method') 
        
    train_sparse_features = train_features[:, important_indices]
    return train_sparse_features, important_indices

# Normalize data using mean and standard deviation
def normalize(train_data, test_data):
    scaler = StandardScaler()
    scaler.fit(train_data)
    train_data_n = scaler.transform(train_data)
    test_data_n = scaler.transform(test_data)
    return train_data_n, test_data_n

# Load images of specified indices using data_loader
def load_images(indices, dataset):
    img_list = []
    labels_list = []
    preds_list = []
    for idx in indices:
        img, label, pred = dataset.__getitem__(idx)
        img_list.append(img)
        labels_list.append(label)
        preds_list.append(pred)
    img_tensor = ch.stack(img_list, dim=0)
    label_tensor = np.array(labels_list)
    pred_tensor = np.array(preds_list)
    return img_tensor, label_tensor, pred_tensor

# Load model at path model_path/model_name
def load_model(model_name, model_path, dataset):
    if model_name == 'ImageNetNat.pt':
        model_kwargs = {
            'arch': 'resnet50',
            'dataset': dataset,
            'pytorch_pretrained': True,
            'parallel': False
        }
    else:
        model_kwargs = {
            'arch': 'resnet50',
            'dataset': dataset,
            'resume_path': model_path + model_name,
            'parallel': False
        }
    model, _ = model_utils.make_and_restore_model(**model_kwargs)
    model.eval()
    return model

def load_robust_model():
    dataset_function = getattr(dataset_utils, 'ImageNet')
    dataset = dataset_function('')

    model_kwargs = {
        'arch': 'resnet50',
        'dataset': dataset,
        'resume_path': f'./models/robust_resnet50.pth',
        'parallel': False
    }
    model, _ = model_utils.make_and_restore_model(**model_kwargs)
    model.eval()
    return model
    
def extract_features(robust_model, dataset, batch_size=32):
    data_loader = ch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size, 
                                              shuffle=False, num_workers=4)
    
    train_features, train_preds, train_labels = [], [], []
    total = 0
    for _, (ims, labels, preds) in enumerate(data_loader):
        ims = ims.cuda()
        batch_size = ims.shape[0]
        (_, features), _ = robust_model(ims, with_latent=True)
        features = features.detach().cpu().numpy()
        
        labels = np.array(labels)
        preds = np.array(preds)
        
        train_features.append(features)
        train_labels.append(labels)
        train_preds.append(preds)
        
    train_features = np.concatenate(train_features, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    train_preds = np.concatenate(train_preds, axis=0)
    return train_features, train_labels, train_preds

def print_with_stars(print_str, total_count=115, prefix="", suffix="", star='*'):
    str_len = len(print_str)
    left_len = (total_count - str_len)//2
    right_len = total_count - left_len - str_len
    final_str = "".join([star]*(left_len)) + print_str + "".join([star]*(right_len))
    final_str = prefix + final_str + suffix
    print(final_str)