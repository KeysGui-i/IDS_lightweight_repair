import torch
import numpy as np
import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
import torch
import torch.nn as nn
import math
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score,confusion_matrix, precision_score, recall_score, f1_score, classification_report
import scipy.optimize as opt
import torch.distributions as dist
from sklearn.metrics import accuracy_score

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def load_data(dataset_dir='NSL_pre_data'):
    # Paths to the CSV files
    train_path = os.path.join(dataset_dir, 'PKDDTrain+.csv')
    test_path  = os.path.join(dataset_dir, 'PKDDTest+.csv')

    # Read CSVs
    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)

    # Assume last column is the label
    label_col = df_train.columns[-1]

    # Separate out labels
    y_train = df_train[label_col].copy()
    y_test  = df_test[label_col].copy()
    df_train = df_train.drop(columns=[label_col])
    df_test  = df_test.drop(columns=[label_col])

    # Automatically find all object-dtype (categorical) columns
    cat_cols = df_train.select_dtypes(include=['object']).columns.tolist()

    # One-hot encode them
    df_train = pd.get_dummies(df_train, columns=cat_cols)
    df_test  = pd.get_dummies(df_test,  columns=cat_cols)

    # Align test to train columns (fill any missing ones with 0)
    df_test = df_test.reindex(columns=df_train.columns, fill_value=0)

    # Convert back to numpy arrays
    X_train = df_train.values
    X_test  = df_test.values

    # Label-encode the targets
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test  = le.transform(y_test)

    # Scale features to [0,1]
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test

class SplitData(BaseEstimator, TransformerMixin):
    def __init__(self, dataset):
        super(SplitData, self).__init__()
        self.dataset = dataset

    def fit(self, X, y=None):
        return self 
    
    def transform(self, X, labels, one_hot_label=True):
        if self.dataset == 'nsl':
            # Preparing the labels
            y = X[labels]
            X_ = X.drop(['labels5', 'labels2'], axis=1)
            # abnormal data is labeled as 1, normal data 0
            y = (y != 'normal')
            y_ = np.asarray(y).astype('float32')

        elif self.dataset == 'unsw':
            # UNSW dataset processing
            y_ = X[labels]
            X_ = X.drop('label', axis=1)

        else:
            raise ValueError("Unsupported dataset type")

        # Normalization
        normalize = MinMaxScaler().fit(X_)
        x_ = normalize.transform(X_)

        return x_, y_

def description(data):
    print("Number of samples(examples) ",data.shape[0]," Number of features",data.shape[1])
    print("Dimension of data set ",data.shape)

class AE(nn.Module):
    def __init__(self, input_dim):
        super(AE, self).__init__()

        # Find the nearest power of 2 to input_dim
        nearest_power_of_2 = 2 ** round(math.log2(input_dim))

        # Calculate the dimensions of the 2nd/4th layer and the 3rd layer.
        second_fourth_layer_size = nearest_power_of_2 // 2  # A half
        third_layer_size = nearest_power_of_2 // 4         # A quarter

        # Create encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, second_fourth_layer_size),
            nn.ReLU(),
            nn.Linear(second_fourth_layer_size, third_layer_size),
        )

        # Create decoder
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(third_layer_size, second_fourth_layer_size),
            nn.ReLU(),
            nn.Linear(second_fourth_layer_size, input_dim),
        )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode

'''
class CRCLoss(nn.Module):
    def __init__(self, device, temperature=0.1, scale_by_temperature=True):
        super(CRCLoss, self).__init__()
        self.device = device
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, features, labels=None, mask=None):        
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float()
        # compute logits
        logits = torch.div(
            torch.matmul(features, features.T),
            self.temperature)  # Calculate the dot product similarity between pairwise samples
        # create mask 
        logits_mask = torch.ones_like(mask).to(self.device) - torch.eye(batch_size).to(self.device)  
        logits_without_ii = logits * logits_mask
        
        logits_normal = logits_without_ii[(labels == 0).squeeze()]
        logits_normal_normal = logits_normal[:,(labels == 0).squeeze()]
        logits_normal_abnormal = logits_normal[:,(labels > 0).squeeze()]
        
        ## This is the denominator for InfoNCE loss: ONE time of traversal
        # sum_of_vium = torch.sum(torch.exp(logits_normal_abnormal), axis=1, keepdims=True)
        ## This is the denominator for our proposed CRC loss: TWO times of traversal
        sum_of_vium = torch.sum(torch.exp(logits_normal_abnormal))
        denominator = torch.exp(logits_normal_normal) + sum_of_vium
        log_probs = logits_normal_normal - torch.log(denominator)
  
        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature
        loss = loss.mean()
        return loss
'''    

def score_detail(model_outputs, y_true):
    # 1) get predicted classes
    y_pred = (
        model_outputs.argmax(dim=1).cpu().numpy()
        if hasattr(model_outputs, 'argmax') 
        else model_outputs.argmax(axis=1)
    )
    # 2) true labels to numpy
    y_true = y_true.cpu().numpy() if hasattr(y_true, 'cpu') else y_true

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec  = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1   = f1_score(y_true, y_pred, average=None, zero_division=0)

    # 3) classification report without fixed target_names
    report = classification_report(
        y_true, y_pred,
        zero_division=0
    )

    return {
        'accuracy': acc,
        'precision_per_class': prec,
        'recall_per_class': rec,
        'f1_per_class': f1,
        'report': report
    }

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def gaussian_pdf(x, mu, sigma):
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def log_likelihood(params, data):
    mu1, sigma1, mu2, sigma2 = params
    pdf1 = gaussian_pdf(data, mu1, sigma1)
    pdf2 = gaussian_pdf(data, mu2, sigma2)
    return -np.sum(np.log(0.5 * pdf1 + 0.5 * pdf2))

def evaluate(normal_temp, normal_recon_temp, x_train, y_train, x_test, y_test, model, get_confidence=False, en_or_de=False):
    num_of_layer = 0

    x_train_normal = x_train[(y_train == 0).squeeze()]
    x_train_abnormal = x_train[(y_train == 1).squeeze()]

    train_features = F.normalize(model(x_train)[num_of_layer], p=2, dim=1)
    train_features_normal = F.normalize(model(x_train_normal)[num_of_layer], p=2, dim=1)
    train_features_abnormal = F.normalize(model(x_train_abnormal)[num_of_layer], p=2, dim=1)
    test_features = F.normalize(model(x_test)[num_of_layer], p=2, dim=1)

    values_features_all, indcies = torch.sort(F.cosine_similarity(train_features, normal_temp.reshape([-1, normal_temp.shape[0]]), dim=1))
    values_features_normal, indcies = torch.sort(F.cosine_similarity(train_features_normal, normal_temp.reshape([-1, normal_temp.shape[0]]), dim=1))
    values_features_abnormal, indcies = torch.sort(F.cosine_similarity(train_features_abnormal, normal_temp.reshape([-1, normal_temp.shape[0]]), dim=1))

    values_features_all = values_features_all.cpu().detach().numpy()

    values_features_test = F.cosine_similarity(test_features, normal_temp.reshape([-1, normal_temp.shape[0]]))

    num_of_output = 1
    train_recon = F.normalize(model(x_train)[num_of_output], p=2, dim=1)
    train_recon_normal = F.normalize(model(x_train_normal)[num_of_output], p=2, dim=1)
    train_recon_abnormal = F.normalize(model(x_train_abnormal)[num_of_output], p=2, dim=1)
    test_recon = F.normalize(model(x_test)[num_of_output], p=2, dim=1)

    values_recon_all, indcies = torch.sort(F.cosine_similarity(train_recon, normal_recon_temp.reshape([-1, normal_recon_temp.shape[0]]), dim=1))
    values_recon_normal, indcies = torch.sort(F.cosine_similarity(train_recon_normal, normal_recon_temp.reshape([-1, normal_recon_temp.shape[0]]), dim=1))
    values_recon_abnormal, indcies = torch.sort(F.cosine_similarity(train_recon_abnormal, normal_recon_temp.reshape([-1, normal_recon_temp.shape[0]]), dim=1))

    values_recon_all = values_recon_all.cpu().detach().numpy()

    values_recon_test = F.cosine_similarity(test_recon, normal_recon_temp.reshape([-1, normal_recon_temp.shape[0]]), dim=1)

    mu1_initial = np.mean(values_features_normal.cpu().detach().numpy())
    sigma1_initial = np.std(values_features_normal.cpu().detach().numpy())

    mu2_initial = np.mean(values_features_abnormal.cpu().detach().numpy())
    sigma2_initial = np.std(values_features_abnormal.cpu().detach().numpy())

    # Fitting data to two Gaussian distributions using Maximum Likelihood Estimation (MLE)
    initial_params = np.array([mu1_initial, sigma1_initial, mu2_initial, sigma2_initial]) # Initial parameters
    result = opt.minimize(log_likelihood, initial_params, args=(values_features_all,), method='Nelder-Mead')
    mu1_fit, sigma1_fit, mu2_fit, sigma2_fit = result.x # Estimated parameter values

    if mu1_fit > mu2_fit:
        gaussian1 = dist.Normal(mu1_fit, sigma1_fit)
        gaussian2 = dist.Normal(mu2_fit, sigma2_fit)
    else:
        gaussian2 = dist.Normal(mu1_fit, sigma1_fit)
        gaussian1 = dist.Normal(mu2_fit, sigma2_fit)

    pdf1 = gaussian1.log_prob(values_features_test).exp()

    pdf2 = gaussian2.log_prob(values_features_test).exp()
    y_test_pred_2 = (pdf2 > pdf1).cpu().numpy().astype("int32")
    y_test_pro_en = (torch.abs(pdf2-pdf1)).cpu().detach().numpy().astype("float32")

    if isinstance(y_test, int) == False:
        if y_test.device != torch.device("cpu"):
            y_test = y_test.cpu().numpy()

    mu3_initial = np.mean(values_recon_normal.cpu().detach().numpy())
    sigma3_initial = np.std(values_recon_normal.cpu().detach().numpy())

    mu4_initial = np.mean(values_recon_abnormal.cpu().detach().numpy())
    sigma4_initial = np.std(values_recon_abnormal.cpu().detach().numpy())

    # Fitting data to two Gaussian distributions using Maximum Likelihood Estimation (MLE)
    initial_params = np.array([mu3_initial, sigma3_initial, mu4_initial, sigma4_initial]) # Initial parameters
    result = opt.minimize(log_likelihood, initial_params, args=(values_recon_all,), method='Nelder-Mead')
    mu3_fit, sigma3_fit, mu4_fit, sigma4_fit = result.x # Estimated parameter values

    if mu3_fit > mu4_fit:
        gaussian3 = dist.Normal(mu3_fit, sigma3_fit)
        gaussian4 = dist.Normal(mu4_fit, sigma4_fit)
    else:
        gaussian4 = dist.Normal(mu3_fit, sigma3_fit)
        gaussian3 = dist.Normal(mu4_fit, sigma4_fit)

    pdf3 = gaussian3.log_prob(values_recon_test).exp()

    pdf4 = gaussian4.log_prob(values_recon_test).exp()
    y_test_pred_4 = (pdf4 > pdf3).cpu().numpy().astype("int32")
    y_test_pro_de = (torch.abs(pdf4-pdf3)).cpu().detach().numpy().astype("float32")

    if not isinstance(y_test, int):
        if y_test.device != torch.device("cpu"):
            y_test = y_test.cpu().numpy()
        result_encoder = score_detail(y_test, y_test_pred_2)
        result_decoder = score_detail(y_test, y_test_pred_4)

    y_test_pred_no_vote = torch.where(torch.from_numpy(y_test_pro_en) > torch.from_numpy(y_test_pro_de), torch.from_numpy(y_test_pred_2), torch.from_numpy(y_test_pred_4))
    
    if not isinstance(y_test, int):
        result_final = score_detail(y_test, y_test_pred_no_vote, if_print=True)
        return result_encoder, result_decoder, result_final
    else:
        return y_test_pred_no_vote
    
def generate_synthetic_variants(
    X_orig: np.ndarray,
    y_orig: np.ndarray,
    attack_label: int,
    feature_idxs: list[int],
    delta_pct: float,
    n_samples: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample n_samples records of class attack_label from (X_orig, y_orig)
    and randomly perturb the columns in feature_idxs by ±delta_pct, clipping to [0,1].
    Returns (X_syn, y_syn).
    """
    idxs = np.where(y_orig == attack_label)[0]
    choices = np.random.choice(idxs, size=n_samples, replace=True)
    X_base = X_orig[choices].copy()

    for fi in feature_idxs:
        perturb = np.random.uniform(-delta_pct, delta_pct, size=n_samples)
        X_base[:, fi] *= (1 + perturb)

    X_base = np.clip(X_base, 0.0, 1.0)
    y_base = np.full(n_samples, attack_label, dtype=int)
    return X_base, y_base