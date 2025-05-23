import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import load_data, score_detail

# === Feature indices in the preprocessed NSL-KDD data ===
# These indices correspond to the columns after preprocessing.
duration_idx      = 0
protocol_type_idx = 1      # one-hot start for protocol type
# (Adjust these based on your actual preprocessing mapping)
src_bytes_idx     = 4
dst_bytes_idx     = 5
count_idx         = 22
srv_count_idx     = 23
serror_rate_idx   = 26
same_srv_rate_idx = 27
diff_srv_rate_idx = 28

# === Autoencoder Definition ===
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 80), nn.ReLU(),
            nn.Linear(80, 40), nn.ReLU(),
            nn.Linear(40, 20), nn.ReLU(),
            nn.Linear(20, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 20), nn.ReLU(),
            nn.Linear(20, 40), nn.ReLU(),
            nn.Linear(40, 80), nn.ReLU(),
            nn.Linear(80, input_dim), nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# === Classifier Definition ===
class Classifier(nn.Module):
    def __init__(self, latent_dim=8, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(latent_dim, num_classes)
    def forward(self, z):
        return self.fc(z)

# === Training Utilities ===

def train_autoencoder(x_train, device, epochs=20, lr=1e-3):
    input_dim = x_train.shape[1]
    ae = Autoencoder(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(ae.parameters(), lr=lr)
    criterion = nn.MSELoss()
    loader = DataLoader(x_train, batch_size=64, shuffle=True)
    ae.train()
    for _ in range(epochs):
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = criterion(ae(batch), batch)
            loss.backward()
            optimizer.step()
    return ae


def extract_latent(ae_model, x, device):
    ae_model.eval()
    with torch.no_grad():
        return ae_model.encoder(x.to(device))


def train_classifier(z_train, y_train, device, epochs=10, lr=1e-2):
    clf = Classifier(latent_dim=z_train.shape[1], num_classes=2).to(device)
    optimizer = torch.optim.Adam(clf.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    dataset = torch.utils.data.TensorDataset(z_train.to(device), y_train.to(device))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    clf.train()
    for _ in range(epochs):
        for z_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(clf(z_batch), y_batch)
            loss.backward()
            optimizer.step()
    return clf

# === Entry Point ===
if __name__ == "__main__":
    # Load raw NSL-KDD arrays: X_train, y_train, X_test, y_test
    X_train_np, y_train_np, X_test_np, y_test_np = load_data('NSL_pre_data')

    # Convert multi-class labels to binary (0=normal, 1=attack)
    y_train_bin = (y_train_np != 0).astype(int)
    y_test_bin  = (y_test_np  != 0).astype(int)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare tensors
    x_train_t = torch.tensor(X_train_np, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_bin, dtype=torch.long)
    x_test_t  = torch.tensor(X_test_np,  dtype=torch.float32)
    y_test_t  = torch.tensor(y_test_bin,  dtype=torch.long)

    # Train AE and binary classifier
    ae = train_autoencoder(x_train_t, device)
    z_train = extract_latent(ae, x_train_t, device)
    clf     = train_classifier(z_train, y_train_t, device)

    # Evaluate on clean test set
    with torch.no_grad():
        z_test = extract_latent(ae, x_test_t, device)
        out    = clf(z_test)
    report = score_detail(out, y_test_t)['report']
    print("=== Binary Clean Test Metrics ===")
    print(report)

    # Now export models or make available for synthetic scenario scripts
