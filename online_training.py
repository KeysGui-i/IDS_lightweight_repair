import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Utility imports: load_data should load NSL_pre_data specifically
from utils import load_data, AE, score_detail, generate_synthetic_variants

def main():
    # 0) Load preprocessed NSL-KDD data
    x_train, y_train, x_test, y_test = load_data('NSL_pre_data')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert to torch tensors
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    x_test_tensor  = torch.tensor(x_test,  dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor  = torch.tensor(y_test,  dtype=torch.long)

    # ----------------------------------------------------------------
    # 1) Train autoencoder on NSL_pre_data (unsupervised reconstruction)
    # ----------------------------------------------------------------
    ae_dataset = TensorDataset(x_train_tensor, x_train_tensor)
    ae_loader  = DataLoader(ae_dataset, batch_size=128, shuffle=True)

    input_dim    = x_train.shape[1]
    ae_model     = AE(input_dim).to(device)
    ae_optimizer = optim.Adam(ae_model.parameters(), lr=1e-3)
    ae_criterion = nn.MSELoss()

    num_epochs_ae = 20
    ae_model.train()
    for epoch in range(num_epochs_ae):
        total_loss = 0.0
        for batch_inputs, _ in ae_loader:
            batch_inputs = batch_inputs.to(device)
            ae_optimizer.zero_grad()
            _, decoded = ae_model(batch_inputs)
            loss = ae_criterion(decoded, batch_inputs)
            loss.backward()
            ae_optimizer.step()
            total_loss += loss.item() * batch_inputs.size(0)
        avg_loss = total_loss / len(ae_loader.dataset)
        print(f"[AE] Epoch {epoch+1}/{num_epochs_ae}, Loss: {avg_loss:.4f}")

    # Save autoencoder weights
    torch.save(ae_model.state_dict(), "ae_nsl_pre_data.pt")

    # --------------------------
    # 2) Extract latent features
    # --------------------------
    ae_model.eval()
    with torch.no_grad():
        z_train = ae_model.encoder(x_train_tensor.to(device)).cpu()
        z_test  = ae_model.encoder(x_test_tensor.to(device)).cpu()

    # --------------------------------
    # 3) Train simple classifier head
    # --------------------------------
    num_classes  = len(torch.unique(y_train_tensor))
    classifier   = nn.Linear(z_train.size(1), num_classes).to(device)
    clf_optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    clf_criterion = nn.CrossEntropyLoss()

    clf_dataset = TensorDataset(z_train, y_train_tensor)
    clf_loader  = DataLoader(clf_dataset, batch_size=64, shuffle=True)

    num_epochs_clf = 10
    classifier.train()
    for epoch in range(num_epochs_clf):
        total_loss = 0.0
        for batch_z, batch_y in clf_loader:
            batch_z = batch_z.to(device)
            batch_y = batch_y.to(device)
            clf_optimizer.zero_grad()
            outputs = classifier(batch_z)
            loss = clf_criterion(outputs, batch_y)
            loss.backward()
            clf_optimizer.step()
            total_loss += loss.item() * batch_z.size(0)
        avg_loss = total_loss / len(clf_loader.dataset)
        print(f"[Clf] Epoch {epoch+1}/{num_epochs_clf}, Loss: {avg_loss:.4f}")

    # --------------------------------------------------
    # 4) Evaluate on clean test set (using latent z_test)
    # --------------------------------------------------
    classifier.eval()
    with torch.no_grad():
        test_outputs = classifier(z_test.to(device))
    metrics_clean = score_detail(test_outputs, y_test_tensor)
    print("=== Clean Test Metrics ===")
    print(metrics_clean['report'])

    # -------------------------------------------------------
    # 5) Generate & evaluate on a small synthetic DoS edit set
    # -------------------------------------------------------
    X_syn, y_syn = generate_synthetic_variants(
        X_orig=x_train, y_orig=y_train,
        attack_label=1,              # e.g. DoS label index
        feature_idxs=[0, 4, 5],      # adjust these column indices as needed
        delta_pct=0.15,
        n_samples=100
    )
    X_syn_tensor = torch.tensor(X_syn, dtype=torch.float32).to(device)
    y_syn_tensor = torch.tensor(y_syn, dtype=torch.long)

    with torch.no_grad():
        z_syn = ae_model.encoder(X_syn_tensor).cpu().to(device)
        syn_outputs = classifier(z_syn)
    metrics_syn = score_detail(syn_outputs, y_syn_tensor)
    print("=== Synthetic DoS Variants Metrics ===")
    print(metrics_syn['report'])

if __name__ == '__main__':
    main()