import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from utils import load_data, generate_synthetic_variants, score_detail
from online_training import train_autoencoder, extract_latent, train_classifier, duration_idx, src_bytes_idx, dst_bytes_idx, count_idx, srv_count_idx, serror_rate_idx, same_srv_rate_idx, diff_srv_rate_idx
import numpy as np

# === Repair Functions ===
def repair_decoder(ae, X_edit_t, device, lr=1e-2, epochs=20, patience=5):
    # Freeze encoder and all but final decoder layer
    for p in ae.encoder.parameters(): p.requires_grad = False
    # decoder children indexed 0...6; final linear is 6
    for idx, module in ae.decoder.named_children():
        if idx != '6':
            for p in module.parameters(): p.requires_grad = False
    params = [p for p in ae.decoder[6].parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)
    criterion = nn.MSELoss()
    ds = TensorDataset(X_edit_t, X_edit_t)
    loader = DataLoader(ds, batch_size=64, shuffle=True)

    best_loss = float('inf')
    patience_counter = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x_in, _ in loader:
            optimizer.zero_grad()
            recon = ae(x_in)
            loss = criterion(recon, x_in)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(loader)
        print(f"[Decoder Repair] Epoch {epoch}: MSE={epoch_loss:.4f}")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            best_state = copy.deepcopy(ae.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping decoder repair")
                break
    ae.load_state_dict(best_state)
    return ae


def repair_classifier_head(clf, ae, X_edit_t, y_edit_t, device,
                           lr=1e-2, epochs=20):
    # Freeze all classifier params except head
    for p in clf.parameters(): p.requires_grad = False
    for p in clf.fc.parameters(): p.requires_grad = True
    optimizer = torch.optim.Adam(clf.fc.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Precompute latent codes for edit set
    ae.eval()
    with torch.no_grad():
        z_edit = ae.encoder(X_edit_t)
    ds = TensorDataset(z_edit, y_edit_t)
    loader = DataLoader(ds, batch_size=64, shuffle=True)

    best_acc = 0.0
    for epoch in range(epochs):
        correct = total = 0
        for z_batch, y_batch in loader:
            optimizer.zero_grad()
            logits = clf(z_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
        acc = correct / total
        print(f"[Head Repair] Epoch {epoch}: acc={acc:.3f}")
        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(clf.state_dict())
    clf.load_state_dict(best_state)
    return clf

# === Main: Demonstrate Repair on All Scenarios ===
if __name__ == '__main__':
    # 1) Load data and train baseline
    X_train_np, y_train_np, X_test_np, y_test_np = load_data('NSL_pre_data')
    y_train_bin = (y_train_np != 0).astype(int)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_train_t = torch.tensor(X_train_np, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_bin, dtype=torch.long)

    ae = train_autoencoder(x_train_t, device)
    z_train = extract_latent(ae, x_train_t, device)
    clf = train_classifier(z_train, y_train_t, device)

    # 2) Define scenarios
    scenarios = [
        ('DNS Tunneling Exfiltration', 0, [dst_bytes_idx, src_bytes_idx], 1.0),
        ('IoT Mass-Scan',             2, [count_idx, srv_count_idx, serror_rate_idx], 0.5),
        ('Encrypted C2 Beaconing',    0, [same_srv_rate_idx, diff_srv_rate_idx], 0.5),
        ('Slowloris HTTP Flood',      0, [duration_idx, count_idx, srv_count_idx], 2.0)
    ]

    for name, label, idxs, delta in scenarios:
        # Generate edit set
        X_edit, _ = generate_synthetic_variants(
            X_test_np, y_test_np,
            attack_label=label,
            feature_idxs=idxs,
            delta_pct=delta,
            n_samples=200
        )
        X_edit_t = torch.tensor(X_edit, dtype=torch.float32).to(device)
        y_edit_t = torch.ones(X_edit.shape[0], dtype=torch.long).to(device)
        # --- 新增：从测试集中拿出相同数量的正常样本 ---
        X_norm = X_test_np[y_test_np == 0]
        X_norm = X_norm[: X_edit.shape[0]]                # 保持平衡
        y_norm = np.zeros(X_norm.shape[0], dtype=int)

        # 把正常 + 攻击 合并
        X_eval = np.vstack([X_norm, X_edit])
        y_eval = np.hstack([y_norm, np.ones(X_edit.shape[0], dtype=int)])

        # 转为 tensor
        X_eval_t = torch.tensor(X_eval, dtype=torch.float32).to(device)
        y_eval_t = torch.tensor(y_eval, dtype=torch.long).to(device)

        # Baseline performance
        print(f"\n=== Baseline: {name} ===")
        with torch.no_grad():
            z = extract_latent(ae, X_edit_t, device)
            out = clf(z)
        print(score_detail(out, y_edit_t)['report'])
        with torch.no_grad():
                z = extract_latent(ae, X_eval_t, device)
                out = clf(z)
        print(score_detail(out, y_eval_t)['report'])
        # Decoder repair
        ae_dec = copy.deepcopy(ae)
        ae_dec = repair_decoder(ae_dec, X_edit_t, device)
        with torch.no_grad():
            z_dec = extract_latent(ae_dec, X_edit_t, device)
            out_dec = clf(z_dec)
        print(f"=== After Decoder Repair: {name} ===")
        print(score_detail(out_dec, y_edit_t)['report'])

        # Classifier-head repair
        clf_head = copy.deepcopy(clf)
        clf_head = repair_classifier_head(clf_head, ae, X_edit_t, y_edit_t, device)
        with torch.no_grad():
            z_orig = extract_latent(ae, X_edit_t, device)
            out_head = clf_head(z_orig)
        print(f"=== After Head Repair: {name} ===")
        print(score_detail(out_head, y_edit_t)['report'])
        
            # --- Unified Repair on Pooled Edit Set ---
    print("\n=== Unified Pooled Repair ===")

    # 1) Generate a small edit set for each scenario (e.g. 50 samples each)
    pooled_X, pooled_y = [], []
    for name, label, idxs, delta in scenarios:
        X_part, _ = generate_synthetic_variants(
            X_test_np, y_test_np,
            attack_label=label,
            feature_idxs=idxs,
            delta_pct=delta,
            n_samples=50      # fewer per scenario
        )
        pooled_X.append(X_part)
        pooled_y.append(np.ones(X_part.shape[0], dtype=int))
    # concatenate into one big array
    X_pooled = np.vstack(pooled_X)
    y_pooled = np.hstack(pooled_y)

    # 2) Prepare tensors
    X_pooled_t = torch.tensor(X_pooled, dtype=torch.float32).to(device)
    y_pooled_t = torch.tensor(y_pooled, dtype=torch.long).to(device)

    # 3) Extract latent codes once
    ae.eval()
    with torch.no_grad():
        Z_pooled = ae.encoder(X_pooled_t)

    # 4) Retrain only the softmax head on the pooled codes
    #    (you can reuse repair_classifier_head, or do it inline:)
    for p in clf.parameters(): p.requires_grad = False
    for p in clf.fc.parameters(): p.requires_grad = True
    opt = torch.optim.Adam(clf.fc.parameters(), lr=1e-2)
    crit = nn.CrossEntropyLoss()
    ds = TensorDataset(Z_pooled, y_pooled_t)
    loader = DataLoader(ds, batch_size=64, shuffle=True)

    # simple 10-epoch retrain
    for epoch in range(10):
        for z_batch, y_batch in loader:
            opt.zero_grad()
            loss = crit(clf(z_batch), y_batch)
            loss.backward()
            opt.step()

    # 5) Evaluate on each scenario
    for name, label, idxs, delta in scenarios:
        X_syn, _ = generate_synthetic_variants(
            X_test_np, y_test_np,
            attack_label=label,
            feature_idxs=idxs,
            delta_pct=delta,
            n_samples=200
        )
        X_t = torch.tensor(X_syn, dtype=torch.float32).to(device)
        with torch.no_grad():
            Z = ae.encoder(X_t)
            out = clf(Z)
        print(f"\n=== Post-Unified Repair: {name} ===")
        print(score_detail(out, torch.ones(200, dtype=torch.long).to(device))['report'])

    import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# … assume `scenarios`, `X_test_np`, `y_test_np`, `ae`, `clf`, `device` are already defined …

print("\n=== Leave-One-Out Pooled Repair ===")
for hold_name, hold_label, _, _ in scenarios:
    # 1) Build pooled edit set *excluding* the held-out scenario
    pooled_X, pooled_y = [], []
    for name, label, idxs, delta in scenarios:
        if name == hold_name:
            continue
        X_part, _ = generate_synthetic_variants(
            X_test_np, y_test_np,
            attack_label=label,
            feature_idxs=idxs,
            delta_pct=delta,
            n_samples=50
        )
        pooled_X.append(X_part)
        pooled_y.append(np.ones(X_part.shape[0], dtype=int))
    X_pool = np.vstack(pooled_X)
    y_pool = np.hstack(pooled_y)

    # 2) Prepare tensors & latent codes
    X_pool_t = torch.tensor(X_pool, dtype=torch.float32).to(device)
    y_pool_t = torch.tensor(y_pool, dtype=torch.long).to(device)
    with torch.no_grad():
        Z_pool = ae.encoder(X_pool_t)

    # 3) Retrain only the head on this pooled set
    for p in clf.parameters(): p.requires_grad = False
    for p in clf.fc.parameters(): p.requires_grad = True
    optimizer = torch.optim.Adam(clf.fc.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(Z_pool, y_pool_t),
                        batch_size=64, shuffle=True)
    for epoch in range(10):
        for z_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(clf(z_batch), y_batch)
            loss.backward()
            optimizer.step()

    # 4) Evaluate on the held-out scenario
    print(f"\n--- Held-out: {hold_name} ---")
    X_ho, _ = generate_synthetic_variants(
        X_test_np, y_test_np,
        attack_label=hold_label,
        feature_idxs=[i for n, l, i, d in scenarios if n == hold_name][0],
        delta_pct=[d for n, l, i, d in scenarios if n == hold_name][0],
        n_samples=200
    )
    X_ho_t = torch.tensor(X_ho, dtype=torch.float32).to(device)
    with torch.no_grad():
        Z_ho = ae.encoder(X_ho_t)
        out  = clf(Z_ho)
    print(score_detail(out, torch.ones(200, dtype=torch.long).to(device))['report'])
