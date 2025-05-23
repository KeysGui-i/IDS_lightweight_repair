import numpy as np
import torch
from utils import load_data, generate_synthetic_variants, score_detail
from online_training import (
    train_autoencoder,
    extract_latent,
    train_classifier,
    duration_idx,
    src_bytes_idx,
    dst_bytes_idx,
    count_idx,
    srv_count_idx,
    serror_rate_idx,
    same_srv_rate_idx,
    diff_srv_rate_idx
)

def eval_scenario(ae, clf, X_syn, name):
    clf.eval(); ae.eval()
    X_t = torch.tensor(X_syn, dtype=torch.float32)
    with torch.no_grad():
        z = extract_latent(ae, X_t, device)
        out = clf(z)
    metrics = score_detail(out, torch.ones(X_syn.shape[0], dtype=torch.long))
    print(f"=== {name} Metrics ===")
    print(metrics['report'])

if __name__ == '__main__':
    # 1. Load and preprocess NSL-KDD data
    X_train_np, y_train_np, X_test_np, y_test_np = load_data('NSL_pre_data')
    # Binary labels: 0 = normal, 1 = attack
    y_train_bin = (y_train_np != 0).astype(int)
    y_test_bin  = (y_test_np  != 0).astype(int)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_train_t = torch.tensor(X_train_np, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_bin, dtype=torch.long)

    # 2. Train AE and binary classifier
    ae = train_autoencoder(x_train_t, device)
    z_train = extract_latent(ae, x_train_t, device)
    clf = train_classifier(z_train, y_train_t, device)

    # 3. Evaluate clean test performance
    x_test_t = torch.tensor(X_test_np, dtype=torch.float32)
    with torch.no_grad():
        z_test = extract_latent(ae, x_test_t, device)
        out = clf(z_test)
    clean_metrics = score_detail(out, torch.tensor(y_test_bin, dtype=torch.long))
    print("=== Clean Test Metrics ===")
    print(clean_metrics['report'])

    # 4. Scenario: DNS Tunneling Exfiltration
    X_dns, _ = generate_synthetic_variants(
        X_test_np, y_test_np,
        attack_label=0,
        feature_idxs=[dst_bytes_idx, src_bytes_idx],
        delta_pct=1.0,
        n_samples=200
    )
    eval_scenario(ae, clf, X_dns, "DNS Tunneling Exfiltration")

    # 5. Scenario: IoT Mass-Scan
    X_iot, _ = generate_synthetic_variants(
        X_test_np, y_test_np,
        attack_label=2,
        feature_idxs=[count_idx, srv_count_idx, serror_rate_idx],
        delta_pct=0.5,
        n_samples=200
    )
    eval_scenario(ae, clf, X_iot, "IoT Mass-Scan")

    # 6. Scenario: Encrypted C2 Beaconing
    X_c2, _ = generate_synthetic_variants(
        X_test_np, y_test_np,
        attack_label=0,
        feature_idxs=[same_srv_rate_idx, diff_srv_rate_idx],
        delta_pct=0.5,
        n_samples=200
    )
    eval_scenario(ae, clf, X_c2, "Encrypted C2 Beaconing")
    X_slow, _ = generate_synthetic_variants(
        X_test_np, y_test_np,
        attack_label=0,
        feature_idxs=[duration_idx, count_idx, srv_count_idx],
        delta_pct=2.0,
        n_samples=200
    )
    eval_scenario(ae, clf, X_slow, "Slowloris HTTP Flood")
