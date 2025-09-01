# My AOC-IDS Experiments

This repository contains my adaptation of the AOC-IDS framework by xinchen930:
https://github.com/xinchen930/AOC-IDS

This project explores **lightweight model repair** for Intrusion Detection Systems (IDS), enabling rapid recovery against **synthetic zero-day attacks** without retraining the entire model. By fine-tuning only the classifier head of a stacked autoencoder, the system restores detection performance efficiently and with minimal resources.

## Features
- **Stacked Autoencoder + Classifier Head** trained on the NSL-KDD dataset  
- **Synthetic Zero-Day Attack Generation** via feature perturbation (DNS tunneling, IoT mass-scan, C2 beaconing, Slowloris flood)  
- **Lightweight Repair** targeting only the classifier head (single linear layer)  
- **Rapid Recovery**: recall restored from as low as 21% to nearly 100%  
- **Generalization**: ≥97% recall across pooled and leave-one-out scenarios  

## Results
- Baseline model fails on novel threats (recall 21–34%)  
- Classifier-head repair restores up to **100% recall** within a few epochs  
- Unified pooled repair covers multiple attacks simultaneously (≥97% recall)  
- Leave-one-out tests confirm strong generalization to unseen attacks  

## Tech Stack
- **Python 3**
- **PyTorch**
- **Pandas / NumPy**
- **Matplotlib** (for results visualization)


