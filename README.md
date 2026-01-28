# Jane Street Market Prediction

AE-MLP + XGBoost ensemble solution achieving **+12% utility score improvement** over baseline.

## Key Results

| Strategy | Utility Score | vs Baseline |
|----------|--------------|-------------|
| **Best Ensemble** (XGB + resp_3 MLP) | **2675** | **+12.0%** |
| Baseline Ensemble (Original) | 2388 | - |
| Best XGBoost | 2020 | -15.4% |
| Best AE-MLP (resp_3) | 2227 | -6.7% |

## Core Optimizations

1. **Unified Target Selection**: resp_3 for both XGBoost and AE-MLP (reduced ensemble variance)
2. **Black Swan Detection**: IQR-based filtering of 30 extreme market days (6% of data)
3. **Threshold Optimization Experiments**: t=0.51 for final ensemble

## Tech Stack

- **Deep Learning**: TensorFlow/Keras (Autoencoder-MLP architecture)
- **Gradient Boosting**: XGBoost (GPU-accelerated)
- **Validation**: PurgedGroupTimeSeriesSplit (prevents data leakage)

## Key Techniques

- Denoising Autoencoder for robust feature learning
- Time series CV with purging gap (31 days)
- Multi-horizon response analysis (resp, resp_1-4)
- Distribution shift detection (date>85 filtering)

## Results Visualization

Final comparison shows ensemble strategy significantly outperforms individual models:
- AUC: 0.536
- Utility: 2675
- Optimal threshold: 0.51

---

*Competition: Jane Street Real-Time Market Data Forecasting*

Data source: https://www.kaggle.com/datasets/jacksong23/jane-street
