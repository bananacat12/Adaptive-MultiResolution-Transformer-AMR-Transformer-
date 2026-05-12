<div align="center">

# 🔭 AMR-Transformer
### Adaptive Multi-Resolution Transformer for Time Series Forecasting

</div>

---

## 📖 Table of Contents

* [Overview](#-overview)
* [Key Contributions](#-key-contributions)
* [Architecture](#-architecture)
  * [Core Components](#core-components)
  * [Data Flow](#data-flow)
* [Datasets](#-datasets)
* [Baselines](#-baselines)
* [Results](#-results)
  * [Overall Performance](#overall-performance)
  * [Multi-Horizon Evaluation](#multi-horizon-evaluation)
  * [Computational Efficiency](#computational-efficiency)
* [Installation](#-installation)
* [Usage](#-usage)
* [Project Structure](#-project-structure)
* [Citation](#-citation)

---

## 🌟 Overview

**Time Series Forecasting (TSF)** is a critical problem in domains such as energy, transportation, and finance. While Transformer-based models excel at capturing long-range dependencies, they suffer from a fundamental limitation: **fixed temporal resolution**. This constraint prevents models from simultaneously capturing both short-term fluctuations and long-term global trends.

This project proposes the **Adaptive Multi-Resolution Transformer (AMR-Transformer)** — a novel architecture that **dynamically learns and fuses features across multiple temporal granularities** within each processing layer.

> **Core Hypothesis:** Early layers should process high-resolution signals (fine details), while deeper layers should focus on low-resolution signals (global trends) — without any rigid, pre-defined scale constraints.

<div align="center">

```
          Fixed-Resolution Models             AMR-Transformer
          ─────────────────────────           ────────────────────────────
          Input ──► [Resolution=1] ──► Output  Input ──► [ARM: Scale 1,2,4,8,12,24]
                        ↑                                        ↓
                  Same scale for                   Layer 1: High-Res  (fine details)
                  every layer                      Layer 2: Mid-Res   (patterns)
                  & every data                     Layer 3: Low-Res   (global trends)
                                                            ↓
                                                   Adaptive Weighted Fusion ──► Output
```

</div>

---

## 🚀 Key Contributions

| # | Contribution | Description |
|---|---|---|
| 1 | **Learnable Resampling** | Continuous 1D bilinear interpolation with differentiable scale parameters — allows a branch initialized at period 2 to fine-tune to 2.15, capturing irregular seasonality missed by fixed methods |
| 2 | **Resolution-Aware Attention** | Multi-scale features fused directly at the QKV level, enabling simultaneous attention to fine-grained local details and broad global trends |
| 3 | **Adaptive Router** | Query-based attention mechanism that dynamically weights resolution branches (α weights) per layer based on input context |
| 4 | **Continuous Scale-Aware PE** | Positional encoding that scales with learned temporal granularity, preserving absolute time information across compressed sequences |
| 5 | **Comprehensive Validation** | Benchmarked against LSTM, Vanilla Transformer, DLinear, and PatchTST on 4 diverse real-world datasets |

---

## 🏗 Architecture

### Core Components

The AMR-Transformer is built upon **4 main components**:

<div align="center">

![AMR-Transformer Overview](assets/amr-overview.png)
*Full AMR-Transformer architecture — from input normalization to forecast output*

</div>

### Data Flow

**Phase 1 — Preprocessing:**
1. Input `X ∈ ℝ^(T×1)` is normalized via **RevIN** (Reversible Instance Normalization)
2. Linear projection to latent dimension *D*
3. Continuous Scale-Aware Positional Encoding (base scale = 1)

**Phase 2 — AMR Encoder (×L layers):**
1. **ARM** generates K branches via learnable resampling: `Xₖ = Resample(X, σₖ)`
2. **Adaptive Router** computes mixing weights: `α = Softmax(MLP(Attention(Q_router, X, X)))`
3. Each branch is refined by a lightweight `Conv1d + GELU` and `Scale-Aware PE`
4. All branches are aligned to length *T* and fused at QKV level:
   - `Q_fused = Σ αₖ·Qₖ + C_mix`
5. Single Multi-Head Attention on fused QKV → residual + LayerNorm → FFN → residual + LayerNorm

<div align="center">

![AMR Block](assets/AMR_Block.jpg)
*Internal data flow of one AMR Transformer Block*

</div>

**Phase 3 — Prediction Head:**
1. Flatten: `Z_flat ∈ ℝ^(T·D)`
2. Linear: `Ŷ_norm ∈ ℝ^τ`
3. RevIN Denormalization: `Ŷ = Ŷ_norm × σ + μ`

### Model Configuration

| Parameter | Value | Description |
|---|---|---|
| Look-back Window | 96 | Input sequence length |
| Prediction Horizon | 15 | Output forecast steps |
| Model Dimension | 64 | Latent embedding size |
| Attention Heads | 4 | Multi-head attention |
| Encoder Layers | 3 | Depth of AMR blocks |
| Dropout | 0.2 | Applied after attention & FFN |
| Initial Periods | [1, 2, 4, 8, 12, 24] | ARM branch scales |
| Optimizer | Adam | lr = 0.0005 |
| Batch Size | 32 | Training batch size |
| Max Epochs | 40 | With early stopping (patience=5) |
| Loss Function | MSE | `nn.MSELoss` |
| Normalization | RevIN | Reversible Instance Normalization |

---

## 📊 Datasets

All four datasets are standard benchmarks from the LTSF research community (used in Autoformer, Informer, LSTNet, PatchTST). A **Channel Independence** strategy is applied: each variable is treated as an independent univariate series.

| Dataset | Timesteps | Features | Frequency | Domain | Characteristics | Split |
|---|---|---|---|---|---|---|
| **Weather** | 52,696 | 21 | 10 min | Meteorology | Smooth, strongly periodic (daily/yearly cycles) | 7:1:2 |
| **Traffic** | 17,544 | 862 | Hourly | Transportation | Strong daily + weekly periodicity (rush hours) | 7:1:2 |
| **Electricity** | 26,304 | 321 | Hourly | Energy | Non-stationary, spikes, distribution shifts | 7:1:2 |
| **Exchange-Rate** | 7,588 | 8 | Daily | Finance | Chaotic, stochastic, no clear periodicity | 7:1:2 |

**Data Sources:**
- **Weather** — Max Planck Institute for Biogeochemistry; standardized by [Autoformer](https://arxiv.org/pdf/2106.13008)
- **Traffic** — Caltrans PeMS (SF Bay Area freeways); introduced by [LSTNet](https://arxiv.org/pdf/1703.07015)
- **Electricity** — UCI Machine Learning Repository; processed by [LSTNet](https://arxiv.org/pdf/1703.07015)
- **Exchange-Rate** — 8 currencies (1990–2016); standardized by [LSTNet](https://arxiv.org/pdf/1703.07015)

**Preprocessing:**
```python
# Channel independence: treat each feature as independent univariate series
X_multi ∈ ℝ^(T×C)  →  X_uni ∈ ℝ^(T·C × 1)

# Data split (chronological, no shuffle)
Train : Validation : Test = 7 : 1 : 2

# Target variable
Y_target = X[:, -1]   # Last column (OT in Weather, last sensor/client in others)
```

---

## ⚖️ Baselines

| Model | Type | Key Feature | Reference |
|---|---|---|---|
| **LSTM** | Recurrent | Gating mechanism for sequential dependencies | [Hochreiter & Schmidhuber, 1997](https://www.bioinf.jku.at/publications/older/2604.pdf) |
| **Transformer** | Attention | Standard fixed-resolution self-attention | [Vaswani et al., 2017](https://arxiv.org/pdf/1706.03762) |
| **DLinear** | Linear | Trend + Seasonality decomposition; single linear layer | [Zeng et al., 2022](https://arxiv.org/pdf/2205.13504) |
| **PatchTST** | Patch+Attention | Patching for locality; fixed patch size P=16 | [Nie et al., 2023](https://arxiv.org/pdf/2211.14730) |

All baselines share the same hyperparameter budget:
- Hidden dimension: **64** | Dropout: **0.2** | Look-back: **96** | Horizon: **15**
- RevIN normalization applied to all models

---

## 📈 Results

### Overall Performance

Forecasting results at horizon **τ = 15**, look-back window **T = 96**.

#### Electricity (Non-stationary)

| Model | MSE | MAE | RMSE | MAPE (%) | SMAPE (%) |
|---|---|---|---|---|---|
| **AMR** ⭐ | **52,344.66** | **163.45** | **228.79** | **4.77** | **4.71** |
| PatchTST | 52,844.55 | 165.72 | 229.88 | 4.84 | 4.79 |
| LSTM | 55,434.59 | 168.52 | 235.45 | 4.92 | 4.89 |
| Transformer | 60,596.18 | 175.61 | 246.16 | 5.16 | 5.10 |
| DLinear | 80,800.72 | 203.08 | 284.25 | 5.95 | 5.90 |

#### Traffic (Periodic)

| Model | MSE | MAE | RMSE | MAPE (%) | SMAPE (%) |
|---|---|---|---|---|---|
| **AMR** ⭐ | **0.000037** | **0.0036** | **0.0061** | **14.89** | **15.16** |
| PatchTST | 0.000041 | 0.0039 | 0.0064 | 18.19 | 20.26 |
| Transformer | 0.000042 | 0.0042 | 0.0065 | 19.06 | 21.27 |
| LSTM | 0.000048 | 0.0041 | 0.0069 | 17.33 | 17.43 |
| DLinear | 0.000065 | 0.0053 | 0.0081 | 21.86 | 20.16 |

> AMR reduces MSE by **~10% vs PatchTST** and **~43% vs DLinear** on Traffic.

#### Weather (Smooth)

| Model | MSE | MAE | RMSE | MAPE (%) | SMAPE (%) |
|---|---|---|---|---|---|
| DLinear ⭐ | **48.53** | **4.62** | **6.97** | **1.05** | **1.05** |
| **AMR** 🥈 | 63.94 | 5.72 | 8.00 | 1.29 | 1.29 |
| PatchTST | 67.67 | 5.82 | 8.23 | 1.33 | 1.32 |
| Transformer | 75.72 | 6.28 | 8.70 | 1.42 | 1.42 |
| LSTM | 95.47 | 7.21 | 9.77 | 1.64 | 1.63 |

#### Exchange-Rate (Chaotic)

| Model | MSE | MAE | RMSE | MAPE (%) | SMAPE (%) |
|---|---|---|---|---|---|
| DLinear ⭐ | **0.000157** | **0.0095** | **0.0125** | **1.27** | **1.26** |
| **AMR** 🥈 | 0.000170 | 0.0099 | 0.0130 | 1.32 | 1.32 |
| PatchTST | 0.000184 | 0.0103 | 0.0136 | 1.37 | 1.37 |
| Transformer | 0.000186 | 0.0103 | 0.0137 | 1.37 | 1.36 |
| LSTM | 0.000189 | 0.0105 | 0.0137 | 1.41 | 1.40 |

> AMR is the **best-performing deep learning model on all 4 datasets**, consistently outperforming all Transformer-based baselines.

---

### Multi-Horizon Evaluation

Performance across four specific steps: **Step 1** (immediate), **Step 6**, **Step 10**, **Step 15** (long-term).

#### Electricity — Multi-horizon MSE

| Model | Step 1 | Step 6 | Step 10 | Step 15 |
|---|---|---|---|---|
| **AMR** | 19,535 | 55,005 | 67,746 | **66,339** ⭐ |
| PatchTST | 20,142 | 49,096 | 63,657 | 68,682 |
| LSTM | 19,381 | 52,187 | 65,357 | 75,800 |
| Transformer | 22,119 | 55,057 | 74,737 | 77,667 |
| DLinear | **17,618** | 70,501 | 102,798 | 118,872 |

> DLinear leads at Step 1, but **degrades catastrophically** to 118K by Step 15. AMR maintains the lowest error at the long-term horizon.

#### Traffic — Multi-horizon MSE

| Model | Step 1 | Step 6 | Step 10 | Step 15 |
|---|---|---|---|---|
| **AMR** | **0.00002** | **0.00004** | **0.00004** | **0.00004** ⭐ |
| PatchTST | 0.00003 | 0.00004 | 0.00004 | 0.00005 |
| Transformer | 0.00003 | 0.00004 | 0.00004 | 0.00005 |
| LSTM | 0.00003 | 0.00005 | 0.00005 | 0.00006 |
| DLinear | 0.00003 | 0.00007 | 0.00007 | 0.00008 |

> AMR's error remains **remarkably stable** (0.00004) from Step 6 to Step 15 — evidence that learned scales successfully capture weekly periodicity.

#### Weather — Multi-horizon (crossover effect)

| Model | Step 1 | Step 6 | Step 10 | Step 15 |
|---|---|---|---|---|
| DLinear | **7.07** | 34.23 | 57.11 | 97.87 |
| **AMR** | 32.33 | **51.93** | **74.03** | **93.92** ⭐ |

> A **"crossing point"** is observed: DLinear dominates at Step 1 but AMR overtakes it by Step 15, confirming that AMR captures complex long-range dependencies better than linear models.

---

### Adaptive Mechanism Analysis

One of the key contributions is the **interpretability** of learned Alpha (α) weights:

#### Key Findings per Dataset

| Dataset | Behavior | Interpretation |
|---|---|---|
| **Electricity** | Layer 3 concentrates on Period≈8 (α≈0.57) | Model discovers the **3-shift industrial work cycle** (24h/3 = 8h) |
| **Traffic** | Layer 3 locks onto Period≈24 | Model learns **daily rush-hour seasonality** |
| **Exchange-Rate** | Flat α distribution across all layers (0.12–0.29) | No dominant cycle → **"diversification" strategy** to avoid overfitting |
| **Weather** | Layer 1: Period 24; Layer 2: Period 4; Layer 3: dual peak (2 & 24) | **Hierarchical multi-scale**: background trend → short oscillations → synthesis |

> Non-integer learned scales (e.g., σ = 7.95, 8.07, 23.92) prove the **differentiable resampling** is actively fine-tuning to match irregular natural cycles.

---

### Computational Efficiency

#### Model Complexity

| Model | GFLOPs | Parameters | Notes |
|---|---|---|---|
| **AMR** | 1.594 | 0.745M | 34% fewer FLOPs than Vanilla Transformer |
| Vanilla Transformer | 2.424 | 0.886M | Heaviest; most memory |
| LSTM | 0.211 | 0.159M | Lightweight; poor long-range modeling |
| PatchTST | 0.036 | 0.112M | Extremely light (patching + shared weights) |
| DLinear | ~0.0001 | 0.003M | Negligible (simple linear layer) |

#### Inference Latency (ms/batch, batch size = 32)

| Dataset | AMR | Transformer | PatchTST | LSTM | DLinear |
|---|---|---|---|---|---|
| Traffic | 19.71 | 53.83 | 1.69 | 0.68 | 0.58 |
| Electricity | 20.61 | 53.43 | 1.23 | 0.64 | 0.36 |
| Weather | 22.85 | 53.78 | 1.20 | 0.52 | 0.38 |
| Exchange-Rate | 22.55 | 53.71 | 1.26 | 0.52 | 0.36 |
| **Average** | **~21.4** | **~53.7** | **~1.3** | **~0.6** | **~0.4** |

> At ~20ms/batch, AMR processes ~1,600 samples/second — sufficient for real-world industrial deployment (traffic systems update every 5–15 min, weather every 10 min).

---

## ⚙️ Installation

### Requirements

```bash
Python >= 3.8
PyTorch >= 2.0
CUDA (recommended for training)
```

### Install Dependencies

```bash
git clone https://github.com/YOUR_USERNAME/AMR-Transformer.git
cd AMR-Transformer

pip install -r requirements.txt
```

**`requirements.txt`**
```
torch>=2.0.0
numpy>=1.23.0
pandas>=1.5.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
statsmodels>=0.13.0
torchinfo>=1.7.0
thop>=0.1.0
```

---

## 🚦 Usage

### 1. Prepare Datasets

Place the benchmark CSV files in your data directory:

```
data/
├── electricity.csv     # 26,304 timesteps × 321 clients (hourly kWh)
├── exchange_rate.csv   # 7,588 timesteps × 8 currencies (daily rates)
├── traffic.csv         # 17,544 timesteps × 862 sensors (hourly occupancy)
└── weather.csv         # 52,696 timesteps × 21 features (10-min meteorology)
```

Each CSV must have a `date` column as the index, with the target variable in the last column (`OT`).

### 2. Run Training (Notebook)

Open and run `Final_AMR_electricity.ipynb` in Google Colab or Jupyter:

```python
# Configure dataset path
PROJECT_PATH = 'data/'   # or your Google Drive path

# Key hyperparameters
INPUT_LEN  = 96     # Look-back window
OUTPUT_LEN = 15     # Forecast horizon
BATCH_SIZE = 32
COMMON_D_MODEL = 64
COMMON_NHEAD   = 4
NUM_LAYERS     = 3
NUM_EPOCHS     = 40
LEARNING_RATE  = 0.0005
PATIENCE       = 5
COMMON_DROPOUT = 0.2
```

To switch datasets, change the path:
```python
PROJECT_PATH = 'data/electricity/'    # or traffic/ weather/ exchange_rate/
```

### 3. Training All Models

```python
# Train all 5 models and save best checkpoints
MODEL_NAMES = ['amr', 'lstm', 'transformer', 'dlinear', 'patchtst']

for mname in MODEL_NAMES:
    # Initializes, trains with early stopping, saves best_model_{mname}.pt
    train_model(mname, train_loader, val_loader, device)
```

### 4. Evaluation

```python
# Evaluate on test set — reports MSE, MAE, RMSE, MAPE, SMAPE
# Also runs multi-horizon analysis (Step 1, 6, 10, 15)
# And generates visual prediction comparisons at sample index 150
evaluate_all_models(MODEL_NAMES, test_loader, scaler, device)
```

### 5. Visualize Adaptive Mechanism

```python
# Load trained AMR model and visualize alpha weights per layer
# Shows which resolution branches each layer prioritizes
plot_alpha_weights(model, test_loader, device)
```

---

## 📁 Project Structure

```
AMR-Transformer/
│
├── 📓 Final_AMR_electricity.ipynb   # Main experiment notebook (all models + all analyses)
│
├── 📂 data/
│   ├── electricity.csv              # UCI Electricity (26K rows, 321 clients)
│   ├── traffic.csv                  # Caltrans PeMS (17.5K rows, 862 sensors)
│   ├── weather.csv                  # MPI Weather (52K rows, 21 features)
│   └── exchange_rate.csv            # Exchange rates (7.6K rows, 8 currencies)
│
├── 📄 report                       # Generated output directory  
│
└── 📄 README.md                     # This file
```

---

## 🔬 Model Components Reference

### RevIN (Reversible Instance Normalization)
Normalizes each input instance independently (removes mean/std), then restores the original scale after prediction. Critical for handling non-stationary time series.

<div align="center">

![RevIN](assets/RevIN.png)
*RevIN normalization and denormalization flow*

</div>

### LearnableResampler1D
Uses 1D bilinear interpolation with a learnable `log_scale` parameter. The scale is constrained via `softplus` and `clamp` to ensure valid output lengths. Gradients flow back through the grid sampling operation to update `log_scale`.

<div align="center">

![LearnableResampler1D](assets/LearnableResampler1D.png)
*Differentiable 1D resampling via grid_sample — gradients flow to `log_scale`*

</div>

### AdaptiveRouter
A Query-Based Attention module where a learnable `Q_router` vector attends to the input sequence, extracting a global context vector `C`. An MLP + Softmax then produces the K mixing weights (α).

<div align="center">

![Adaptive Resolution Module](assets/Adaptive_Resolution_Module.png)
*ARM core: Learnable Query → Cross-Attention → Router MLP → Softmax → α weights*

</div>

### ContinuousScaleAwarePE
Extends the standard sinusoidal PE formula by multiplying position indices by the learned scale σ:
```
PE(pos, 2i)   = sin(pos·σ / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos·σ / 10000^(2i/d_model))
```

<div align="center">

![Continuous Scale-Aware Positional Encoding](assets/Continuous_Scale-Aware_Positional_Encoding.png)
*Scale-aware PE: positions are scaled by σ to preserve absolute time across resolutions*

</div>

### AdvancedResolutionAwareAttention
Aligns all K branches to length T via interpolation, computes unshared Q/K/V projections per branch, fuses them using α weights from ARM, then runs a single Multi-Head Attention on the unified representation.

<div align="center">

![Advanced Resolution-Aware Attention](assets/AdvancedResolutionAttention.png)
*Three-phase attention: alignment → α-guided QKV fusion → single MHA*

</div>

---

## 📚 Related Work

| Paper | Key Idea | Limitation addressed by AMR |
|---|---|---|
| [Transformer](https://arxiv.org/pdf/1706.03762) (2017) | Self-attention for sequence modeling | Fixed resolution; O(L²) complexity |
| [Informer](https://arxiv.org/pdf/2012.07436) (2021) | ProbSparse attention; O(L log L) | Fixed resolution; sparse selection |
| [Autoformer](https://arxiv.org/pdf/2106.13008) (2021) | Auto-Correlation via FFT | Fixed resolution |
| [Pyraformer](https://openreview.net/pdf?id=0EXmFzUn5I) (2022) | Pyramidal multi-resolution | Fixed hierarchy; pre-defined scales |
| [DLinear](https://arxiv.org/pdf/2205.13504) (2022) | Simple linear; Trend + Seasonality | No feature learning; poor on periodic |
| [PatchTST](https://arxiv.org/pdf/2211.14730) (2023) | Patching; Channel Independence | Fixed patch size P; single resolution |
| [MultiResFormer](https://arxiv.org/pdf/2311.18780) (2024) | Adaptive periods via FFT | Pre-computed; not fully end-to-end |
| **AMR-Transformer** (2025) | **Learnable scales; layer-wise adaptation** | — |

---

