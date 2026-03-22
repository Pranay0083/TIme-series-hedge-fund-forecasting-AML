# Time-Series Hedge Fund Forecasting: Multi-Horizon, Multi-Entity Anonymous Time-Series Forecasting

> **Phase 1 Project Report**  
> **Course:** Advanced Machine Learning & Deep Learning (SEM -6)  
> **Institution:** Rishihood University  
> **Team:** Bias and Variance  

## Authors
- **Vishesh Rao** ([vishesh.23csai@gmail.com](mailto:vishesh.23csai@nst.rishihood.edu.in))
- **Pranay Vishwakarma** ([pranay.v23csai@nast.rishihood.edu.in](mailto:pranay.v23csai@nst.rishihood.edu.in))

---

## 📌 Project Overview
This project focuses on predicting synthetic financial instrument returns across multiple prediction horizons ($h \in \{1, 3, 10, 25\}$) based on the **Kaggle: Hedge Fund — Time Series Forecasting** competition. The dataset is thoroughly anonymized, involving an expanding universe of assets, temporal non-stationarity, and severe noise-to-signal ratios typical of quantitative finance. 

Our goal is to create a structured, production-ready time series forecasting solution. This repository encapsulates **Phase 1** of our research, primarily covering:
1. Deep Exploratory Data Analysis (EDA).
2. Theoretical and Mathematical foundation for model selection.
3. Creation of a robust preprocessing pipeline mitigating leakage and structural concept drift.

---

## 🎯 Problem Statement

Predict future values for time series data across multiple forecast horizons (1, 3, 10, 25). Models are evaluated on out-of-sample weighted RMSE score.

**Scoring Formula:**

$$Score = \sqrt{1 - \min\left(\max\left(\frac{\sum w_i(y_i - \hat{y}_i)^2}{\sum w_i y_i^2}, 0\right), 1\right)}$$

### Dataset

| Column | Description |
|--------|-------------|
| `id` | Unique key: `code__sub_code__sub_category__horizon__ts_index` |
| `horizon` | Forecast horizon: 1, 3, 10, 25 |
| `weight` | Row weight for loss (**DO NOT USE AS FEATURE**) |
| `feature_a` to `feature_ch` | 86 anonymized features |

### 📊 Data Horizons

| Horizon | Description | Rows |
|---------|-------------|------|
| 1 | Short-term | ~1.4M |
| 3 | Medium-short | ~1.4M |
| 10 | Medium-long | ~1.3M |
| 25 | Long-term | ~1.2M |

### Tips

- Focus on recent periods (recency weighting)
- Low signal-to-noise ratio is a key challenge
- Process is non-stationary over time
- External data is forbidden

---

## 🏗️ Architecture & Codebase Structure
The repository is segmented into modular environments enabling structured scientific exploration and reproducible model pipelines.

```text
├── literature_review/
│   └── research_paper/       # Phase 1 IEEE LaTeX & Theoretical reports
├── notebooks/                
│   ├── 01-EDA.ipynb          # Baseline target & distribution analysis
│   ├── 02-Advanced-EDA.ipynb # Feature interactions, drift, & redundancy 
│   └── 03-Preprocessing.ipynb# Pipeline prototyping 
├── pipeline/                 # Production Python ML modules
│   ├── preprocess.py         # Leakage-proof data transformations
│   ├── preprocess_config.json# Dimension pruning & configuration rules
│   ├── timeseries_split.py   # Expanding window GroupTimeSeries validation
│   └── *_pipeline.py         # Advanced tree-based modeling blueprints
└── requirements.txt          # Python environments
```

## 🧠 Key Phase 1 Insights

### 1. The Financial Data Problem
The task is a sequence-to-scalar cross-sectional prediction. We identified severe violations of classical I.I.D. statistical models due to an expanding entity universe and drifting feature covariance matrices. 

### 2. Signal vs. Noise
- **Fat Tails:** The target variable exhibits extreme leptokurtosis (kurtosis ~290), rendering standard Mean Squared Error (MSE) heavily biased toward outliers. Robust loss functions (Huber/MAE) are mapped securely.
- **Injected Noise:** Out of 85+ features, variables `feature_b` to `feature_g` behave identically but carry zero predictive correlation with the target. 
- **Horizon Normalization:** The target variance scales with prediction horizons ($\sqrt{h}$). The pipeline corrects this by standardizing cross-sectional variances per horizon. 

### 3. Proposed Engineering Pipeline
To ensure mathematical rigor, our preprocessing module (`preprocess.py`) performs:
1. Explicit **dimensional pruning** via hierarchical clustering.
2. Handling sparse categorical entities (`sub_code`) natively using **grouped median imputation** while preserving missing values as Boolean vectors (`_is_missing`).
3. Implementation of **Time-Aware validation** (`GroupTimeSeriesSplit` based on `ts_index`) to eliminate temporal peeking.

## 🚀 Next Steps (Phase 2)
Moving forward, `Team Bias and Variance` will dive into heavy modeling:
- Deep hyperparameter optimization using **Optuna** on localized cuts (e.g., specific horizons per category).
- Weight decay implementations to balance extreme financial sample weights.
- Sequence-to-Sequence models (LSTMs/Transformers) to map auto-correlative temporal momentum.

---

## 📁 Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/architecture.md) | Project structure & folder organization |
| [Setup](docs/setup.md) | Environment setup instructions |

---

## 🚀 Setup & Quick Start

```bash
# Clone the repository
git clone <your-repository-url>
cd TIme-series-hedge-fund-forecasting-AML

# Setup environment
conda create -n venv python=3.11 && conda activate venv
pip install -r requirements.txt

# Download & prepare data
kaggle competitions download -c ts-forecasting
unzip ts-forecasting.zip -d data/raw/combined/
python src/data/horizon_split.py
```

*For theoretical breakdown, refer to the `literature_review/research_paper/` directory.*
