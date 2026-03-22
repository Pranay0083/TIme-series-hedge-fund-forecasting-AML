# Multi-Horizon, Multi-Entity Anonymous Time-Series Forecasting

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

This repository encapsulates **Phase 1** of our research, primarily covering:
1. Deep Exploratory Data Analysis (EDA).
2. Theoretical and Mathematical foundation for model selection.
3. Creation of a robust preprocessing pipeline mitigating leakage and structural concept drift.

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

### Setup & Usage
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Read the theoretical breakdown in the `literature_review/research_paper/` directory.
