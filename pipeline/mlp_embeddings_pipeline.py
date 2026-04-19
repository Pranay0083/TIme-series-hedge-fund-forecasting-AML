#!/usr/bin/env python3
"""
MLP + Entity Embeddings Pipeline for Hedge Fund Time-Series Forecasting
=======================================================================

Implements a fast and high-performance baseline MLP with categorical embeddings.
- Embedding layers for categorical features (emb_dim = min(50, cardinality // 2 + 1))
- Numerical features concatenated directly with embeddings
- MLP blocks with Linear -> ReLU -> BatchNorm/Dropout -> Linear...

Trains on FULL training data and generates Kaggle submission.
"""

import gc
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ============================================================================
# REPRODUCIBILITY
# ============================================================================
SEED = 42

def set_seed(seed: int = SEED):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(SEED)

# ============================================================================
# PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_PATH = DATA_DIR / "train.parquet"
TEST_PATH = DATA_DIR / "test.parquet"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DEVICE
# ============================================================================
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# ============================================================================
# LOGGING
# ============================================================================
def setup_logging(log_file: str = "mlp_embeddings_pipeline.log") -> logging.Logger:
    """Configure logging to console and file."""
    logger = logging.getLogger("MLPEmbeddings")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    fh = logging.FileHandler(LOGS_DIR / log_file, mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

# ============================================================================
# COMPETITION METRIC
# ============================================================================
def weighted_rmse_score(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> float:
    """
    Competition scoring metric:
        score = sqrt(1 - min(max(sum(w*(y-yhat)^2) / sum(w*y^2), 0), 1))
    """
    y_true, y_pred, w = np.asarray(y_true), np.asarray(y_pred), np.asarray(w)
    denom = np.sum(w * y_true ** 2)
    if denom <= 0:
        return 0.0
    numer = np.sum(w * (y_true - y_pred) ** 2)
    ratio = np.clip(numer / denom, 0.0, 1.0)
    return float(np.sqrt(1.0 - ratio))

# ============================================================================
# DATA LOADING & PREPROCESSING (Reused from FT-Transformer)
# ============================================================================
NOISE_FEATURES = ['feature_b', 'feature_c', 'feature_d', 'feature_e', 'feature_f', 'feature_g']
CATEGORICAL_STRING_COLS = ['code', 'sub_code', 'sub_category']
CATEGORICAL_INT_COLS = ['feature_a', 'horizon', 'feature_ch']
META_COLS = ['id', 'ts_index', 'weight', 'y_target']

def identify_features(columns: List[str]) -> Tuple[List[str], List[str]]:
    exclude = set(META_COLS) | set(NOISE_FEATURES) | set(CATEGORICAL_STRING_COLS)
    cat_cols = [c for c in CATEGORICAL_INT_COLS if c in columns]
    cat_cols += list(CATEGORICAL_STRING_COLS)
    num_cols = [c for c in columns if c not in exclude and c not in cat_cols]
    return cat_cols, num_cols

class LabelEncoderDict:
    def __init__(self):
        self.encoders: Dict[str, Dict] = {}
        self.n_categories: Dict[str, int] = {}

    def fit(self, df: pd.DataFrame, cat_cols: List[str]):
        for col in cat_cols:
            uniques = df[col].unique()
            mapping = {val: idx for idx, val in enumerate(sorted(uniques, key=str))}
            self.encoders[col] = mapping
            self.n_categories[col] = len(mapping) + 1

    def transform(self, df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
        df = df.copy()
        for col in cat_cols:
            mapping = self.encoders[col]
            unseen_idx = len(mapping)
            df[col] = df[col].map(mapping).fillna(unseen_idx).astype(np.int64)
        return df

class NumericalNormalizer:
    def __init__(self):
        self.means: Optional[pd.Series] = None
        self.stds: Optional[pd.Series] = None

    def fit(self, df: pd.DataFrame, num_cols: List[str]):
        self.means = df[num_cols].mean()
        self.stds = df[num_cols].std().replace(0, 1.0)

    def transform(self, df: pd.DataFrame, num_cols: List[str]) -> pd.DataFrame:
        df = df.copy()
        df[num_cols] = (df[num_cols] - self.means[num_cols]) / self.stds[num_cols]
        return df

def load_and_preprocess(
    logger: logging.Logger
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str], LabelEncoderDict, NumericalNormalizer]:
    logger.info("=" * 70)
    logger.info("STEP 1 & 2: LOADING DATA")
    logger.info("=" * 70)
    
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)
    
    # Drop noise
    for col in NOISE_FEATURES:
        if col in train_df.columns:
            train_df.drop(columns=[col], inplace=True)
        if col in test_df.columns:
            test_df.drop(columns=[col], inplace=True)

    train_df = train_df[train_df['weight'] > 0].reset_index(drop=True)
    cat_cols, num_cols = identify_features(train_df.columns.tolist())

    train_df[num_cols] = train_df[num_cols].fillna(0).astype(np.float32)
    test_df[num_cols] = test_df[num_cols].fillna(0).astype(np.float32)

    for col in CATEGORICAL_STRING_COLS:
        train_df[col] = train_df[col].fillna("__MISSING__").astype(str)
        test_df[col] = test_df[col].fillna("__MISSING__").astype(str)

    logger.info("STEP 3: ENCODING CATEGORICALS")
    le = LabelEncoderDict()
    le.fit(train_df, cat_cols)
    train_df = le.transform(train_df, cat_cols)
    test_df = le.transform(test_df, cat_cols)

    logger.info("NORMALIZING NUMERICAL FEATURES")
    normalizer = NumericalNormalizer()
    normalizer.fit(train_df, num_cols)
    train_df = normalizer.transform(train_df, num_cols)
    test_df = normalizer.transform(test_df, num_cols)

    train_df[num_cols] = train_df[num_cols].fillna(0).astype(np.float32)
    test_df[num_cols] = test_df[num_cols].fillna(0).astype(np.float32)

    return train_df, test_df, cat_cols, num_cols, le, normalizer

# ============================================================================
# PYTORCH DATASET
# ============================================================================
class TabularDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cat_cols: List[str], num_cols: List[str], target_col: Optional[str] = None):
        self.cat_data = torch.tensor(df[cat_cols].values, dtype=torch.long)
        self.num_data = torch.tensor(df[num_cols].values, dtype=torch.float32)
        if target_col and target_col in df.columns:
            self.targets = torch.tensor(df[target_col].values, dtype=torch.float32)
        else:
            self.targets = None
        self.n_samples = len(df)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        item = {'cat': self.cat_data[idx], 'num': self.num_data[idx]}
        if self.targets is not None:
            item['target'] = self.targets[idx]
        return item

# ============================================================================
# MLP + EMBEDDINGS MODEL
# ============================================================================
class MLPWithEmbeddings(nn.Module):
    """
    MLP Model with Entity Embeddings for Tabular Data.
    Architecture:
      - Categorical columns mapped to Embeddings (dim = min(50, card // 2 + 1))
      - Concatenated with numerical features
      - MLP Layers: Linear -> ReLU -> BatchNorm/Dropout...
    """
    def __init__(
        self,
        cat_cardinalities: List[int],
        n_num_features: int,
        dropout: float = 0.25,
    ):
        super().__init__()
        
        self.embeddings = nn.ModuleList()
        total_emb_dim = 0
        
        for card in cat_cardinalities:
            emb_dim = min(50, (card // 2) + 1)
            self.embeddings.append(nn.Embedding(card, emb_dim))
            total_emb_dim += emb_dim
            
        mlp_input_dim = total_emb_dim + n_num_features
        
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(128, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights_fn)

    @staticmethod
    def _init_weights_fn(module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.01)

    def forward(self, cat_features: torch.Tensor, num_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cat_features: (B, n_cat) long tensor
            num_features: (B, n_num) float tensor
        Returns:
            (B,) scalar predictions
        """
        emb_outputs = []
        if self.embeddings and cat_features is not None:
            for i, emb_layer in enumerate(self.embeddings):
                emb_outputs.append(emb_layer(cat_features[:, i]))
        
        if emb_outputs:
            cat_out = torch.cat(emb_outputs, dim=1)
            x = torch.cat([cat_out, num_features], dim=1)
        else:
            x = num_features
            
        out = self.mlp(x)
        return out.squeeze(-1)

# ============================================================================
# TRAINING PIPELINE
# ============================================================================
class TrainingConfig:
    batch_size: int = 2048
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 10
    grad_clip_norm: float = 1.0
    dropout: float = 0.25
    warmup_epochs: int = 1
    min_lr: float = 1e-6

def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr=1e-6):
    base_lr = optimizer.defaults['lr']
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr / base_lr, cosine_decay)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_one_epoch(
    model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR, grad_clip_norm: float,
    device: torch.device, epoch: int, logger: logging.Logger
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        cat, num, target = batch['cat'].to(device), batch['num'].to(device), batch['target'].to(device)

        optimizer.zero_grad()
        pred = model(cat, num)
        loss = F.mse_loss(pred, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1

        if batch_idx % 200 == 0:
            current_lr = scheduler.get_last_lr()[0]
            logger.info(
                f"  Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | "
                f"Loss: {loss.item():.6f} | LR: {current_lr:.2e}"
            )

    return total_loss / max(n_batches, 1)

@torch.no_grad()
def predict(model: nn.Module, dataloader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    predictions = []
    for batch in dataloader:
        cat, num = batch['cat'].to(device), batch['num'].to(device)
        pred = model(cat, num)
        predictions.append(pred.cpu().numpy())
    return np.concatenate(predictions, axis=0)

def train_mlp_pipeline(
    train_df: pd.DataFrame, test_df: pd.DataFrame, cat_cols: List[str], num_cols: List[str],
    le: LabelEncoderDict, logger: logging.Logger, config: TrainingConfig
):
    logger.info("=" * 70)
    logger.info("STEP 4 & 5: BUILDING & TRAINING MLP MODEL")
    logger.info("=" * 70)

    train_dataset = TabularDataset(train_df, cat_cols, num_cols, target_col='y_target')
    test_dataset = TabularDataset(test_df, cat_cols, num_cols, target_col=None)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=0, pin_memory=(DEVICE.type == 'cuda'), drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size * 2, shuffle=False,
        num_workers=0, pin_memory=(DEVICE.type == 'cuda')
    )

    cat_cardinalities = [le.n_categories[col] for col in cat_cols]
    
    model = MLPWithEmbeddings(
        cat_cardinalities=cat_cardinalities,
        n_num_features=len(num_cols),
        dropout=config.dropout,
    ).to(DEVICE)

    logger.info(f"Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    total_steps = len(train_loader) * config.epochs
    warmup_steps = len(train_loader) * config.warmup_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, config.min_lr)

    logger.info("TRAINING START")
    best_loss = float('inf')
    
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()
        avg_loss = train_one_epoch(model, train_loader, optimizer, scheduler, config.grad_clip_norm, DEVICE, epoch, logger)
        logger.info(f"Epoch {epoch}/{config.epochs} | Loss: {avg_loss:.6f} | Time: {time.time()-epoch_start:.1f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), OUTPUTS_DIR / "mlp_best.pt")

    logger.info(f"Training complete! Best training loss: {best_loss:.6f}")
    
    model.load_state_dict(torch.load(OUTPUTS_DIR / "mlp_best.pt", weights_only=True))
    
    logger.info("STEP 6: INFERENCE ON TEST DATA")
    return predict(model, test_loader, DEVICE), model

# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main():
    logger = setup_logging()
    logger.info("=" * 70)
    logger.info("MLP + EMBEDDINGS PIPELINE FOR HEDGE FUND FORECASTING")
    
    train_df, test_df, cat_cols, num_cols, le, normalizer = load_and_preprocess(logger)

    config = TrainingConfig()
    test_preds, model = train_mlp_pipeline(train_df, test_df, cat_cols, num_cols, le, logger, config)

    logger.info("STEP 7: GENERATING SUBMISSION FILE")
    submission = pd.DataFrame({'id': test_df['id'].values, 'y_target': test_preds})
    
    if submission['y_target'].isna().sum() > 0:
        submission['y_target'] = submission['y_target'].fillna(0.0)

    submission_path = OUTPUTS_DIR / "mlp_embeddings_submission.csv"
    submission.to_csv(submission_path, index=False)
    
    logger.info(f"Submission saved to: {submission_path}")
    logger.info("DONE!")

if __name__ == "__main__":
    main()
