#!/usr/bin/env python3
"""
FT-Transformer Pipeline for Hedge Fund Time-Series Forecasting
================================================================

Implements the Feature Tokenizer + Transformer architecture for tabular data.
- Feature Tokenizer: Embedding layers for categorical, linear projection for numerical
- Transformer Encoder: Multi-head self-attention + feed-forward blocks
- Output Head: FC layers → scalar prediction

Trains on FULL training data (no validation split) and generates Kaggle submission.

Reference: Gorishniy et al., "Revisiting Deep Learning Models for Tabular Data" (NeurIPS 2021)
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
def setup_logging(log_file: str = "ft_transformer_pipeline.log") -> logging.Logger:
    """Configure logging to console and file."""
    logger = logging.getLogger("FTTransformer")
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
# DATA LOADING & PREPROCESSING
# ============================================================================

# Columns known to be noise/non-predictive from EDA (README notes feature_b-g are noise)
NOISE_FEATURES = ['feature_b', 'feature_c', 'feature_d', 'feature_e', 'feature_f', 'feature_g']

# Categorical columns: string-typed identifiers + integer-coded categoricals
CATEGORICAL_STRING_COLS = ['code', 'sub_code', 'sub_category']
CATEGORICAL_INT_COLS = ['feature_a', 'horizon']
# feature_ch is int64 but it could be a high-cardinality categorical or numeric.
# We'll treat it as categorical since it's int-typed and the README groups it with features.

META_COLS = ['id', 'ts_index', 'weight', 'y_target']


def identify_features(columns: List[str]) -> Tuple[List[str], List[str]]:
    """
    Separate columns into categorical and numerical feature lists.

    Returns:
        cat_cols: columns to embed as categorical tokens
        num_cols: columns to project as numerical tokens
    """
    exclude = set(META_COLS) | set(NOISE_FEATURES) | set(CATEGORICAL_STRING_COLS)

    cat_cols = list(CATEGORICAL_INT_COLS)  # horizon, feature_a
    # Add string categoricals (will be label-encoded first)
    cat_cols += list(CATEGORICAL_STRING_COLS)

    num_cols = []
    for c in columns:
        if c in exclude or c in cat_cols:
            continue
        num_cols.append(c)

    return cat_cols, num_cols


class LabelEncoderDict:
    """Fit label encoders for categorical columns, handles unseen labels."""

    def __init__(self):
        self.encoders: Dict[str, Dict] = {}
        self.n_categories: Dict[str, int] = {}

    def fit(self, df: pd.DataFrame, cat_cols: List[str]):
        for col in cat_cols:
            uniques = df[col].unique()
            mapping = {val: idx for idx, val in enumerate(sorted(uniques, key=str))}
            self.encoders[col] = mapping
            self.n_categories[col] = len(mapping) + 1  # +1 for unseen

    def transform(self, df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
        df = df.copy()
        for col in cat_cols:
            mapping = self.encoders[col]
            unseen_idx = len(mapping)  # last index for OOV
            df[col] = df[col].map(mapping).fillna(unseen_idx).astype(np.int64)
        return df


class NumericalNormalizer:
    """Fit per-feature mean/std from training data; transform with same stats."""

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
    """
    Load train/test parquet files, identify feature types, encode categoricals,
    normalize numericals. Returns processed DataFrames and fitted transformers.
    """
    logger.info("=" * 70)
    logger.info("STEP 1 & 2: LOADING DATA")
    logger.info("=" * 70)
    logger.info(f"Train path: {TRAIN_PATH}")
    logger.info(f"Test path:  {TEST_PATH}")

    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    logger.info(f"Train shape: {train_df.shape}")
    logger.info(f"Test shape:  {test_df.shape}")

    # Verify target column
    assert 'y_target' in train_df.columns, "Missing y_target in train"
    assert 'y_target' not in test_df.columns, "y_target should not be in test"

    # Drop noise features
    for col in NOISE_FEATURES:
        if col in train_df.columns:
            train_df.drop(columns=[col], inplace=True)
        if col in test_df.columns:
            test_df.drop(columns=[col], inplace=True)
    logger.info(f"Dropped noise features: {NOISE_FEATURES}")

    # Drop zero-weight rows from training
    n_before = len(train_df)
    train_df = train_df[train_df['weight'] > 0].reset_index(drop=True)
    logger.info(f"Dropped {n_before - len(train_df)} zero-weight rows. Remaining: {len(train_df)}")

    # Identify feature types
    cat_cols, num_cols = identify_features(train_df.columns.tolist())
    logger.info(f"\nCategorical features ({len(cat_cols)}): {cat_cols}")
    logger.info(f"Numerical features ({len(num_cols)}): {num_cols[:10]}... ({len(num_cols)} total)")

    # Fill NaN in numerical features with 0 (simple imputation for transformer)
    train_df[num_cols] = train_df[num_cols].fillna(0).astype(np.float32)
    test_df[num_cols] = test_df[num_cols].fillna(0).astype(np.float32)

    # Fill NaN in categorical string columns
    for col in CATEGORICAL_STRING_COLS:
        train_df[col] = train_df[col].fillna("__MISSING__").astype(str)
        test_df[col] = test_df[col].fillna("__MISSING__").astype(str)

    # Fit label encoders on TRAIN, transform both
    logger.info("\nSTEP 3: ENCODING CATEGORICALS")
    le = LabelEncoderDict()
    le.fit(train_df, cat_cols)
    train_df = le.transform(train_df, cat_cols)
    test_df = le.transform(test_df, cat_cols)

    for col in cat_cols:
        logger.info(f"  {col}: {le.n_categories[col]} categories (incl. unseen)")

    # Normalize numerical features
    logger.info("\nNORMALIZING NUMERICAL FEATURES")
    normalizer = NumericalNormalizer()
    normalizer.fit(train_df, num_cols)
    train_df = normalizer.transform(train_df, num_cols)
    test_df = normalizer.transform(test_df, num_cols)

    # Final NaN cleanup after normalization
    train_df[num_cols] = train_df[num_cols].fillna(0).astype(np.float32)
    test_df[num_cols] = test_df[num_cols].fillna(0).astype(np.float32)

    logger.info(f"\nFinal train shape: {train_df.shape}")
    logger.info(f"Final test shape:  {test_df.shape}")

    return train_df, test_df, cat_cols, num_cols, le, normalizer


# ============================================================================
# PYTORCH DATASET
# ============================================================================
class TabularDataset(Dataset):
    """PyTorch Dataset for tabular data with categorical + numerical features."""

    def __init__(
        self,
        df: pd.DataFrame,
        cat_cols: List[str],
        num_cols: List[str],
        target_col: Optional[str] = None,
    ):
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
        item = {
            'cat': self.cat_data[idx],
            'num': self.num_data[idx],
        }
        if self.targets is not None:
            item['target'] = self.targets[idx]
        return item


# ============================================================================
# FT-TRANSFORMER MODEL
# ============================================================================

class NumericalTokenizer(nn.Module):
    """
    Projects each numerical feature into d_model-dimensional embedding space.
    Each feature gets its own linear projection (weight + bias).
    """

    def __init__(self, n_num_features: int, d_model: int):
        super().__init__()
        self.n_num_features = n_num_features
        self.d_model = d_model
        # Per-feature weight and bias
        self.weight = nn.Parameter(torch.empty(n_num_features, d_model))
        self.bias = nn.Parameter(torch.empty(n_num_features, d_model))
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = self.weight.shape[0]
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_num_features) float tensor
        Returns:
            (batch, n_num_features, d_model) token embeddings
        """
        # x: (B, N) -> (B, N, 1) * (N, D) -> (B, N, D)
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class CategoricalTokenizer(nn.Module):
    """
    Embedding layers for categorical features.
    Each categorical feature has its own embedding table.
    """

    def __init__(self, cat_cardinalities: List[int], d_model: int):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(n_cat, d_model) for n_cat in cat_cardinalities
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_cat_features) long tensor of category indices
        Returns:
            (batch, n_cat_features, d_model) token embeddings
        """
        tokens = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        return torch.stack(tokens, dim=1)


class TransformerBlock(nn.Module):
    """Single Transformer encoder block: LayerNorm → MHA → Residual → LayerNorm → FFN → Residual."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + self.dropout1(h)

        h = self.norm2(x)
        h = self.ffn(h)
        x = x + h

        return x


class FTTransformer(nn.Module):
    """
    Feature Tokenizer + Transformer for tabular data.

    Architecture:
        1. Feature Tokenizer:
            - CategoricalTokenizer: embedding per categorical feature
            - NumericalTokenizer:  linear projection per numerical feature
        2. [CLS] token prepended to token sequence
        3. Transformer Encoder: N layers of MHA + FFN
        4. Output Head: [CLS] representation → MLP → scalar prediction
    """

    def __init__(
        self,
        cat_cardinalities: List[int],
        n_num_features: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff_multiplier: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_cat = len(cat_cardinalities)
        self.n_num = n_num_features
        self.n_tokens = self.n_cat + self.n_num + 1  # +1 for CLS

        # Feature tokenizers
        if self.n_cat > 0:
            self.cat_tokenizer = CategoricalTokenizer(cat_cardinalities, d_model)
        else:
            self.cat_tokenizer = None

        if self.n_num > 0:
            self.num_tokenizer = NumericalTokenizer(n_num_features, d_model)
        else:
            self.num_tokenizer = None

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        # Transformer encoder
        d_ff = d_model * d_ff_multiplier
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # Output head: [CLS] → prediction
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        # Initialize weights
        self.apply(self._init_weights_fn)

    @staticmethod
    def _init_weights_fn(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, cat_features: torch.Tensor, num_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cat_features: (B, n_cat) long tensor
            num_features: (B, n_num) float tensor
        Returns:
            (B,) predictions
        """
        batch_size = num_features.shape[0]
        tokens = []

        # Categorical tokens
        if self.cat_tokenizer is not None and cat_features is not None:
            cat_tokens = self.cat_tokenizer(cat_features)  # (B, n_cat, D)
            tokens.append(cat_tokens)

        # Numerical tokens
        if self.num_tokenizer is not None and num_features is not None:
            num_tokens = self.num_tokenizer(num_features)  # (B, n_num, D)
            tokens.append(num_tokens)

        # Concatenate all feature tokens
        x = torch.cat(tokens, dim=1)  # (B, n_cat + n_num, D)

        # Prepend [CLS] token
        cls = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, D)
        x = torch.cat([cls, x], dim=1)  # (B, 1 + n_cat + n_num, D)

        # Transformer encoder layers
        for layer in self.transformer_layers:
            x = layer(x)

        # Final layer norm
        x = self.final_norm(x)

        # Take [CLS] token output
        cls_output = x[:, 0, :]  # (B, D)

        # Prediction head
        out = self.head(cls_output)  # (B, 1)
        return out.squeeze(-1)  # (B,)


# ============================================================================
# TRAINING PIPELINE
# ============================================================================

class TrainingConfig:
    """Hyperparameters for training."""
    # Model architecture
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 3
    dropout: float = 0.2

    # Training
    batch_size: int = 1024
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 15
    grad_clip_norm: float = 1.0

    # Scheduler
    warmup_epochs: int = 2
    min_lr: float = 1e-6


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr: float = 1e-6,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine annealing with linear warmup."""
    base_lr = optimizer.defaults['lr']

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr / base_lr, cosine_decay)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    grad_clip_norm: float,
    device: torch.device,
    epoch: int,
    logger: logging.Logger,
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        cat = batch['cat'].to(device)
        num = batch['num'].to(device)
        target = batch['target'].to(device)

        optimizer.zero_grad()
        pred = model(cat, num)

        loss = F.mse_loss(pred, target)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1

        if batch_idx % 500 == 0:
            current_lr = scheduler.get_last_lr()[0]
            logger.info(
                f"  Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | "
                f"Loss: {loss.item():.6f} | LR: {current_lr:.2e}"
            )

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss


@torch.no_grad()
def predict(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Run inference and return predictions."""
    model.eval()
    predictions = []

    for batch in dataloader:
        cat = batch['cat'].to(device)
        num = batch['num'].to(device)

        pred = model(cat, num)
        predictions.append(pred.cpu().numpy())

    return np.concatenate(predictions, axis=0)


def train_ft_transformer(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cat_cols: List[str],
    num_cols: List[str],
    le: LabelEncoderDict,
    logger: logging.Logger,
    config: Optional[TrainingConfig] = None,
) -> Tuple[np.ndarray, FTTransformer]:
    """
    Full training pipeline:
    1. Create datasets and dataloaders
    2. Build FT-Transformer model
    3. Train for N epochs on full training data
    4. Generate test predictions
    """
    if config is None:
        config = TrainingConfig()

    logger.info("=" * 70)
    logger.info("STEP 4 & 5: BUILDING & TRAINING FT-TRANSFORMER")
    logger.info("=" * 70)

    # --- Dataset & DataLoader ---
    logger.info("Creating PyTorch datasets...")
    train_dataset = TabularDataset(train_df, cat_cols, num_cols, target_col='y_target')
    test_dataset = TabularDataset(test_df, cat_cols, num_cols, target_col=None)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE.type == 'cuda'),
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == 'cuda'),
    )

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Test batches:  {len(test_loader)}")

    # --- Model ---
    cat_cardinalities = [le.n_categories[col] for col in cat_cols]
    logger.info(f"\nCategory cardinalities: {dict(zip(cat_cols, cat_cardinalities))}")
    logger.info(f"Numerical features: {len(num_cols)}")

    model = FTTransformer(
        cat_cardinalities=cat_cardinalities,
        n_num_features=len(num_cols),
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        dropout=config.dropout,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"\nModel architecture:")
    logger.info(f"  d_model = {config.d_model}")
    logger.info(f"  n_heads = {config.n_heads}")
    logger.info(f"  n_layers = {config.n_layers}")
    logger.info(f"  dropout = {config.dropout}")
    logger.info(f"  Total parameters: {n_params:,}")
    logger.info(f"  Device: {DEVICE}")

    # --- Optimizer & Scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    total_steps = len(train_loader) * config.epochs
    warmup_steps = len(train_loader) * config.warmup_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps, config.min_lr
    )

    logger.info(f"\nOptimizer: AdamW (lr={config.lr}, wd={config.weight_decay})")
    logger.info(f"Scheduler: Cosine with {warmup_steps} warmup steps / {total_steps} total")
    logger.info(f"Grad clip: {config.grad_clip_norm}")
    logger.info(f"Epochs: {config.epochs}")

    # --- Training Loop ---
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING START")
    logger.info("=" * 70)

    best_loss = float('inf')
    start_time = time.time()

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()

        avg_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            config.grad_clip_norm, DEVICE, epoch, logger
        )

        epoch_time = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()[0]

        logger.info(
            f"Epoch {epoch:2d}/{config.epochs} | "
            f"Loss: {avg_loss:.6f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {epoch_time:.1f}s"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save best model
            torch.save(model.state_dict(), OUTPUTS_DIR / "ft_transformer_best.pt")
            logger.info(f"  ↳ New best loss! Model saved.")

    total_time = time.time() - start_time
    logger.info(f"\nTraining complete! Total time: {total_time:.1f}s")
    logger.info(f"Best training loss: {best_loss:.6f}")

    # Load best model for inference
    model.load_state_dict(torch.load(OUTPUTS_DIR / "ft_transformer_best.pt", weights_only=True))

    # --- Inference on Test ---
    logger.info("\n" + "=" * 70)
    logger.info("STEP 6: INFERENCE ON TEST DATA")
    logger.info("=" * 70)

    test_preds = predict(model, test_loader, DEVICE)
    logger.info(f"Test predictions shape: {test_preds.shape}")
    logger.info(f"Test predictions stats: mean={test_preds.mean():.6f}, "
                f"std={test_preds.std():.6f}, "
                f"min={test_preds.min():.6f}, max={test_preds.max():.6f}")

    return test_preds, model


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main():
    """Run the full FT-Transformer pipeline."""
    logger = setup_logging()

    logger.info("=" * 70)
    logger.info("FT-TRANSFORMER PIPELINE FOR HEDGE FUND FORECASTING")
    logger.info("=" * 70)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Seed: {SEED}")

    # ---- Step 1-3: Load & Preprocess ----
    train_df, test_df, cat_cols, num_cols, le, normalizer = load_and_preprocess(logger)

    # ---- Step 4-6: Train & Predict ----
    config = TrainingConfig()
    test_preds, model = train_ft_transformer(
        train_df, test_df, cat_cols, num_cols, le, logger, config
    )

    # ---- Step 7: Generate Submission ----
    logger.info("\n" + "=" * 70)
    logger.info("STEP 7: GENERATING SUBMISSION FILE")
    logger.info("=" * 70)

    submission = pd.DataFrame({
        'id': test_df['id'].values,
        'y_target': test_preds,
    })

    # Verify no NaNs
    n_nan = submission['y_target'].isna().sum()
    if n_nan > 0:
        logger.warning(f"Found {n_nan} NaN predictions! Filling with 0.")
        submission['y_target'] = submission['y_target'].fillna(0.0)

    submission_path = OUTPUTS_DIR / "ft_transformer_submission.csv"
    submission.to_csv(submission_path, index=False)

    logger.info(f"Submission saved to: {submission_path}")
    logger.info(f"Submission rows: {len(submission):,}")
    logger.info(f"Submission columns: {submission.columns.tolist()}")
    logger.info(f"\nSubmission head:\n{submission.head(10)}")

    # ---- Summary ----
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Train samples:     {len(train_df):,}")
    logger.info(f"  Test samples:      {len(test_df):,}")
    logger.info(f"  Categorical feats: {len(cat_cols)}")
    logger.info(f"  Numerical feats:   {len(num_cols)}")
    logger.info(f"  Model params:      {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"  Epochs trained:    {config.epochs}")
    logger.info(f"  Submission file:   {submission_path}")
    logger.info(f"  End time:          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)
    logger.info("DONE!")

    # Cleanup
    del train_df, test_df, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
