import sys
import os
import gc
import json
import logging
import pickle
import tempfile
import random
from datetime import datetime
import warnings

import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)
import shap
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast  # [OPT] Mixed precision imports
from captum.attr import IntegratedGradients
from google.cloud import storage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ASSETS, BUCKET, WINDOWS

warnings.filterwarnings("ignore", category=UserWarning)

# GCP credentials: bridges the gap between gcloud CLI auth and the Python
# google-cloud-storage library, which looks for ADC not the CLI session.
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/workspace/maic/gcp-key.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# [OPT] TF32: enables tensor core acceleration on all matmul ops — one line, free speed
torch.set_float32_matmul_precision('high')

FEATURES_PREFIX = "v2/features/"
LABELS_PREFIX   = "v2/labels/"
RESULTS_PREFIX  = "v2/results_run1/"

# [OPT] GCS cache: persistent volume — eliminates redundant downloads across folds.
# With expanding windows, fold 4 would otherwise re-download everything from folds 1-3.
CACHE_DIR = "/workspace/maic/data_cache"

# Stage cache: persistent volume for model-level resume state.
# Stores probs + preds + metrics per model per mode per fold.
# Survives pod restarts — unlike /tmp which is ephemeral.
STAGE_CACHE_DIR = "/workspace/maic/stage_cache"

SEQ_LENGTH       = 60
BATCH_SIZE       = 2048
CNN_BATCH_SIZE   = 4096  # [OPT] was 256 — 4ch × 60×60 GAF = ~236MB/batch, well within 34GB VRAM
EPOCHS           = 20
PURGE_ROWS       = 30
EARLY_STOP_PATIENCE = 5
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# [OPT] Workers: 16 safe because sklearn/XGBoost finish before DataLoaders spin up,
# leaving all 28 vCPUs free during the PyTorch phase. Leave 2 for OS + main process.
NUM_WORKERS = min(16, os.cpu_count() - 2)

N_CLASSES          = 3
BINARY_STRESS_LABEL = 2

GAF_FEATURES = [
    "RV_300s",
    "OFI_300s",
    "Kyle_lambda_300s",
    "intensity_300s",
]

COMMON_FEATURES = [
    "price", "volume", "rv",
    "OFI_10s", "TCI_10s", "intensity_10s", "VWAP_10s", "ILLIQ_10s", "RV_10s",
    "OFI_60s", "TCI_60s", "intensity_60s", "VWAP_60s", "ILLIQ_60s", "RV_60s",
    "Kyle_lambda_60s",
    "OFI_300s", "TCI_300s", "intensity_300s", "VWAP_300s", "ILLIQ_300s",
    "RV_300s", "Kyle_lambda_300s",
    "VWAP_dev_10s", "VWAP_dev_60s", "VWAP_dev_300s", "CV_dt_10s",
]


# =============================================================================
# SEED
# =============================================================================

def set_seed(seed=42):
    """Locks all randomness for baseline reproducibility using modern APIs."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    return rng


# =============================================================================
# RESUMPTION HELPERS
# =============================================================================

def get_fold_prefix(asset, fold, is_pooled):
    if is_pooled:
        return f"{RESULTS_PREFIX}pooled/fold_{fold}/"
    return f"{RESULTS_PREFIX}{asset}/fold_{fold}/"


def get_stage_local_dir(asset, fold, is_pooled):
    """
    Persistent-volume directory for stage cache files.
    Used by both save and load so the path is always consistent.
    """
    tag  = "pooled" if is_pooled else asset
    path = os.path.join(STAGE_CACHE_DIR, tag, f"fold_{fold}")
    os.makedirs(path, exist_ok=True)
    return path


def fold_already_complete(bucket, asset, fold, is_pooled):
    """Fold-level check: metrics_binary.json marks the entire fold as done."""
    prefix = get_fold_prefix(asset, fold, is_pooled)
    return bucket.blob(f"{prefix}metrics_binary.json").exists()


def stage_already_complete(bucket, asset, fold, is_pooled, mode, model_name):
    """
    Model-level check via zero-byte GCS marker.
    One marker per (mode, model_name) per fold.
    Checked before training each model — crash mid-fold won't re-run
    models that already finished.
    """
    prefix = get_fold_prefix(asset, fold, is_pooled)
    return bucket.blob(f"{prefix}.done_{mode}_{model_name}").exists()


def mark_stage_complete(bucket, asset, fold, is_pooled, mode, model_name):
    """Upload zero-byte GCS marker once a model stage finishes."""
    prefix = get_fold_prefix(asset, fold, is_pooled)
    bucket.blob(f"{prefix}.done_{mode}_{model_name}").upload_from_string("")
    logger.info(f"    Stage marked complete: {mode}/{model_name}")


def save_stage_outputs(bucket, asset, fold, is_pooled, mode, model_name,
                       probs, preds, metrics_dict):
    """
    Persist stage outputs to two locations:
      1. Persistent volume  — fast reload on restart without GCS roundtrip
      2. GCS               — backup / source of truth if persistent volume is wiped

    Arrays saved as Parquet via Polars — schema-aware and version-stable across
    NumPy/Python upgrades. Safer than .npy on restarts after environment changes.
    Metrics saved as JSON.
    """
    local_dir = get_stage_local_dir(asset, fold, is_pooled)
    prefix    = get_fold_prefix(asset, fold, is_pooled)

    probs_local   = os.path.join(local_dir, f"{mode}_{model_name}_probs.parquet")
    preds_local   = os.path.join(local_dir, f"{mode}_{model_name}_preds.parquet")
    metrics_local = os.path.join(local_dir, f"{mode}_{model_name}_metrics.json")

    # Probs: 2D array — one column per class (prob_0, prob_1, ...)
    probs_df = pl.DataFrame(
        {f"prob_{c}": probs[:, c].astype(np.float32) for c in range(probs.shape[1])}
    )
    probs_df.write_parquet(probs_local)

    # Preds: 1D array — single column of hard predictions
    preds_df = pl.DataFrame({"pred": preds.astype(np.int32)})
    preds_df.write_parquet(preds_local)

    with open(metrics_local, "w") as f:
        json.dump(metrics_dict, f, indent=2)

    upload_to_gcs(bucket, probs_local,   f"{prefix}stage/{mode}_{model_name}_probs.parquet")
    upload_to_gcs(bucket, preds_local,   f"{prefix}stage/{mode}_{model_name}_preds.parquet")
    upload_to_gcs(bucket, metrics_local, f"{prefix}stage/{mode}_{model_name}_metrics.json")


def load_stage_outputs(bucket, asset, fold, is_pooled, mode, model_name):
    """
    Load probs, preds, and metrics for a completed stage.

    Priority:
      1. Persistent volume (fast, no network)
      2. GCS download (fallback if pod was wiped or this is a new pod)

    Returns: (probs np.ndarray, preds np.ndarray, metrics dict)
    """
    local_dir = get_stage_local_dir(asset, fold, is_pooled)
    prefix    = get_fold_prefix(asset, fold, is_pooled)

    probs_local   = os.path.join(local_dir, f"{mode}_{model_name}_probs.parquet")
    preds_local   = os.path.join(local_dir, f"{mode}_{model_name}_preds.parquet")
    metrics_local = os.path.join(local_dir, f"{mode}_{model_name}_metrics.json")

    for local_path, gcs_path in [
        (probs_local,   f"{prefix}stage/{mode}_{model_name}_probs.parquet"),
        (preds_local,   f"{prefix}stage/{mode}_{model_name}_preds.parquet"),
        (metrics_local, f"{prefix}stage/{mode}_{model_name}_metrics.json"),
    ]:
        if not os.path.exists(local_path):
            logger.info(f"    Downloading stage file from GCS: {gcs_path}")
            bucket.blob(gcs_path).download_to_filename(local_path)

    # Reconstruct numpy arrays from Polars DataFrames
    probs = pl.read_parquet(probs_local).to_numpy().astype(np.float32)
    preds = pl.read_parquet(preds_local)["pred"].to_numpy().astype(np.int32)

    with open(metrics_local) as f:
        metrics_dict = json.load(f)

    return probs, preds, metrics_dict


# =============================================================================
# GAF
# =============================================================================

def compute_gasf_pytorch(x):
    """
    Computes Gramian Angular Summation Fields purely on the GPU.
    x shape: (batch_size, seq_length, n_features)
    """
    x = x.permute(0, 2, 1)  # (batch, n_features, seq_length)

    x_min = x.min(dim=-1, keepdim=True)[0]
    x_max = x.max(dim=-1, keepdim=True)[0]
    denom = x_max - x_min
    denom = torch.where(denom == 0, torch.ones_like(denom), denom)

    x_norm = ((x - x_min) / denom) * 2.0 - 1.0
    x_norm = torch.clamp(x_norm, -1.0, 1.0)

    x_i = x_norm.unsqueeze(-1)
    x_j = x_norm.unsqueeze(-2)

    sqrt_i = torch.sqrt(1.0 - x_i**2)
    sqrt_j = torch.sqrt(1.0 - x_j**2)

    return x_i * x_j - sqrt_i * sqrt_j


# =============================================================================
# DATA LOADING
# =============================================================================

def parse_window_months(start_ym, end_ym):
    months = []
    sy, sm = int(start_ym[:4]), int(start_ym[5:])
    ey, em = int(end_ym[:4]), int(end_ym[5:])
    y, m = sy, sm
    while (y, m) <= (ey, em):
        months.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def load_window_data(bucket, asset, window_idx, feature_cols=None):
    """
    [OPT] Persistent-volume cache: downloads each asset/month parquet once and
    caches to disk. All subsequent calls (across folds) read from disk only.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    start_ym, end_ym = WINDOWS[window_idx]
    months = parse_window_months(start_ym, end_ym)

    feature_frames = []
    label_frames   = []

    for year, month in months:
        if asset == "SOLUSDT" and (year, month) < (2020, 11):
            continue

        cache_key   = f"{CACHE_DIR}/{asset}-{year}-{month:02d}"
        feat_cache  = f"{cache_key}-features.parquet"
        label_cache = f"{cache_key}-labels.parquet"

        if os.path.exists(feat_cache) and os.path.exists(label_cache):
            feature_frames.append(pl.read_parquet(feat_cache))
            label_frames.append(pl.read_parquet(label_cache))
            continue

        f_blob = bucket.blob(f"{FEATURES_PREFIX}{asset}-features-{year}-{month:02d}.parquet")
        l_blob = bucket.blob(f"{LABELS_PREFIX}{asset}-labels-{year}-{month:02d}.parquet")

        if not f_blob.exists() or not l_blob.exists():
            logger.warning(f"  Missing data for {asset} {year}-{month:02d}")
            continue

        try:
            f_blob.download_to_filename(feat_cache)
            l_blob.download_to_filename(label_cache)
            feature_frames.append(pl.read_parquet(feat_cache))
            label_frames.append(pl.read_parquet(label_cache))
            logger.info(f"  Cached {asset} {year}-{month:02d} to persistent volume")
        except Exception as e:
            # Remove partial files so next run retries cleanly
            for p in [feat_cache, label_cache]:
                if os.path.exists(p):
                    os.remove(p)
            logger.warning(f"  Download failed for {asset} {year}-{month:02d}: {e}")
            continue

    if not feature_frames:
        return None

    df_feat  = pl.concat(feature_frames).sort("time")
    df_label = pl.concat(label_frames).sort("time")
    df = df_feat.join(df_label, on="time", how="inner")

    if feature_cols:
        available = [c for c in feature_cols if c in df.columns]
        df = df.select(available + ["label", "time"])

    return df


# =============================================================================
# DATA PREP
# =============================================================================

def prepare_xy(df, binary=False, purge_end=True):
    if purge_end:
        df = df.head(len(df) - PURGE_ROWS)

    feature_cols = [c for c in df.columns if c not in ["label", "time"]]
    X = df.select(feature_cols).to_numpy()
    y = df["label"].to_numpy()
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if binary:
        y = (y == BINARY_STRESS_LABEL).astype(np.int8)

    return X, y, feature_cols


def get_class_weights(y, n_classes):
    weights = []
    for c in range(n_classes):
        count = (y == c).sum()
        weights.append(1.0 if count == 0 else len(y) / (n_classes * count))
    return weights


# =============================================================================
# MODELS
# =============================================================================

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, seq_length):
        self.X          = torch.FloatTensor(X)
        self.y          = torch.LongTensor(y.astype(np.int64))
        self.seq_length = seq_length

    def __len__(self):
        return max(0, len(self.X) - self.seq_length)

    def __getitem__(self, idx):
        return (
            self.X[idx: idx + self.seq_length],
            self.y[idx + self.seq_length - 1],
        )


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, n_classes, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])


class CNNGAFClassifier(nn.Module):
    def __init__(self, n_features, seq_length, n_classes):
        super().__init__()
        self.conv1   = nn.Conv2d(n_features, 32, kernel_size=3, padding=1)
        self.conv2   = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool    = nn.MaxPool2d(2, 2)
        self.relu    = nn.ReLU()
        flat_size    = 64 * (seq_length // 4) * (seq_length // 4)
        self.fc1     = nn.Linear(flat_size, 128)
        self.fc2     = nn.Linear(128, n_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.reshape(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)


# =============================================================================
# TRAINING
# =============================================================================

def _make_dataloaders(train_ds, test_ds, batch_size):
    """Shared DataLoader factory with all [OPT] settings applied."""
    kw = dict(
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kw),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kw),
    )


def train_lstm(X_train, y_train, X_test, y_test, class_weights, n_classes):
    input_dim = X_train.shape[1]
    model = LSTMClassifier(
        input_dim=input_dim, hidden_dim=64, num_layers=2,
        n_classes=n_classes, dropout=0.5
    ).to(DEVICE)

    # [OPT] torch.compile: JIT-compiles the graph. First epoch slower (compilation),
    # every subsequent epoch faster. Net win across 20 epochs.
    if DEVICE.type == "cuda":
        model = torch.compile(model)

    weight_tensor = torch.FloatTensor(class_weights).to(DEVICE)
    criterion     = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer     = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler     = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    amp_scaler    = GradScaler("cuda")  # [OPT] BF16 AMP loss scaler

    train_ds = TimeSeriesDataset(X_train, y_train, SEQ_LENGTH)
    test_ds  = TimeSeriesDataset(X_test,  y_test,  SEQ_LENGTH)
    train_loader, test_loader = _make_dataloaders(train_ds, test_ds, BATCH_SIZE)

    train_losses, val_losses = [], []
    best_val_loss    = np.inf
    best_state       = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch_x, batch_y in train_loader:
            # [OPT] non_blocking: overlaps CPU→GPU transfer with compute
            batch_x = batch_x.to(DEVICE, non_blocking=True)
            batch_y = batch_y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            # [OPT] BF16 autocast: Blackwell natively supports BF16 tensor cores.
            # BF16 preferred over FP16 for financial data — wider dynamic range,
            # no NaN risk from large magnitude spikes in market microstructure features.
            with autocast("cuda", dtype=torch.bfloat16):
                outputs = model(batch_x)
                loss    = criterion(outputs, batch_y)

            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(DEVICE, non_blocking=True)
                batch_y = batch_y.to(DEVICE, non_blocking=True)
                with autocast("cuda", dtype=torch.bfloat16):
                    outputs   = model(batch_x)
                    val_loss += criterion(outputs, batch_y).item()

        avg_val_loss = val_loss / max(len(test_loader), 1)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        logger.info(f"    LSTM Epoch {epoch+1}/{EPOCHS} train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss    = avg_val_loss
            best_state       = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                logger.info(f"    LSTM early stopping at epoch {epoch+1}, best val_loss={best_val_loss:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    all_probs, all_preds = [], []
    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(DEVICE, non_blocking=True)
            with autocast("cuda", dtype=torch.bfloat16):
                outputs = model(batch_x)
            # Cast to float32 before softmax — BF16 softmax can be numerically unstable
            all_probs.extend(torch.softmax(outputs.float(), dim=1).cpu().numpy())
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())

    return model, np.array(all_probs), np.array(all_preds), train_losses, val_losses


def train_cnn_gaf(X_train, y_train, X_test, y_test, class_weights, n_classes, seq_length):
    train_ds = TimeSeriesDataset(X_train, y_train, seq_length)
    test_ds  = TimeSeriesDataset(X_test,  y_test,  seq_length)

    # [OPT] CNN_BATCH_SIZE=4096: was 256 — 16x fewer batches/epoch, same VRAM headroom
    train_loader, test_loader = _make_dataloaders(train_ds, test_ds, CNN_BATCH_SIZE)

    n_features = X_train.shape[1]
    model = CNNGAFClassifier(
        n_features=n_features, seq_length=seq_length, n_classes=n_classes
    ).to(DEVICE)

    if DEVICE.type == "cuda":
        model = torch.compile(model)

    weight_tensor = torch.FloatTensor(class_weights).to(DEVICE)
    criterion     = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer     = torch.optim.Adam(model.parameters(), lr=0.001)
    amp_scaler    = GradScaler("cuda")

    train_losses, val_losses = [], []
    best_val_loss    = np.inf
    best_state       = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE, non_blocking=True)
            batch_y = batch_y.to(DEVICE, non_blocking=True)

            with autocast("cuda", dtype=torch.bfloat16):
                batch_x_gaf = compute_gasf_pytorch(batch_x)
                outputs     = model(batch_x_gaf)
                loss        = criterion(outputs, batch_y)

            optimizer.zero_grad()
            amp_scaler.scale(loss).backward()
            amp_scaler.step(optimizer)
            amp_scaler.update()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(DEVICE, non_blocking=True)
                batch_y = batch_y.to(DEVICE, non_blocking=True)
                with autocast("cuda", dtype=torch.bfloat16):
                    batch_x_gaf = compute_gasf_pytorch(batch_x)
                    outputs     = model(batch_x_gaf)
                    val_loss   += criterion(outputs, batch_y).item()

        avg_val_loss = val_loss / max(len(test_loader), 1)
        val_losses.append(avg_val_loss)

        logger.info(f"    CNN-GAF Epoch {epoch+1}/{EPOCHS} train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss    = avg_val_loss
            best_state       = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                logger.info(f"    CNN-GAF early stopping at epoch {epoch+1}, best val_loss={best_val_loss:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    all_probs, all_preds = [], []
    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(DEVICE, non_blocking=True)
            with autocast("cuda", dtype=torch.bfloat16):
                batch_x_gaf = compute_gasf_pytorch(batch_x)
                outputs     = model(batch_x_gaf)
            all_probs.extend(torch.softmax(outputs.float(), dim=1).cpu().numpy())
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())

    return model, np.array(all_probs), np.array(all_preds), train_losses, val_losses


# =============================================================================
# EXPLAINABILITY
# =============================================================================

def compute_shap(model_obj, X_train, X_test, feature_names, model_name):
    logger.info(f"    Computing SHAP values for {model_name}")
    try:
        if model_name in ["rf", "xgb"]:
            explainer   = shap.TreeExplainer(model_obj)
            sample      = X_test[:1000] if len(X_test) > 1000 else X_test
            shap_values = explainer.shap_values(sample)
        elif model_name == "lr":
            explainer   = shap.LinearExplainer(model_obj, X_train)
            sample      = X_test[:1000] if len(X_test) > 1000 else X_test
            shap_values = explainer.shap_values(sample)
        else:
            return None
        return shap_values, sample, feature_names
    except Exception as e:
        logger.warning(f"    SHAP failed for {model_name}: {e}")
        return None


def compute_integrated_gradients(model, X_test, y_test, feature_names, n_classes):
    logger.info("    Computing Integrated Gradients for LSTM")
    try:
        model.eval()
        ds     = TimeSeriesDataset(X_test, y_test, SEQ_LENGTH)
        loader = DataLoader(ds, batch_size=64, shuffle=False)
        ig     = IntegratedGradients(model)
        all_attrs = []

        with torch.backends.cudnn.flags(enabled=False):
            for batch_x, batch_y in loader:
                batch_x  = batch_x.to(DEVICE).requires_grad_(True)
                baseline = torch.zeros_like(batch_x).to(DEVICE)
                for c in range(n_classes):
                    attrs = ig.attribute(batch_x, baseline, target=c)
                    all_attrs.append(attrs.mean(dim=1).abs().cpu().detach().numpy())
                if len(all_attrs) >= 10:
                    break

        mean_attrs = np.mean(all_attrs, axis=0)
        return {k: float(v) for k, v in zip(feature_names, mean_attrs.mean(axis=0))}
    except Exception as e:
        logger.warning(f"    Integrated Gradients failed: {e}")
        return None


# =============================================================================
# PLOTS
# =============================================================================

def plot_confusion_matrix(y_true, y_pred, labels, title, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_curves(y_true, probs_dict, n_classes, title, save_path):
    fig, axes   = plt.subplots(1, n_classes, figsize=(6 * n_classes, 5))
    class_names = ["calm", "elevated", "stress"]

    for c in range(n_classes):
        ax    = axes[c] if n_classes > 1 else axes
        y_bin = (y_true == c).astype(int)

        for model_name, probs in probs_dict.items():
            if probs is None or len(probs) != len(y_bin):
                continue
            try:
                prob_c      = probs[:, c] if probs.ndim == 2 else probs
                fpr, tpr, _ = roc_curve(y_bin, prob_c)
                auc         = roc_auc_score(y_bin, prob_c)
                ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})")
            except Exception:
                continue

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_title(f"ROC - {class_names[c]}")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend(fontsize=8)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_loss_curves(train_losses, val_losses, model_name, title, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label="Train Loss")
    ax.plot(val_losses,   label="Val Loss")
    ax.set_title(f"{model_name} Loss Curves - {title}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_shap_summary(shap_values, X_sample, feature_names, title, save_path):
    try:
        plt.figure(figsize=(10, 8))
        sv = shap_values[2] if isinstance(shap_values, list) else shap_values
        shap.summary_plot(sv, X_sample, feature_names=feature_names, show=False, plot_type="bar")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        logger.warning(f"    SHAP plot failed: {e}")


# =============================================================================
# GCS UPLOAD
# =============================================================================

def upload_to_gcs(bucket, local_path, gcs_path):
    try:
        bucket.blob(gcs_path).upload_from_filename(local_path)
        logger.info(f"    Uploaded {gcs_path}")
    except Exception as e:
        logger.warning(f"    Upload failed for {gcs_path}: {e}")


# =============================================================================
# FOLD RUNNER
# =============================================================================

def run_fold(
    gcs_client, bucket, asset, fold, train_windows, test_window,
    feature_cols, is_pooled=False, pooled_assets=None
):
    tag = f"{'pooled' if is_pooled else asset} fold {fold}"

    if fold_already_complete(bucket, asset, fold, is_pooled):
        logger.info(f"  {tag} - already complete, skipping")
        return

    logger.info(f"  {tag} - loading data")

    if is_pooled:
        train_frames, test_frames = [], []
        for a in pooled_assets:
            for w in train_windows:
                df = load_window_data(bucket, a, w, feature_cols)
                if df is not None and not df.is_empty():
                    df = df.with_columns(pl.lit(pooled_assets.index(a)).alias("asset_id"))
                    train_frames.append(df)
            df = load_window_data(bucket, a, test_window, feature_cols)
            if df is not None and not df.is_empty():
                df = df.with_columns(pl.lit(pooled_assets.index(a)).alias("asset_id"))
                test_frames.append(df)

        if not train_frames or not test_frames:
            logger.warning(f"  {tag} insufficient data, skipping")
            return

        train_df = pl.concat(train_frames).sort("time")
        test_df  = pl.concat(test_frames).sort("time")
    else:
        train_frames = [load_window_data(bucket, asset, w, feature_cols) for w in train_windows]
        valid_train  = [f for f in train_frames if f is not None and not f.is_empty()]

        if not valid_train:
            logger.warning(f"  {tag} no valid training data, skipping")
            return

        train_df = pl.concat(valid_train).sort("time")
        test_df  = load_window_data(bucket, asset, test_window, feature_cols)

    if test_df is None or test_df.is_empty() or train_df.is_empty():
        logger.warning(f"  {tag} no data, skipping")
        return

    logger.info(f"  {tag} - train: {len(train_df):,} rows, test: {len(test_df):,} rows")

    output_dir = f"/tmp/{asset}_fold{fold}"
    os.makedirs(output_dir, exist_ok=True)
    models_dir = f"{output_dir}/models"
    os.makedirs(models_dir, exist_ok=True)

    for binary in [False, True]:
        mode        = "binary" if binary else "multiclass"
        n_classes   = 2 if binary else N_CLASSES
        label_names = ["not_stress", "stress"] if binary else ["calm", "elevated", "stress"]

        logger.info(f"  {tag} - {mode} training")

        X_train, y_train, feat_names = prepare_xy(train_df, binary=binary, purge_end=True)
        X_test,  y_test,  _          = prepare_xy(test_df,  binary=binary, purge_end=False)

        class_weights = get_class_weights(y_train, n_classes)
        logger.info(f"    Class weights: {class_weights}")

        # Scaler always runs — it's fast and required regardless of which models are resuming
        scaler     = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        predictions = {}  # model_name → probs array (for ROC curves + pred parquet)
        metrics     = {}  # model_name → classification_report dict (for metrics JSON)

        # ── Logistic Regression ──────────────────────────────────────────────

        if stage_already_complete(bucket, asset, fold, is_pooled, mode, "lr"):
            logger.info("    LR already complete — loading saved outputs")
            lr_probs, lr_preds, metrics["lr"] = load_stage_outputs(
                bucket, asset, fold, is_pooled, mode, "lr"
            )
            predictions["lr"] = lr_probs
        else:
            logger.info("    Training Logistic Regression")
            lr = LogisticRegression(
                class_weight={i: w for i, w in enumerate(class_weights)},
                max_iter=2000, C=1.0
            )
            lr.fit(X_train_sc, y_train)
            lr_preds = lr.predict(X_test_sc)
            lr_probs = lr.predict_proba(X_test_sc)
            predictions["lr"] = lr_probs
            metrics["lr"] = classification_report(
                y_test, lr_preds, target_names=label_names, output_dict=True
            )

            shap_result = compute_shap(lr, X_train_sc, X_test_sc, feat_names, "lr")
            if shap_result:
                plot_shap_summary(shap_result[0], shap_result[1], feat_names,
                                  f"LR SHAP - {tag} {mode}", f"{output_dir}/shap_lr_{mode}.png")

            with open(f"{models_dir}/lr_{mode}.pkl", "wb") as f:
                pickle.dump({"model": lr, "scaler": scaler}, f)

            save_stage_outputs(bucket, asset, fold, is_pooled, mode, "lr",
                               lr_probs, lr_preds, metrics["lr"])
            mark_stage_complete(bucket, asset, fold, is_pooled, mode, "lr")

        # ── Random Forest ────────────────────────────────────────────────────
        # Aggressive memory cleanup before RF — on large folds (fold 4 pooled =
        # 18.8M rows) prior model objects accumulate enough heap fragmentation
        # to trigger the OOM killer mid-RF. gc.collect() releases Python objects,
        # malloc_trim returns fragmented heap pages to the OS.
        import ctypes
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
            logger.info("    malloc_trim: heap released to OS")
        except Exception as e:
            logger.warning(f"    malloc_trim failed (non-fatal): {e}")

        if stage_already_complete(bucket, asset, fold, is_pooled, mode, "rf"):
            logger.info("    RF already complete — loading saved outputs")
            rf_probs, rf_preds, metrics["rf"] = load_stage_outputs(
                bucket, asset, fold, is_pooled, mode, "rf"
            )
            predictions["rf"] = rf_probs
        else:
            logger.info("    Training Random Forest")
            rf = RandomForestClassifier(
                n_estimators=200, max_depth=12,
                class_weight={i: w for i, w in enumerate(class_weights)},
                n_jobs=-1, random_state=42
            )
            rf.fit(X_train_sc, y_train)
            rf_preds = rf.predict(X_test_sc)
            rf_probs = rf.predict_proba(X_test_sc)
            predictions["rf"] = rf_probs
            metrics["rf"] = classification_report(
                y_test, rf_preds, target_names=label_names, output_dict=True
            )

            shap_result = compute_shap(rf, X_train_sc, X_test_sc, feat_names, "rf")
            if shap_result:
                plot_shap_summary(shap_result[0], shap_result[1], feat_names,
                                  f"RF SHAP - {tag} {mode}", f"{output_dir}/shap_rf_{mode}.png")

            with open(f"{models_dir}/rf_{mode}.pkl", "wb") as f:
                pickle.dump({"model": rf, "scaler": scaler}, f)

            save_stage_outputs(bucket, asset, fold, is_pooled, mode, "rf",
                               rf_probs, rf_preds, metrics["rf"])
            mark_stage_complete(bucket, asset, fold, is_pooled, mode, "rf")

        # ── XGBoost ──────────────────────────────────────────────────────────

        if stage_already_complete(bucket, asset, fold, is_pooled, mode, "xgb"):
            logger.info("    XGBoost already complete — loading saved outputs")
            xgb_probs, xgb_preds, metrics["xgb"] = load_stage_outputs(
                bucket, asset, fold, is_pooled, mode, "xgb"
            )
            predictions["xgb"] = xgb_probs
        else:
            logger.info("    Training XGBoost")
            sample_weight = np.array([class_weights[int(y)] for y in y_train])

            # [OPT] subsample + colsample_bytree: reduces per-tree rows on 6.3M dataset,
            # speeds up GPU hist builds, improves generalisation on collinear features.
            xgb_params = dict(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                tree_method="hist",
                device="cuda" if DEVICE.type == "cuda" else "cpu",
                subsample=0.8, colsample_bytree=0.8,
                n_jobs=-1, random_state=42,
                objective="multi:softprob" if not binary else "binary:logistic",
                num_class=n_classes if not binary else None,
                eval_metric="mlogloss" if not binary else "logloss"
            )
            if binary:
                xgb_params.pop("num_class")

            xgb_model = xgb.XGBClassifier(**xgb_params)
            xgb_model.fit(X_train_sc, y_train, sample_weight=sample_weight)
            xgb_preds = xgb_model.predict(X_test_sc)
            xgb_probs = xgb_model.predict_proba(X_test_sc)
            predictions["xgb"] = xgb_probs
            metrics["xgb"] = classification_report(
                y_test, xgb_preds, target_names=label_names, output_dict=True
            )

            shap_result = compute_shap(xgb_model, X_train_sc, X_test_sc, feat_names, "xgb")
            if shap_result:
                plot_shap_summary(shap_result[0], shap_result[1], feat_names,
                                  f"XGB SHAP - {tag} {mode}", f"{output_dir}/shap_xgb_{mode}.png")

            with open(f"{models_dir}/xgb_{mode}.pkl", "wb") as f:
                pickle.dump({"model": xgb_model, "scaler": scaler}, f)

            save_stage_outputs(bucket, asset, fold, is_pooled, mode, "xgb",
                               xgb_probs, xgb_preds, metrics["xgb"])
            mark_stage_complete(bucket, asset, fold, is_pooled, mode, "xgb")

        # ── LSTM ─────────────────────────────────────────────────────────────

        if stage_already_complete(bucket, asset, fold, is_pooled, mode, "lstm"):
            logger.info("    LSTM already complete — loading saved outputs")
            lstm_probs, lstm_preds, metrics["lstm"] = load_stage_outputs(
                bucket, asset, fold, is_pooled, mode, "lstm"
            )
            predictions["lstm"] = lstm_probs
        else:
            logger.info("    Training LSTM")
            lstm_model, lstm_probs, lstm_preds, lstm_tl, lstm_vl = train_lstm(
                X_train_sc, y_train, X_test_sc, y_test, class_weights, n_classes
            )
            predictions["lstm"] = lstm_probs
            metrics["lstm"] = classification_report(
                y_test[SEQ_LENGTH:], lstm_preds, target_names=label_names, output_dict=True
            )

            plot_loss_curves(lstm_tl, lstm_vl, "LSTM",
                             f"{tag} {mode}", f"{output_dir}/loss_lstm_{mode}.png")

            ig_result = compute_integrated_gradients(lstm_model, X_test_sc, y_test, feat_names, n_classes)
            if ig_result:
                with open(f"{output_dir}/ig_lstm_{mode}.json", "w") as f:
                    json.dump(ig_result, f, indent=2)

            torch.save(lstm_model.state_dict(), f"{models_dir}/lstm_{mode}.pt")

            save_stage_outputs(bucket, asset, fold, is_pooled, mode, "lstm",
                               lstm_probs, lstm_preds, metrics["lstm"])
            mark_stage_complete(bucket, asset, fold, is_pooled, mode, "lstm")

        # ── CNN-GAF ──────────────────────────────────────────────────────────

        if stage_already_complete(bucket, asset, fold, is_pooled, mode, "cnn_gaf"):
            logger.info("    CNN-GAF already complete — loading saved outputs")
            cnn_probs, cnn_preds, metrics["cnn_gaf"] = load_stage_outputs(
                bucket, asset, fold, is_pooled, mode, "cnn_gaf"
            )
            predictions["cnn_gaf"] = cnn_probs
        else:
            logger.info("    Training CNN-GAF")
            gaf_feat_idx = [feat_names.index(f) for f in GAF_FEATURES if f in feat_names]
            X_train_gaf  = X_train_sc[:, gaf_feat_idx]
            X_test_gaf   = X_test_sc[:, gaf_feat_idx]

            cnn_model, cnn_probs, cnn_preds, cnn_tl, cnn_vl = train_cnn_gaf(
                X_train_gaf, y_train, X_test_gaf, y_test, class_weights, n_classes, SEQ_LENGTH
            )
            predictions["cnn_gaf"] = cnn_probs
            metrics["cnn_gaf"] = classification_report(
                y_test[SEQ_LENGTH:], cnn_preds, target_names=label_names, output_dict=True
            )

            plot_loss_curves(cnn_tl, cnn_vl, "CNN-GAF",
                             f"{tag} {mode}", f"{output_dir}/loss_cnn_{mode}.png")

            torch.save(cnn_model.state_dict(), f"{models_dir}/cnn_gaf_{mode}.pt")

            save_stage_outputs(bucket, asset, fold, is_pooled, mode, "cnn_gaf",
                               cnn_probs, cnn_preds, metrics["cnn_gaf"])
            mark_stage_complete(bucket, asset, fold, is_pooled, mode, "cnn_gaf")

        # ── Aggregate outputs (runs every time — fast, uses loaded/trained probs) ──
        #
        # Confusion matrices: reconstruct hard preds from probs via argmax.
        # This works correctly whether stage was trained fresh or loaded from cache.

        for model_name in ["lr", "rf", "xgb"]:
            probs = predictions.get(model_name)
            if probs is not None and len(probs) == len(y_test):
                plot_confusion_matrix(
                    y_test, probs.argmax(axis=1), label_names,
                    f"{model_name.upper()} - {tag} {mode}",
                    f"{output_dir}/cm_{model_name}_{mode}.png"
                )

        plot_roc_curves(
            y_test,
            {k: v for k, v in predictions.items() if v is not None and len(v) == len(y_test)},
            n_classes,
            f"ROC Curves - {tag} {mode}",
            f"{output_dir}/roc_{mode}.png"
        )

        with open(f"{output_dir}/metrics_{mode}.json", "w") as f:
            json.dump(metrics, f, indent=2)

        pred_records = {"y_true": y_test.tolist()}
        for model_name, probs in predictions.items():
            if probs is not None and len(probs) == len(y_test):
                if probs.ndim == 2:
                    for c in range(probs.shape[1]):
                        pred_records[f"{model_name}_prob_class{c}"] = probs[:, c].tolist()
                else:
                    pred_records[f"{model_name}_prob"] = probs.tolist()

        pl.DataFrame(pred_records).write_parquet(f"{output_dir}/predictions_{mode}.parquet")

        gc.collect()

    # Upload everything in output_dir to GCS
    prefix = get_fold_prefix(asset, fold, is_pooled)
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            local_path = os.path.join(root, file)
            gcs_path   = f"{prefix}{os.path.relpath(local_path, output_dir)}"
            upload_to_gcs(bucket, local_path, gcs_path)

    logger.info(f"  {tag} - complete, results uploaded to gs://{BUCKET}/{prefix}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    set_seed(42)
    logger.info("Starting Training Pipeline")
    logger.info(f"Device         : {DEVICE}")
    logger.info(f"SEQ_LENGTH     : {SEQ_LENGTH}")
    logger.info(f"EPOCHS         : {EPOCHS}")
    logger.info(f"N_CLASSES      : {N_CLASSES}")
    logger.info(f"PURGE_ROWS     : {PURGE_ROWS}")
    logger.info(f"EARLY_STOP     : patience={EARLY_STOP_PATIENCE}")
    logger.info(f"BATCH_SIZE     : {BATCH_SIZE} (LSTM) / {CNN_BATCH_SIZE} (CNN-GAF)")
    logger.info(f"NUM_WORKERS    : {NUM_WORKERS}")
    logger.info(f"CACHE_DIR      : {CACHE_DIR}")
    logger.info(f"STAGE_CACHE    : {STAGE_CACHE_DIR}")

    # Pre-flight: verify persistent volume paths are writable before any work starts.
    # Catches permission or mount issues immediately rather than mid-run.
    for critical_dir in [CACHE_DIR, STAGE_CACHE_DIR]:
        os.makedirs(critical_dir, exist_ok=True)
        probe = os.path.join(critical_dir, ".write_probe")
        try:
            with open(probe, "w") as f:
                f.write("ok")
            os.remove(probe)
            logger.info(f"Pre-flight OK  : {critical_dir}")
        except Exception as e:
            raise RuntimeError(f"Cannot write to {critical_dir} — check volume mount: {e}")

    gcs_client = storage.Client()
    bucket     = gcs_client.bucket(BUCKET)
    n_folds    = len(WINDOWS) - 1

    for asset in ASSETS:
        logger.info(f"{'=' * 60}")
        logger.info(f"Asset-specific training: {asset}")

        for fold in range(1, n_folds + 1):
            train_windows = list(range(0, fold))
            test_window   = fold
            logger.info(f"  Fold {fold}/{n_folds}: train={train_windows} test={test_window}")
            run_fold(
                gcs_client, bucket, asset, fold,
                train_windows, test_window,
                feature_cols=None, is_pooled=False
            )
            gc.collect()

    logger.info(f"{'=' * 60}")
    logger.info("Pooled training: all assets")

    for fold in range(1, n_folds + 1):
        train_windows = list(range(0, fold))
        test_window   = fold
        logger.info(f"  Fold {fold}/{n_folds}: train={train_windows} test={test_window}")
        run_fold(
            gcs_client, bucket, "pooled", fold,
            train_windows, test_window,
            feature_cols=COMMON_FEATURES, is_pooled=True, pooled_assets=ASSETS
        )
        gc.collect()

    logger.info("=" * 60)
    logger.info("Training Pipeline Complete")


if __name__ == "__main__":
    main()

