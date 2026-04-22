import sys
import os
import gc
import json
import logging
import pickle
import tempfile
from datetime import datetime

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
from captum.attr import IntegratedGradients
from google.cloud import storage
from pyts.image import GramianAngularField

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ASSETS, BUCKET, WINDOWS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

FEATURES_PREFIX = "v2/features/"
LABELS_PREFIX = "v2/labels/"
RESULTS_PREFIX = "v2/results/"

SEQ_LENGTH = 60
BATCH_SIZE = 2048
EPOCHS = 20
PURGE_ROWS = 30
EARLY_STOP_PATIENCE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_CLASSES = 3
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
    start_ym, end_ym = WINDOWS[window_idx]
    months = parse_window_months(start_ym, end_ym)

    feature_frames = []
    label_frames = []

    for year, month in months:
        if asset == "SOLUSDT" and (year, month) < (2020, 11):
            continue

        f_blob = bucket.blob(
            f"{FEATURES_PREFIX}{asset}-features-{year}-{month:02d}.parquet"
        )
        l_blob = bucket.blob(
            f"{LABELS_PREFIX}{asset}-labels-{year}-{month:02d}.parquet"
        )

        if not f_blob.exists() or not l_blob.exists():
            logger.warning(f"  Missing data for {asset} {year}-{month:02d}")
            continue

        tmp_f = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet")
        tmp_l = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet")

        try:
            f_blob.download_to_filename(tmp_f.name)
            l_blob.download_to_filename(tmp_l.name)

            df_f = pl.read_parquet(tmp_f.name)
            df_l = pl.read_parquet(tmp_l.name)

            feature_frames.append(df_f)
            label_frames.append(df_l)
        finally:
            os.remove(tmp_f.name)
            os.remove(tmp_l.name)

    if not feature_frames:
        return None

    df_feat = pl.concat(feature_frames).sort("time")
    df_label = pl.concat(label_frames).sort("time")
    df = df_feat.join(df_label, on="time", how="inner")

    if feature_cols:
        available = [c for c in feature_cols if c in df.columns]
        df = df.select(available + ["label", "time"])

    return df


def prepare_xy(df, binary=False, purge_end=True):
    if purge_end:
        df = df.head(len(df) - PURGE_ROWS)

    feature_cols = [
        c for c in df.columns if c not in ["label", "time"]
    ]

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
        if count == 0:
            weights.append(1.0)
        else:
            weights.append(len(y) / (n_classes * count))
    return weights


def fold_already_complete(bucket, asset, fold, is_pooled):
    prefix = (
        f"{RESULTS_PREFIX}pooled/fold_{fold}/"
        if is_pooled
        else f"{RESULTS_PREFIX}{asset}/fold_{fold}/"
    )
    marker = f"{prefix}metrics_binary.json"
    return bucket.blob(marker).exists()


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, seq_length):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y.astype(np.int64))
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
        self.conv1 = nn.Conv2d(n_features, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        flat_size = 64 * (seq_length // 4) * (seq_length // 4)
        self.fc1 = nn.Linear(flat_size, 128)
        self.fc2 = nn.Linear(128, n_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)


def train_lstm(X_train, y_train, X_test, y_test, class_weights, n_classes):
    input_dim = X_train.shape[1]
    
    # REGULARISATION TWEAKS APPLIED HERE
    model = LSTMClassifier(
        input_dim=input_dim,
        hidden_dim=64,             # Reduced from 128
        num_layers=2,
        n_classes=n_classes,
        dropout=0.5                # Increased from 0.2
    ).to(DEVICE)

    weight_tensor = torch.FloatTensor(class_weights).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    
    # WEIGHT DECAY APPLIED HERE
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    train_ds = TimeSeriesDataset(X_train, y_train, SEQ_LENGTH)
    test_ds = TimeSeriesDataset(X_test, y_test, SEQ_LENGTH)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    train_losses = []
    val_losses = []
    best_val_loss = np.inf
    best_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                outputs = model(batch_x)
                val_loss += criterion(outputs, batch_y).item()

        avg_val_loss = val_loss / max(len(test_loader), 1)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        logger.info(
            f"    LSTM Epoch {epoch+1}/{EPOCHS} "
            f"train_loss={avg_train_loss:.4f} "
            f"val_loss={avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                logger.info(
                    f"    LSTM early stopping at epoch {epoch+1}, "
                    f"best val_loss={best_val_loss:.4f}"
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    all_probs = []
    all_preds = []
    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(DEVICE)
            outputs = model(batch_x)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)

    probs_array = np.array(all_probs)

    return model, probs_array, np.array(all_preds), train_losses, val_losses


def train_cnn_gaf(X_train, y_train, X_test, y_test, class_weights, n_classes, seq_length):
    train_ds = TimeSeriesDataset(X_train, y_train, seq_length)
    test_ds = TimeSeriesDataset(X_test, y_test, seq_length)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    gaf = GramianAngularField(image_size=seq_length, method="summation")
    n_features = X_train.shape[1]

    model = CNNGAFClassifier(
        n_features=n_features,
        seq_length=seq_length,
        n_classes=n_classes,
    ).to(DEVICE)

    weight_tensor = torch.FloatTensor(class_weights).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    val_losses = []
    best_val_loss = np.inf
    best_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch_x, batch_y in train_loader:
            b_size, seq_len, n_feat = batch_x.shape
            batch_np = batch_x.permute(0, 2, 1).numpy()

            gaf_images = np.zeros((b_size, n_feat, seq_len, seq_len))
            for f in range(n_feat):
                gaf_images[:, f] = gaf.fit_transform(batch_np[:, f, :])

            batch_x_gaf = torch.FloatTensor(gaf_images).to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(batch_x_gaf)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                b_size, seq_len, n_feat = batch_x.shape
                batch_np = batch_x.permute(0, 2, 1).numpy()
                gaf_images = np.zeros((b_size, n_feat, seq_len, seq_len))
                for f in range(n_feat):
                    gaf_images[:, f] = gaf.fit_transform(batch_np[:, f, :])
                batch_x_gaf = torch.FloatTensor(gaf_images).to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                val_loss += criterion(model(batch_x_gaf), batch_y).item()

        avg_val_loss = val_loss / max(len(test_loader), 1)
        val_losses.append(avg_val_loss)

        logger.info(
            f"    CNN-GAF Epoch {epoch+1}/{EPOCHS} "
            f"train_loss={avg_train_loss:.4f} "
            f"val_loss={avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                logger.info(
                    f"    CNN-GAF early stopping at epoch {epoch+1}, "
                    f"best val_loss={best_val_loss:.4f}"
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    all_probs = []
    all_preds = []
    with torch.no_grad():
        for batch_x, _ in test_loader:
            b_size, seq_len, n_feat = batch_x.shape
            batch_np = batch_x.permute(0, 2, 1).numpy()
            gaf_images = np.zeros((b_size, n_feat, seq_len, seq_len))
            for f in range(n_feat):
                gaf_images[:, f] = gaf.fit_transform(batch_np[:, f, :])
            batch_x_gaf = torch.FloatTensor(gaf_images).to(DEVICE)
            outputs = model(batch_x_gaf)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)

    return model, np.array(all_probs), np.array(all_preds), train_losses, val_losses


def compute_shap(model_obj, X_train, X_test, feature_names, model_name):
    logger.info(f"    Computing SHAP values for {model_name}")
    try:
        if model_name in ["rf", "xgb"]:
            explainer = shap.TreeExplainer(model_obj)
            sample = X_test[:1000] if len(X_test) > 1000 else X_test
            shap_values = explainer.shap_values(sample)
        elif model_name == "lr":
            explainer = shap.LinearExplainer(model_obj, X_train)
            sample = X_test[:1000] if len(X_test) > 1000 else X_test
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
        ds = TimeSeriesDataset(X_test, y_test, SEQ_LENGTH)
        loader = DataLoader(ds, batch_size=64, shuffle=False)

        ig = IntegratedGradients(model)
        all_attrs = []

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(DEVICE).requires_grad_(True)
            baseline = torch.zeros_like(batch_x).to(DEVICE)

            for c in range(n_classes):
                attrs = ig.attribute(batch_x, baseline, target=c)
                all_attrs.append(attrs.mean(dim=1).abs().cpu().detach().numpy())

            if len(all_attrs) >= 10:
                break

        mean_attrs = np.mean(all_attrs, axis=0)
        importance = dict(zip(feature_names, mean_attrs.mean(axis=0)))
        return importance
    except Exception as e:
        logger.warning(f"    Integrated Gradients failed: {e}")
        return None


def plot_confusion_matrix(y_true, y_pred, labels, title, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_curves(y_true, probs_dict, n_classes, title, save_path):
    fig, axes = plt.subplots(1, n_classes, figsize=(6 * n_classes, 5))
    class_names = ["calm", "elevated", "stress"]

    for c in range(n_classes):
        ax = axes[c] if n_classes > 1 else axes
        y_bin = (y_true == c).astype(int)

        for model_name, probs in probs_dict.items():
            if probs is None or len(probs) != len(y_bin):
                continue
            try:
                if probs.ndim == 2:
                    prob_c = probs[:, c]
                else:
                    prob_c = probs
                fpr, tpr, _ = roc_curve(y_bin, prob_c)
                auc = roc_auc_score(y_bin, prob_c)
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
    ax.plot(val_losses, label="Val Loss")
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
        if isinstance(shap_values, list):
            shap.summary_plot(
                shap_values[2], X_sample,
                feature_names=feature_names,
                show=False,
                plot_type="bar"
            )
        else:
            shap.summary_plot(
                shap_values, X_sample,
                feature_names=feature_names,
                show=False,
                plot_type="bar"
            )
        plt.title(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        logger.warning(f"    SHAP plot failed: {e}")


def upload_to_gcs(bucket, local_path, gcs_path):
    try:
        bucket.blob(gcs_path).upload_from_filename(local_path)
        logger.info(f"    Uploaded {gcs_path}")
    except Exception as e:
        logger.warning(f"    Upload failed for {gcs_path}: {e}")


def run_fold(
    gcs_client, bucket, asset, fold,
    train_windows, test_window,
    feature_cols, is_pooled=False,
    pooled_assets=None
):
    tag = f"{'pooled' if is_pooled else asset} fold {fold}"

    if fold_already_complete(bucket, asset, fold, is_pooled):
        logger.info(f"  {tag} - already complete, skipping")
        return

    logger.info(f"  {tag} - loading data")

    if is_pooled:
        train_frames = []
        test_frames = []
        for a in pooled_assets:
            for w in train_windows:
                df = load_window_data(bucket, a, w, feature_cols)
                if df is not None:
                    df = df.with_columns(
                        pl.lit(pooled_assets.index(a)).alias("asset_id")
                    )
                    train_frames.append(df)
            df = load_window_data(bucket, a, test_window, feature_cols)
            if df is not None:
                df = df.with_columns(
                    pl.lit(pooled_assets.index(a)).alias("asset_id")
                )
                test_frames.append(df)

        if not train_frames or not test_frames:
            logger.warning(f"  {tag} insufficient data, skipping")
            return

        train_df = pl.concat(train_frames).sort("time")
        test_df = pl.concat(test_frames).sort("time")
    else:
        train_frames = [
            load_window_data(bucket, asset, w, feature_cols)
            for w in train_windows
        ]
        train_df = pl.concat(
            [f for f in train_frames if f is not None]
        ).sort("time")
        test_df = load_window_data(bucket, asset, test_window, feature_cols)

    if test_df is None or train_df.is_empty():
        logger.warning(f"  {tag} no data, skipping")
        return

    logger.info(
        f"  {tag} - train: {len(train_df):,} rows, "
        f"test: {len(test_df):,} rows"
    )

    output_dir = f"/tmp/{asset}_fold{fold}"
    os.makedirs(output_dir, exist_ok=True)

    for binary in [False, True]:
        mode = "binary" if binary else "multiclass"
        n_classes = 2 if binary else N_CLASSES
        label_names = (
            ["not_stress", "stress"] if binary
            else ["calm", "elevated", "stress"]
        )

        logger.info(f"  {tag} - {mode} training")

        X_train, y_train, feat_names = prepare_xy(
            train_df, binary=binary, purge_end=True
        )
        X_test, y_test, _ = prepare_xy(
            test_df, binary=binary, purge_end=False
        )

        class_weights = get_class_weights(y_train, n_classes)
        logger.info(f"    Class weights: {class_weights}")

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)

        predictions = {}
        metrics = {}

        logger.info("    Training Logistic Regression")
        lr_weights = {i: w for i, w in enumerate(class_weights)}
        lr = LogisticRegression(
            class_weight=lr_weights, max_iter=2000, n_jobs=-1, C=1.0
        )
        lr.fit(X_train_sc, y_train)
        lr_probs = lr.predict_proba(X_test_sc)
        lr_preds = lr.predict(X_test_sc)
        predictions["lr"] = lr_probs
        metrics["lr"] = classification_report(
            y_test, lr_preds, target_names=label_names, output_dict=True
        )

        shap_result = compute_shap(lr, X_train_sc, X_test_sc, feat_names, "lr")
        if shap_result:
            plot_shap_summary(
                shap_result[0], shap_result[1], feat_names,
                f"LR SHAP - {tag} {mode}",
                f"{output_dir}/shap_lr_{mode}.png"
            )

        logger.info("    Training Random Forest")
        rf_weights = {i: w for i, w in enumerate(class_weights)}
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=12,
            class_weight=rf_weights, n_jobs=-1, random_state=42
        )
        rf.fit(X_train_sc, y_train)
        rf_probs = rf.predict_proba(X_test_sc)
        rf_preds = rf.predict(X_test_sc)
        predictions["rf"] = rf_probs
        metrics["rf"] = classification_report(
            y_test, rf_preds, target_names=label_names, output_dict=True
        )

        shap_result = compute_shap(rf, X_train_sc, X_test_sc, feat_names, "rf")
        if shap_result:
            plot_shap_summary(
                shap_result[0], shap_result[1], feat_names,
                f"RF SHAP - {tag} {mode}",
                f"{output_dir}/shap_rf_{mode}.png"
            )

        logger.info("    Training XGBoost")
        sample_weight = np.array([class_weights[int(y)] for y in y_train])
        xgb_params = dict(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            tree_method="hist", device="cuda" if DEVICE.type == "cuda" else "cpu",
            n_jobs=-1, random_state=42,
            objective="multi:softprob" if not binary else "binary:logistic",
            num_class=n_classes if not binary else None,
            eval_metric="mlogloss" if not binary else "logloss",
        )
        if binary:
            xgb_params.pop("num_class")
        xgb_model = xgb.XGBClassifier(**xgb_params)
        xgb_model.fit(X_train_sc, y_train, sample_weight=sample_weight)
        xgb_probs = xgb_model.predict_proba(X_test_sc)
        xgb_preds = xgb_model.predict(X_test_sc)
        predictions["xgb"] = xgb_probs
        metrics["xgb"] = classification_report(
            y_test, xgb_preds, target_names=label_names, output_dict=True
        )

        shap_result = compute_shap(
            xgb_model, X_train_sc, X_test_sc, feat_names, "xgb"
        )
        if shap_result:
            plot_shap_summary(
                shap_result[0], shap_result[1], feat_names,
                f"XGB SHAP - {tag} {mode}",
                f"{output_dir}/shap_xgb_{mode}.png"
            )

        logger.info("    Training LSTM")
        lstm_model, lstm_probs, lstm_preds, lstm_tl, lstm_vl = train_lstm(
            X_train_sc, y_train, X_test_sc, y_test, class_weights, n_classes
        )
        predictions["lstm"] = lstm_probs
        metrics["lstm"] = classification_report(
            y_test[SEQ_LENGTH - 1:], lstm_preds,
            target_names=label_names, output_dict=True
        )

        plot_loss_curves(
            lstm_tl, lstm_vl, "LSTM", f"{tag} {mode}",
            f"{output_dir}/loss_lstm_{mode}.png"
        )

        ig_result = compute_integrated_gradients(
            lstm_model, X_test_sc, y_test, feat_names, n_classes
        )
        if ig_result:
            with open(f"{output_dir}/ig_lstm_{mode}.json", "w") as f:
                json.dump(ig_result, f, indent=2)

        logger.info("    Training CNN-GAF")
        gaf_feat_idx = [
            feat_names.index(f) for f in GAF_FEATURES if f in feat_names
        ]
        X_train_gaf = X_train_sc[:, gaf_feat_idx]
        X_test_gaf = X_test_sc[:, gaf_feat_idx]

        cnn_model, cnn_probs, cnn_preds, cnn_tl, cnn_vl = train_cnn_gaf(
            X_train_gaf, y_train, X_test_gaf, y_test,
            class_weights, n_classes, SEQ_LENGTH
        )
        predictions["cnn_gaf"] = cnn_probs
        metrics["cnn_gaf"] = classification_report(
            y_test[SEQ_LENGTH - 1:], cnn_preds,
            target_names=label_names, output_dict=True
        )

        plot_loss_curves(
            cnn_tl, cnn_vl, "CNN-GAF", f"{tag} {mode}",
            f"{output_dir}/loss_cnn_{mode}.png"
        )

        for model_name, preds in [
            ("lr", lr_preds), ("rf", rf_preds),
            ("xgb", xgb_preds),
        ]:
            plot_confusion_matrix(
                y_test, preds, label_names,
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

        pred_df = pl.DataFrame(pred_records)
        pred_path = f"{output_dir}/predictions_{mode}.parquet"
        pred_df.write_parquet(pred_path)

        models_dir = f"{output_dir}/models"
        os.makedirs(models_dir, exist_ok=True)

        for model_name, model_obj in [
            ("lr", lr), ("rf", rf), ("xgb", xgb_model)
        ]:
            with open(f"{models_dir}/{model_name}_{mode}.pkl", "wb") as f:
                pickle.dump({"model": model_obj, "scaler": scaler}, f)

        torch.save(lstm_model.state_dict(), f"{models_dir}/lstm_{mode}.pt")
        torch.save(cnn_model.state_dict(), f"{models_dir}/cnn_gaf_{mode}.pt")

        gc.collect()

    prefix = (
        f"{RESULTS_PREFIX}pooled/fold_{fold}/"
        if is_pooled
        else f"{RESULTS_PREFIX}{asset}/fold_{fold}/"
    )

    for root, dirs, files in os.walk(output_dir):
        for file in files:
            local_path = os.path.join(root, file)
            relative = os.path.relpath(local_path, output_dir)
            gcs_path = f"{prefix}{relative}"
            upload_to_gcs(bucket, local_path, gcs_path)

    logger.info(f"  {tag} - complete, results uploaded to gs://{BUCKET}/{prefix}")


def main():
    logger.info(f"Starting Training Pipeline")
    logger.info(f"Device       : {DEVICE}")
    logger.info(f"SEQ_LENGTH   : {SEQ_LENGTH}")
    logger.info(f"EPOCHS       : {EPOCHS}")
    logger.info(f"N_CLASSES    : {N_CLASSES}")
    logger.info(f"PURGE_ROWS   : {PURGE_ROWS}")
    logger.info(f"EARLY_STOP   : patience={EARLY_STOP_PATIENCE}")

    gcs_client = storage.Client()
    bucket = gcs_client.bucket(BUCKET)

    n_folds = len(WINDOWS) - 1

    for asset in ASSETS:
        logger.info(f"{'=' * 60}")
        logger.info(f"Asset-specific training: {asset}")

        asset_features = None

        for fold in range(1, n_folds + 1):
            train_windows = list(range(0, fold))
            test_window = fold

            logger.info(
                f"  Fold {fold}/{n_folds}: "
                f"train={train_windows} test={test_window}"
            )

            run_fold(
                gcs_client, bucket, asset, fold,
                train_windows, test_window,
                feature_cols=asset_features,
                is_pooled=False,
            )

            gc.collect()

    logger.info(f"{'=' * 60}")
    logger.info("Pooled training: all assets")

    for fold in range(1, n_folds + 1):
        train_windows = list(range(0, fold))
        test_window = fold

        logger.info(
            f"  Fold {fold}/{n_folds}: "
            f"train={train_windows} test={test_window}"
        )

        run_fold(
            gcs_client, bucket, "pooled", fold,
            train_windows, test_window,
            feature_cols=COMMON_FEATURES,
            is_pooled=True,
            pooled_assets=ASSETS,
        )

        gc.collect()

    logger.info("=" * 60)
    logger.info("Training Pipeline Complete")


if __name__ == "__main__":
    main()

