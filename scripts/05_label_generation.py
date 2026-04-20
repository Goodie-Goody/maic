import sys
import os
import gc
import logging
import tempfile
import pickle
from datetime import date

import numpy as np
import polars as pl
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from google.cloud import storage

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
N_STATES = 3
N_ITER = 100

HMM_FEATURES = [
    "RV_300s",
    "OFI_300s",
    "Kyle_lambda_300s",
    "intensity_300s",
]

STATE_NAMES = {0: "calm", 1: "elevated", 2: "stress"}


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


def get_expected_months(asset):
    months = []
    for start_ym, end_ym in WINDOWS:
        for year, month in parse_window_months(start_ym, end_ym):
            if asset == "SOLUSDT" and (year, month) < (2020, 11):
                continue
            months.append((year, month))
    return months


def load_asset_features(gcs_client, bucket, asset):
    logger.info(f"Loading all {asset} feature files for global HMM fit")
    months = get_expected_months(asset)
    frames = []

    for year, month in months:
        blob_path = f"{FEATURES_PREFIX}{asset}-features-{year}-{month:02d}.parquet"
        blob = bucket.blob(blob_path)

        if not blob.exists():
            logger.warning(f"  Missing {blob_path}, skipping")
            continue

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet")
        try:
            blob.download_to_filename(tmp.name)
            df = pl.read_parquet(tmp.name, columns=["time"] + HMM_FEATURES)
            df = df.with_columns([
                pl.lit(year).alias("year"),
                pl.lit(month).alias("month"),
            ])
            frames.append(df)
        finally:
            os.remove(tmp.name)

    if not frames:
        raise RuntimeError(f"No feature files found for {asset}")

    combined = pl.concat(frames)
    logger.info(f"  Loaded {combined.shape[0]:,} observations for {asset}")
    return combined


def fit_hmm(df, asset):
    logger.info(f"Fitting {N_STATES}-state Gaussian HMM on {asset}")
    logger.info(f"  Covariance type: diag")
    logger.info(f"  Running 5 seeds, keeping best log likelihood")

    X = df.select(HMM_FEATURES).to_numpy()
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_model = None
    best_score = -np.inf
    best_seed_val = 0

    for seed in range(5):
        model = GaussianHMM(
            n_components=N_STATES,
            covariance_type="diag",
            n_iter=N_ITER,
            random_state=seed,
            verbose=False,
        )
        model.fit(X_scaled)
        score = model.score(X_scaled)
        logger.info(f"  Seed {seed}: log likelihood {score:.4f}, converged: {model.monitor_.converged}")
        if score > best_score:
            best_score = score
            best_model = model
            best_seed_val = seed

    logger.info(f"  Best log likelihood: {best_score:.4f} (Seed {best_seed_val})")

    states = best_model.predict(X_scaled)

    rv_idx = HMM_FEATURES.index("RV_300s")
    state_means = {
        s: best_model.means_[s][rv_idx]
        for s in range(N_STATES)
    }

    sorted_states = sorted(state_means.items(), key=lambda x: x[1])
    state_map = {
        sorted_states[0][0]: 0,
        sorted_states[1][0]: 1,
        sorted_states[2][0]: 2,
    }

    mapped_states = np.array([state_map[s] for s in states])

    for label, name in STATE_NAMES.items():
        count = (mapped_states == label).sum()
        pct = count / len(mapped_states) * 100
        logger.info(f"  State {label} ({name}): {count:,} observations ({pct:.1f}%)")

    logger.info("  Transition matrix (rows=from, cols=to):")
    tm = best_model.transmat_
    for i in range(N_STATES):
        row = " ".join([f"{tm[i][j]:.3f}" for j in range(N_STATES)])
        logger.info(f"    State {i}: [{row}]")

    os.makedirs("logs", exist_ok=True)
    model_path = f"logs/{asset}_hmm_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": best_model,
            "scaler": scaler,
            "state_map": state_map,
            "hmm_features": HMM_FEATURES,
            "n_states": N_STATES,
            "covariance_type": "diag",
            "best_seed": best_seed_val,
        }, f)
    logger.info(f"  Saved model and scaler to {model_path}")

    return mapped_states, best_model, scaler, state_map


def validate_against_events(df, states, asset):
    known_stress_periods = [
        ("COVID crash", "2020-03-12", "2020-03-13"),
        ("May 2021 crash", "2021-05-19", "2021-05-20"),
        ("Terra-Luna", "2022-05-09", "2022-05-12"),
        ("FTX collapse", "2022-11-08", "2022-11-11"),
    ]

    df = df.with_columns([
        pl.Series("label", states),
        pl.col("time").dt.date().alias("date"),
    ])

    logger.info(f"Validating {asset} labels against known stress events")

    for event_name, start_str, end_str in known_stress_periods:
        start = date(int(start_str[:4]), int(start_str[5:7]), int(start_str[8:]))
        end = date(int(end_str[:4]), int(end_str[5:7]), int(end_str[8:]))

        period = df.filter(
            (pl.col("date") >= start) & (pl.col("date") <= end)
        )

        if period.is_empty():
            logger.info(f"  {event_name}: not in dataset windows")
            continue

        stress_pct = (period["label"] == 2).mean() * 100
        elevated_pct = (period["label"] == 1).mean() * 100
        calm_pct = (period["label"] == 0).mean() * 100
        logger.info(
            f"  {event_name}: {stress_pct:.1f}% stress, "
            f"{elevated_pct:.1f}% elevated, "
            f"{calm_pct:.1f}% calm"
        )


def save_labels(gcs_client, bucket, asset, df, states):
    logger.info(f"Saving labels for {asset}")

    df = df.with_columns([
        pl.Series("label", states),
    ])

    months = df.select(["year", "month"]).unique().sort(["year", "month"])

    for row in months.iter_rows(named=True):
        year = row["year"]
        month = row["month"]

        month_df = df.filter(
            (pl.col("year") == year) & (pl.col("month") == month)
        ).select(["time", "label"])

        output_blob = f"{LABELS_PREFIX}{asset}-labels-{year}-{month:02d}.parquet"

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet")
        try:
            month_df.write_parquet(tmp.name, compression="snappy")
            bucket.blob(output_blob).upload_from_filename(tmp.name)
            logger.info(
                f"  Saved {asset} {year}-{month:02d}: "
                f"{month_df.shape[0]:,} rows"
            )
        finally:
            os.remove(tmp.name)
            gc.collect()


def main():
    gcs_client = storage.Client()
    bucket = gcs_client.bucket(BUCKET)

    logger.info("Starting Label Generation Pipeline (Targeted Rerun)")
    logger.info(f"States       : {N_STATES}")
    logger.info(f"HMM features : {HMM_FEATURES}")
    logger.info(f"Iterations   : {N_ITER}")
    logger.info(f"Scaling      : StandardScaler (zero mean, unit variance)")

    failed = []

    for asset in ASSETS:
        logger.info(f"{'=' * 60}")
        logger.info(f"Processing {asset}")

        try:
            df = load_asset_features(gcs_client, bucket, asset)
            mapped_states, best_model, scaler, state_map = fit_hmm(df, asset)
            validate_against_events(df, mapped_states, asset)
            save_labels(gcs_client, bucket, asset, df, mapped_states)

        except Exception as e:
            logger.error(f"{asset} failed: {e}")
            failed.append(asset)
            continue

        gc.collect()

    logger.info("=" * 60)
    logger.info("Targeted Label Generation Complete")

    if failed:
        logger.error(f"Failed assets: {failed}")
        sys.exit(1)

    logger.info("ETHUSDT labeled successfully")


if __name__ == "__main__":
    main()

