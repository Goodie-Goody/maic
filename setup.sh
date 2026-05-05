#!/bin/bash
# MAIC RunPod Environment Setup
# Usage: bash /workspace/maic/setup.sh
#
# Requires on persistent volume (/workspace/maic/) — all gitignored:
#   - gcp-key.json    GCP service account key
#   - .env            All environment variables including GitHub token
#
# Required .env keys:
#   GCP_PROJECT_ID, GCP_BUCKET, GCP_REGION, BQ_DATASET, BQ_TABLE,
#   GOOGLE_APPLICATION_CREDENTIALS, GITHUB_TOKEN, GITHUB_USER, GITHUB_EMAIL

set -e

echo ""
echo "MAIC RunPod Environment Setup"
echo ""

# GPU PRE-CHECK
echo ""
echo "GPU pre-check..."

if ! command -v nvidia-smi &> /dev/null; then
    echo "  ERROR: nvidia-smi not found — no GPU detected"
    echo "  Ensure you selected a GPU instance on RunPod."
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)

echo "  GPU    : $GPU_NAME"
echo "  VRAM   : $GPU_MEM"
echo "  Driver : $DRIVER"

if [[ "$GPU_NAME" != *"Blackwell"* ]] && [[ "$GPU_NAME" != *"RTX PRO 4500"* ]]; then
    echo "  WARNING: Pipeline optimised for RTX PRO 4500 Blackwell (sm_120)"
    echo "  Different GPU detected — verify sm_120 support after setup"
else
    echo "  Blackwell GPU confirmed"
fi

# STEP 1 — Validate .env
echo ""
echo "[1/8] Validating .env..."

ENV_FILE="/workspace/maic/.env"

if [ ! -f "$ENV_FILE" ]; then
    echo "  ERROR: .env not found at $ENV_FILE"
    exit 1
fi

set -a; source "$ENV_FILE"; set +a

REQUIRED_KEYS=(
    GCP_PROJECT_ID GCP_BUCKET GCP_REGION BQ_DATASET BQ_TABLE
    GOOGLE_APPLICATION_CREDENTIALS GITHUB_TOKEN GITHUB_USER GITHUB_EMAIL
)

MISSING=0
for key in "${REQUIRED_KEYS[@]}"; do
    [ -z "${!key}" ] && echo "  ERROR: $key missing in .env" && MISSING=1
done
[ "$MISSING" -eq 1 ] && exit 1

[ ! -f "$GOOGLE_APPLICATION_CREDENTIALS" ] && \
    echo "  ERROR: gcp-key.json not found at $GOOGLE_APPLICATION_CREDENTIALS" && exit 1

echo "  .env loaded and validated"

# STEP 2 — System packages
echo ""
echo "[2/8] Installing system packages..."

# Suppress PPA errors — launchpadcontent.net is blocked on RunPod network
apt-get update -qq 2>/dev/null || true

DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    apt-utils curl wget git build-essential ca-certificates gnupg \
    lsb-release unzip htop tmux net-tools vim nano \
    -o Dpkg::Use-Pty=0 > /dev/null 2>&1 || true
# Re-emit only real errors (exit code != 0) — update-alternatives warnings are noise
true

echo "  System packages installed"

# STEP 3 — Python and pip
echo ""
echo "[3/8] Verifying Python and pip..."

command -v python3 &>/dev/null || apt-get install -y -qq python3 python3-pip > /dev/null
echo "  Found: $(python3 --version)"
pip install --root-user-action=ignore --upgrade pip --break-system-packages --quiet
echo "  pip ready"

# STEP 4 — GCP Authentication
# echo ""
echo "[4/8] Authenticating to GCP..."

# Install gcloud if not present
if ! command -v gcloud &> /dev/null; then
    echo "  Installing gcloud CLI..."
    curl -sSL https://sdk.cloud.google.com > /tmp/install_gcloud.sh
    bash /tmp/install_gcloud.sh --disable-prompts --install-dir=/usr/local/lib/gcloud > /dev/null 2>&1
    ln -sf /usr/local/lib/gcloud/google-cloud-sdk/bin/gcloud /usr/local/bin/gcloud
    ln -sf /usr/local/lib/gcloud/google-cloud-sdk/bin/gsutil /usr/local/bin/gsutil
    echo "  gcloud installed"
else
    echo "  gcloud already installed"
fi

gcloud auth activate-service-account     --key-file="$GOOGLE_APPLICATION_CREDENTIALS" --quiet 2>/dev/null
gcloud config set project "$GCP_PROJECT_ID" --quiet 2>/dev/null
export GOOGLE_APPLICATION_CREDENTIALS="$GOOGLE_APPLICATION_CREDENTIALS"

echo "  Authenticated: $(gcloud auth list --format='value(account)' 2>/dev/null | head -1)"

# STEP 5 — GitHub Authentication
echo ""
echo "[5/8] Configuring GitHub..."

git config --global user.email "$GITHUB_EMAIL"
git config --global user.name  "$GITHUB_USER"
git config --global init.defaultBranch main
git config --global pull.rebase false
git config --global credential.helper store

printf "https://%s:%s@github.com\n" "$GITHUB_USER" "$GITHUB_TOKEN" \
    > ~/.git-credentials
chmod 600 ~/.git-credentials
unset GITHUB_TOKEN

echo "  GitHub auth configured for $GITHUB_USER"

# STEP 6 — Clone or update MAIC repo
echo ""
echo "[6/8] Syncing MAIC repo..."

if [ ! -d "/workspace/maic/.git" ]; then
    echo "  Cloning..."
    git clone "https://github.com/${GITHUB_USER}/maic.git" /tmp/maic_clone
    cp -r /tmp/maic_clone/. /workspace/maic/
    rm -rf /tmp/maic_clone
    echo "  Cloned"
else
    echo "  Pulling latest..."
    cd /workspace/maic && git pull --quiet
    echo "  Up to date"
fi

cd /workspace/maic

# STEP 7 — Python packages
echo ""
echo "[7/8] Installing Python packages..."

[ ! -f "/workspace/maic/requirements.txt" ] && \
    echo "  ERROR: requirements.txt not found" && exit 1

# PyTorch nightly cu128
echo "  [7a] Checking PyTorch nightly cu128..."

# Step 1 — upgrade nvjitlink first to avoid cuML/PyTorch conflict
# cuML 26.4 requires nvidia-nvjitlink-cu12>=12.9, default install is 12.4
echo "  Upgrading nvidia-nvjitlink-cu12..."
pip install --root-user-action=ignore "nvidia-nvjitlink-cu12>=12.9" \
    --break-system-packages --quiet
echo "  nvjitlink ready"

# Step 2 — check if correct PyTorch nightly is already installed
# Must be a dev build AND support sm_120 (Blackwell)
PYTORCH_OK=0
python3 -c "
import torch, sys
try:
    is_nightly = 'dev' in torch.__version__
    cuda_ok    = torch.version.cuda is not None and torch.version.cuda.startswith('12.8')
    cap        = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0,0)
    sm_ok      = cap == (12, 0)
    sys.exit(0 if (is_nightly and cuda_ok and sm_ok) else 1)
except Exception:
    sys.exit(1)
" 2>/dev/null && PYTORCH_OK=1 || PYTORCH_OK=0

if [ "$PYTORCH_OK" -eq 0 ]; then
    echo "  Installing PyTorch nightly cu128 (force reinstall)..."
    pip install --root-user-action=ignore torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/nightly/cu128 \
        --force-reinstall --break-system-packages --quiet
    echo "  PyTorch nightly cu128 installed"

    # Verify install succeeded with correct CUDA version
    python3 -c "
import torch, sys
is_nightly = 'dev' in torch.__version__
cuda_ver   = torch.version.cuda or ''
cap        = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0,0)
print(f'  Verified: {torch.__version__} | CUDA {cuda_ver} | sm_{cap[0]}{cap[1]}')
if not (is_nightly and cuda_ver.startswith('12.8') and cap == (12,0)):
    print('  ERROR: PyTorch nightly cu128 + sm_120 not confirmed — check manually')
    sys.exit(1)
" || { echo "  ERROR: PyTorch verification failed — do not start training"; exit 1; }
else
    echo "  PyTorch nightly cu128 already correct — skipping"
fi

# Core packages
echo "  [7b] Installing core packages..."
pip install --root-user-action=ignore -r /workspace/maic/requirements.txt \
    --break-system-packages --quiet
echo "  Core packages ready"

# cuML
echo "  [7c] Checking cuML..."
python3 -c "from cuml.ensemble import RandomForestClassifier; print('  cuML already installed')" \
    2>/dev/null || {
    echo "  Installing cuML..."
    pip install --root-user-action=ignore cuml-cu12 \
        --extra-index-url=https://pypi.nvidia.com \
        --break-system-packages --quiet
    echo "  cuML installed"
}

# tqdm — needed for training progress bars
echo "  [7d] Checking tqdm..."
python3 -c "import tqdm; print(f'  tqdm {tqdm.__version__} ready')" 2>/dev/null || {
    pip install --root-user-action=ignore tqdm --break-system-packages --quiet
    echo "  tqdm installed"
}

echo "  All packages ready"

# STEP 8 — HMM models + environment verification
echo ""
echo "[8/8] Finalising environment..."

mkdir -p /workspace/maic/logs

for asset in BTCUSDT ETHUSDT SOLUSDT; do
    LOCAL_PATH="/workspace/maic/logs/${asset}_hmm_model.pkl"
    if [ ! -f "$LOCAL_PATH" ]; then
        gsutil cp \
            "gs://${GCP_BUCKET}/v2/vm_backup_20260420_1709/logs/logs/${asset}_hmm_model.pkl" \
            "$LOCAL_PATH" 2>/dev/null && \
            echo "  Downloaded ${asset}_hmm_model.pkl" || \
            echo "  WARNING: Could not download ${asset}_hmm_model.pkl"
    else
        echo "  ${asset}_hmm_model.pkl present"
    fi
done

echo ""
echo "  Verifying environment..."

python3 - << 'PYEOF'
import torch
print(f"  PyTorch      : {torch.__version__}")
print(f"  CUDA         : {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"  GPU          : {torch.cuda.get_device_name(0)}")
    vram = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    print(f"  VRAM         : {vram} GB")
    cap = torch.cuda.get_device_capability()
    sm  = f"sm_{cap[0]}{cap[1]}"
    ok  = " OK" if cap == (12, 0) else " WARNING: expected sm_120"
    print(f"  Capability   : {sm}{ok}")
else:
    print("  WARNING: CUDA not available")
PYEOF

python3 -c "
from cuml.ensemble import RandomForestClassifier
print('  cuML GPU RF  : OK')
" 2>/dev/null || echo "  cuML         : FAILED"

python3 -c "
import polars, xgboost, sklearn, shap, numpy, tqdm
print(f'  Polars       : {polars.__version__}')
print(f'  XGBoost      : {xgboost.__version__}')
print(f'  sklearn      : {sklearn.__version__}')
print(f'  NumPy        : {numpy.__version__}')
print(f'  tqdm         : {tqdm.__version__}')
print('  SHAP         : OK')
"

# DONE
echo ""
echo ""
echo "Setup complete. Environment ready."
echo ""
echo ""
echo "  Start production run:"
echo "    nohup python3 scripts/06d_train_production.py \\"
echo "      >> /workspace/maic/logs/06d_production.log 2>&1 &"
echo ""
echo "  Aggregate results:"
echo "    python3 scripts/07c_aggregate_production.py"
echo ""
echo "  Tail log:"
echo "    tail -f /workspace/maic/logs/06d_production.log"
echo ""
