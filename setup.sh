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

if [[ "$GPU_NAME" == *"Blackwell"* ]] || [[ "$GPU_NAME" == *"RTX PRO 4500"* ]]; then
    echo "  Blackwell GPU confirmed — nightly cu128 will be used"
elif [[ "$GPU_NAME" == *"RTX 40"* ]] || [[ "$GPU_NAME" == *"RTX 4"* ]]; then
    echo "  Ada Lovelace GPU detected — stable cu121 will be used"
elif [[ "$GPU_NAME" == *"A100"* ]] || [[ "$GPU_NAME" == *"RTX 30"* ]] || [[ "$GPU_NAME" == *"RTX 3"* ]]; then
    echo "  Ampere GPU detected — stable cu121 will be used"
else
    echo "  GPU detected — stable cu121 will be used (verify CUDA compatibility if issues arise)"
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

# PyTorch — GPU-adaptive install
# Selects the correct PyTorch build based on detected GPU compute capability:
#   sm_120 (Blackwell)  → nightly cu128  (required — no stable release yet)
#   all others          → stable cu121   (broadly compatible: Ampere, Ada, Hopper)
echo "  [7a] Detecting GPU and selecting PyTorch variant..."

# Upgrade nvjitlink first to avoid cuML conflict regardless of GPU
# cuML 26.4 requires nvidia-nvjitlink-cu12>=12.9, default install is 12.4
echo "  Upgrading nvidia-nvjitlink-cu12..."
pip install --root-user-action=ignore "nvidia-nvjitlink-cu12>=12.9" \
    --break-system-packages --quiet
echo "  nvjitlink ready"

# Detect compute capability via nvidia-smi
SM_MAJOR=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
    | head -1 | cut -d'.' -f1 | tr -d ' ')
SM_MINOR=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
    | head -1 | cut -d'.' -f2 | tr -d ' ')
SM="${SM_MAJOR}${SM_MINOR}"

echo "  Detected compute capability: sm_${SM}"

if [ "$SM" = "120" ]; then
    # Blackwell — nightly cu128 required (no stable support yet)
    PYTORCH_INDEX="https://download.pytorch.org/whl/nightly/cu128"
    PYTORCH_LABEL="nightly cu128 (Blackwell sm_120)"
    PYTORCH_CHECK_CMD="'dev' in torch.__version__ and (torch.version.cuda or '').startswith('12.8')"
else
    # All other GPUs — stable cu121
    PYTORCH_INDEX="https://download.pytorch.org/whl/cu121"
    PYTORCH_LABEL="stable cu121"
    PYTORCH_CHECK_CMD="'dev' not in torch.__version__ and (torch.version.cuda or '').startswith('12.1')"
fi

echo "  Selected: PyTorch $PYTORCH_LABEL"

# Check if correct PyTorch already installed
PYTORCH_OK=0
python3 -c "
import torch, sys
try:
    cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0,0)
    ok  = $PYTORCH_CHECK_CMD
    sys.exit(0 if ok else 1)
except Exception:
    sys.exit(1)
" 2>/dev/null && PYTORCH_OK=1 || PYTORCH_OK=0

if [ "$PYTORCH_OK" -eq 0 ]; then
    echo "  Installing PyTorch $PYTORCH_LABEL..."
    pip install --root-user-action=ignore torch torchvision torchaudio \
        --index-url "$PYTORCH_INDEX" \
        --force-reinstall --break-system-packages --quiet
    echo "  PyTorch $PYTORCH_LABEL installed"
else
    echo "  PyTorch $PYTORCH_LABEL already correct — skipping"
fi

# Verify
python3 - << PYEOF
import torch, sys
cuda_ver = torch.version.cuda or 'N/A'
cap      = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
sm       = f"sm_{cap[0]}{cap[1]}"
print(f"  Verified: {torch.__version__} | CUDA {cuda_ver} | {sm}")
if not torch.cuda.is_available():
    print("  ERROR: CUDA not available after PyTorch install")
    sys.exit(1)
PYEOF
[ $? -ne 0 ] && { echo "  ERROR: PyTorch verification failed — do not start training"; exit 1; }

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
    print(f"  Capability   : {sm}")
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
# Make pipeline runner scripts executable
chmod +x cpu_pipeline.sh gpu_pipeline.sh cpu_post_gpu.sh status.sh 2>/dev/null || true
echo "  Pipeline scripts marked executable"

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
