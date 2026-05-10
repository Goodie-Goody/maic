#!/bin/bash
# MAIC Environment Setup (Fully Portable: RunPod & Local)
# Usage: bash setup.sh
#
# Requires in the repository root directory (gitignored):
#   - gcp-key.json    GCP service account key
#   - .env            All environment variables including GitHub token

set -e

# Dynamically determine the repository root based on where this script lives
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo ""
echo "========================================================"
echo "MAIC Environment Setup"
echo "Repository Root: $REPO_ROOT"
echo "========================================================"
echo ""

# GPU PRE-CHECK (Adaptive)
echo "[0/8] Hardware detection..."

HAS_GPU=0
if ! command -v nvidia-smi &> /dev/null; then
    echo "  No GPU detected (nvidia-smi not found)."
    echo "  Setting up for CPU-only execution (data prep and analysis)."
    HAS_GPU=0
else
    HAS_GPU=1
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
        echo "  GPU detected — stable cu121 will be used"
    fi
fi

# STEP 1 — Validate .env
echo ""
echo "[1/8] Validating .env..."

ENV_FILE="$REPO_ROOT/.env"

if [ ! -f "$ENV_FILE" ]; then
    echo "  ERROR: .env not found at $ENV_FILE"
    exit 1
fi

set -a; source "$ENV_FILE"; set +a

# Ensure credentials path is absolute if it was set relatively in .env
if [[ "$GOOGLE_APPLICATION_CREDENTIALS" != /* ]]; then
    export GOOGLE_APPLICATION_CREDENTIALS="$REPO_ROOT/$GOOGLE_APPLICATION_CREDENTIALS"
fi

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

# Setup sudo command if running locally as non-root
SUDO_CMD=""
if [ "$(id -u)" -ne 0 ] && command -v sudo &> /dev/null; then
    SUDO_CMD="sudo"
fi

$SUDO_CMD apt-get update -qq 2>/dev/null || true

DEBIAN_FRONTEND=noninteractive $SUDO_CMD apt-get install -y -qq \
    apt-utils curl wget git build-essential ca-certificates gnupg \
    lsb-release unzip htop tmux net-tools vim nano \
    -o Dpkg::Use-Pty=0 > /dev/null 2>&1 || true
true

echo "  System packages installed/verified"

# STEP 3 — Python and pip
echo ""
echo "[3/8] Verifying Python and pip..."

command -v python3 &>/dev/null || $SUDO_CMD apt-get install -y -qq python3 python3-pip > /dev/null
echo "  Found: $(python3 --version)"
pip install --root-user-action=ignore --upgrade pip --break-system-packages --quiet
echo "  pip ready"

# STEP 4 — GCP Authentication
echo ""
echo "[4/8] Authenticating to GCP..."

# 1. Check if it's already working
if ! command -v gcloud &> /dev/null; then
    echo "  gcloud not in PATH. Checking default installation directory..."
    
    # 2. If it's installed but not in PATH, fix the PATH for this session
    if [ -d "$HOME/google-cloud-sdk" ]; then
        echo "  Found SDK directory at $HOME/google-cloud-sdk. Repairing PATH..."
        export PATH="$PATH:$HOME/google-cloud-sdk/bin"
    else
        # 3. If it's actually missing, install it
        echo "  SDK not found on disk. Installing gcloud CLI..."
        curl -sSL https://sdk.cloud.google.com > /tmp/install_gcloud.sh
        # CLOUDSDK_PYTHON=python3 ensures it works on modern distros without a 'python' alias
        CLOUDSDK_PYTHON=python3 bash /tmp/install_gcloud.sh --disable-prompts --install-dir="$HOME" > /dev/null 2>&1 || {
            echo "  ERROR: gcloud installation failed."
            exit 1
        }
        export PATH="$PATH:$HOME/google-cloud-sdk/bin"
    fi
else
    echo "  gcloud already installed and active"
fi

# 4. Permanent PATH injection (Safe for re-runs)
# For Bash users
if [ -f "$HOME/.bashrc" ] && ! grep -q "google-cloud-sdk" "$HOME/.bashrc"; then
    echo 'export PATH="$PATH:$HOME/google-cloud-sdk/bin"' >> "$HOME/.bashrc"
    echo "  Added SDK to .bashrc"
fi

# For Fish users (like you)
if [ -d "$HOME/.config/fish" ] && ! grep -q "google-cloud-sdk" "$HOME/.config/fish/config.fish" 2>/dev/null; then
    echo "source $HOME/google-cloud-sdk/path.fish.inc" >> "$HOME/.config/fish/config.fish"
    echo "  Added SDK to Fish config"
fi

# 5. Final verification & Auth
# Explicitly use the full path just in case the export didn't propagate to the sub-shell
GCLOUD_BIN="$HOME/google-cloud-sdk/bin/gcloud"
$GCLOUD_BIN auth activate-service-account --key-file="$GOOGLE_APPLICATION_CREDENTIALS" --quiet 2>/dev/null
$GCLOUD_BIN config set project "$GCP_PROJECT_ID" --quiet 2>/dev/null

echo "  Authenticated: $($GCLOUD_BIN auth list --format='value(account)' 2>/dev/null | head -1)"

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

if [ ! -d "$REPO_ROOT/.git" ]; then
    echo "  WARNING: Directory is not a git repo. Initializing..."
    git clone "https://github.com/${GITHUB_USER}/maic.git" /tmp/maic_clone
    cp -r /tmp/maic_clone/. "$REPO_ROOT/"
    rm -rf /tmp/maic_clone
    echo "  Repo synced"
else
    echo "  Pulling latest..."
    cd "$REPO_ROOT" && git pull --quiet
    echo "  Up to date"
fi

cd "$REPO_ROOT"

# STEP 7 — Python packages
echo ""
echo "[7/8] Installing Python packages..."

[ ! -f "$REPO_ROOT/requirements.txt" ] && \
    echo "  ERROR: requirements.txt not found" && exit 1

if [ "$HAS_GPU" -eq 1 ]; then
    echo "  [7a] Detecting GPU and selecting PyTorch variant..."
    echo "  Upgrading nvidia-nvjitlink-cu12..."
    pip install --root-user-action=ignore "nvidia-nvjitlink-cu12>=12.9" --break-system-packages --quiet
    
    SM_MAJOR=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | cut -d'.' -f1 | tr -d ' ')
    SM_MINOR=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | cut -d'.' -f2 | tr -d ' ')
    SM="${SM_MAJOR}${SM_MINOR}"

    if [ "$SM" = "120" ]; then
        PYTORCH_INDEX="https://download.pytorch.org/whl/nightly/cu128"
        PYTORCH_LABEL="nightly cu128 (Blackwell sm_120)"
    else
        PYTORCH_INDEX="https://download.pytorch.org/whl/cu121"
        PYTORCH_LABEL="stable cu121"
    fi
else
    echo "  [7a] CPU detected. Selecting PyTorch CPU variant..."
    PYTORCH_INDEX="https://download.pytorch.org/whl/cpu"
    PYTORCH_LABEL="CPU Only"
fi

echo "  Installing PyTorch $PYTORCH_LABEL..."
pip install --root-user-action=ignore torch torchvision torchaudio \
    --index-url "$PYTORCH_INDEX" \
    --force-reinstall --break-system-packages --quiet

echo "  [7b] Installing core packages..."
pip install --root-user-action=ignore -r "$REPO_ROOT/requirements.txt" \
    --break-system-packages --quiet

if [ "$HAS_GPU" -eq 1 ]; then
    echo "  [7c] Installing cuML for GPU..."
    pip install --root-user-action=ignore cuml-cu12 \
        --extra-index-url=https://pypi.nvidia.com \
        --break-system-packages --quiet
else
    echo "  [7c] Skipping cuML (No GPU present)..."
fi

echo "  [7d] Checking tqdm..."
pip install --root-user-action=ignore tqdm --break-system-packages --quiet

echo "  All packages ready"

# STEP 8 — HMM models + environment verification
echo ""
echo "[8/8] Finalising environment..."

mkdir -p "$REPO_ROOT/logs"

# Download existing HMM models if available
for asset in BTCUSDT ETHUSDT SOLUSDT; do
    LOCAL_PATH="$REPO_ROOT/logs/${asset}_hmm_model.pkl"
    if [ ! -f "$LOCAL_PATH" ]; then
        gsutil cp \
            "gs://${GCP_BUCKET}/v2/vm_backup_20260420_1709/logs/logs/${asset}_hmm_model.pkl" \
            "$LOCAL_PATH" 2>/dev/null && \
            echo "  Downloaded ${asset}_hmm_model.pkl" || true
    fi
done

echo ""
echo "  Verifying environment..."

python3 - << 'PYEOF'
import torch
print(f"  PyTorch      : {torch.__version__}")
if torch.cuda.is_available():
    print(f"  CUDA         : {torch.version.cuda}")
    print(f"  GPU          : {torch.cuda.get_device_name(0)}")
else:
    print("  CUDA         : Not Available (CPU Mode)")
PYEOF

python3 -c "
import polars, xgboost, sklearn, numpy
print(f'  Polars       : {polars.__version__}')
print(f'  XGBoost      : {xgboost.__version__}')
print(f'  NumPy        : {numpy.__version__}')
"

# DONE
echo ""
echo ""
chmod +x "$REPO_ROOT"/*.sh "$REPO_ROOT"/scripts/*.sh 2>/dev/null || true
echo "  Pipeline scripts marked executable"

echo "========================================================"
echo "Setup complete. Environment ready."
echo "========================================================"
echo ""
if [ "$HAS_GPU" -eq 0 ]; then
    echo "  CPU DETECTED: You can run data prep and analysis scripts locally."
    echo "  Run: bash cpu_pipeline.sh"
    echo ""
    echo "  NOTE: You will need to switch to a GPU instance (e.g., RunPod)"
    echo "  to execute the 'gpu_pipeline.sh' training scripts."
else
    echo "  GPU DETECTED: You can run the full pipeline."
    echo "  1. bash cpu_pipeline.sh"
    echo "  2. bash gpu_pipeline.sh"
    echo "  3. bash cpu_post_gpu.sh"
fi
echo ""
