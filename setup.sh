#!/bin/bash
# MAIC Environment Setup (Fully Portable: RunPod & Local)
# Usage: bash setup.sh
#
# Requires in the repository root directory (gitignored):
#   - gcp-key.json    GCP service account key
#   - .env            All environment variables including GitHub token

set -e

# =============================================================================
# SIGINT / SIGTERM TRAP
# Ctrl+C now prints a clean message instead of silently dying mid-step.
# =============================================================================
trap 'echo ""; echo "  Setup interrupted (Ctrl+C). Re-run bash setup.sh to resume — it is safe to rerun."; exit 1' INT TERM

# =============================================================================
# DYNAMIC PATH RESOLUTION
# =============================================================================
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo ""
echo "========================================================"
echo "MAIC Environment Setup"
echo "Repository Root: $REPO_ROOT"
echo "========================================================"
echo ""

# =============================================================================
# [0/8] Hardware detection
# =============================================================================
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

# =============================================================================
# [1/8] Validate .env
# =============================================================================
echo ""
echo "[1/8] Validating .env..."

ENV_FILE="$REPO_ROOT/.env"

if [ ! -f "$ENV_FILE" ]; then
    echo "  ERROR: .env not found at $ENV_FILE"
    exit 1
fi

set -a; source "$ENV_FILE"; set +a

# Make GOOGLE_APPLICATION_CREDENTIALS absolute if relative
if [[ "$GOOGLE_APPLICATION_CREDENTIALS" != /* ]]; then
    export GOOGLE_APPLICATION_CREDENTIALS="$REPO_ROOT/$GOOGLE_APPLICATION_CREDENTIALS"
fi

REQUIRED_KEYS=(
    GCP_PROJECT_ID GCP_BUCKET GCP_REGION BQ_DATASET BQ_TABLE
    GOOGLE_APPLICATION_CREDENTIALS GITHUB_USER GITHUB_EMAIL
)

MISSING=0
for key in "${REQUIRED_KEYS[@]}"; do
    [ -z "${!key}" ] && echo "  ERROR: $key missing in .env" && MISSING=1
done
[ "$MISSING" -eq 1 ] && exit 1

[ ! -f "$GOOGLE_APPLICATION_CREDENTIALS" ] && \
    echo "  ERROR: gcp-key.json not found at $GOOGLE_APPLICATION_CREDENTIALS" && exit 1

echo "  .env loaded and validated"

# =============================================================================
# [2/8] System packages
# Removed > /dev/null suppression so failures are visible.
# =============================================================================
echo ""
echo "[2/8] Installing system packages..."

SUDO_CMD=""
if [ "$(id -u)" -ne 0 ] && command -v sudo &> /dev/null; then
    SUDO_CMD="sudo"
fi

$SUDO_CMD apt-get update -qq 2>/dev/null || true

DEBIAN_FRONTEND=noninteractive $SUDO_CMD apt-get install -y \
    apt-utils curl wget git build-essential ca-certificates gnupg \
    lsb-release unzip htop tmux net-tools vim nano \
    -o Dpkg::Use-Pty=0 2>&1 | grep -E "^(Err|W:|E:|Setting up)" || true

# Verify the packages we actually care about are present
for pkg in curl wget git nano vim tmux htop unzip; do
    command -v "$pkg" &>/dev/null && \
        echo "  ✓ $pkg" || echo "  ✗ $pkg NOT FOUND — retrying..."
done

# Retry any missing ones explicitly without quiet flags
for pkg in nano vim curl wget git tmux htop unzip; do
    if ! command -v "$pkg" &>/dev/null; then
        DEBIAN_FRONTEND=noninteractive $SUDO_CMD apt-get install -y "$pkg" || true
    fi
done

echo "  System packages step complete"

# =============================================================================
# [3/8] Python and pip
# =============================================================================
echo ""
echo "[3/8] Verifying Python and pip..."

command -v python3 &>/dev/null || \
    $SUDO_CMD apt-get install -y python3 python3-pip > /dev/null
echo "  Found: $(python3 --version)"
pip install --root-user-action=ignore --upgrade pip --break-system-packages --quiet
echo "  pip ready"

# =============================================================================
# [4/8] GCP Authentication
# =============================================================================
echo ""
echo "[4/8] Authenticating to GCP..."

if ! command -v gcloud &> /dev/null; then
    echo "  gcloud not in PATH. Checking default installation directory..."

    if [ -d "$HOME/google-cloud-sdk" ]; then
        echo "  Found SDK at $HOME/google-cloud-sdk — repairing PATH..."
        export PATH="$PATH:$HOME/google-cloud-sdk/bin"
    else
        echo "  SDK not found. Installing gcloud CLI..."
        curl -sSL https://sdk.cloud.google.com > /tmp/install_gcloud.sh
        CLOUDSDK_PYTHON=python3 bash /tmp/install_gcloud.sh \
            --disable-prompts --install-dir="$HOME" > /dev/null 2>&1 || {
            echo "  ERROR: gcloud installation failed."
            exit 1
        }
        export PATH="$PATH:$HOME/google-cloud-sdk/bin"
    fi
else
    echo "  gcloud already installed and active"
fi

# Persist PATH to shell configs (safe for re-runs)
if [ -f "$HOME/.bashrc" ] && ! grep -q "google-cloud-sdk" "$HOME/.bashrc"; then
    echo 'export PATH="$PATH:$HOME/google-cloud-sdk/bin"' >> "$HOME/.bashrc"
    echo "  Added SDK to .bashrc"
fi

if [ -d "$HOME/.config/fish" ] && \
   ! grep -q "google-cloud-sdk" "$HOME/.config/fish/config.fish" 2>/dev/null; then
    echo "source $HOME/google-cloud-sdk/path.fish.inc" \
        >> "$HOME/.config/fish/config.fish"
    echo "  Added SDK to Fish config"
fi

# Auth using whichever gcloud is reachable
GCLOUD_BIN="$(command -v gcloud 2>/dev/null || echo "$HOME/google-cloud-sdk/bin/gcloud")"
$GCLOUD_BIN auth activate-service-account \
    --key-file="$GOOGLE_APPLICATION_CREDENTIALS" --quiet 2>/dev/null || true
$GCLOUD_BIN config set project "$GCP_PROJECT_ID" --quiet 2>/dev/null || true

echo "  Authenticated: $($GCLOUD_BIN auth list --format='value(account)' 2>/dev/null | head -1)"

# =============================================================================
# [5/8] GitHub Authentication
# =============================================================================
echo ""
echo "[5/8] Configuring GitHub..."

git config --global user.email "$GITHUB_EMAIL"
git config --global user.name  "$GITHUB_USER"
git config --global init.defaultBranch main
git config --global pull.rebase false

# Uses SSH, not an HTTPS token -- run `ssh -T git@github.com` to confirm
# your key is registered before this script clones/pulls anything.
echo "  GitHub auth: using SSH key (git@github.com) for $GITHUB_USER"

# =============================================================================
# [6/8] Clone or update MAIC repo
# =============================================================================
echo ""
echo "[6/8] Syncing MAIC repo..."

if [ ! -d "$REPO_ROOT/.git" ]; then
    echo "  Directory is not a git repo — cloning..."
    git clone "git@github.com:${GITHUB_USER}/maic.git" /tmp/maic_clone
    cp -r /tmp/maic_clone/. "$REPO_ROOT/"
    rm -rf /tmp/maic_clone
    echo "  Repo synced"
else
    echo "  Pulling latest..."
    cd "$REPO_ROOT" && git pull --quiet
    echo "  Up to date"
fi

cd "$REPO_ROOT"

# =============================================================================
# [7/8] Python packages
# =============================================================================
echo ""
echo "[7/8] Installing Python packages..."

[ ! -f "$REPO_ROOT/requirements.txt" ] && \
    echo "  ERROR: requirements.txt not found" && exit 1

# ---------------------------------------------------------------------------
# [7a] PyTorch — skip reinstall if correct version already present
# ---------------------------------------------------------------------------
echo "  [7a] Checking PyTorch..."

if [ "$HAS_GPU" -eq 1 ]; then
    echo "  Upgrading nvidia-nvjitlink-cu12..."
    pip install --root-user-action=ignore "nvidia-nvjitlink-cu12>=12.9" \
        --break-system-packages --quiet

    SM_MAJOR=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
        | head -1 | cut -d'.' -f1 | tr -d ' ')
    SM_MINOR=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
        | head -1 | cut -d'.' -f2 | tr -d ' ')
    SM="${SM_MAJOR}${SM_MINOR}"
    echo "  Detected compute capability: sm_${SM}"

    if [ "$SM" = "120" ]; then
        PYTORCH_INDEX="https://download.pytorch.org/whl/nightly/cu128"
        PYTORCH_LABEL="nightly cu128 (Blackwell sm_120)"
        PYTORCH_CHECK="'dev' in torch.__version__ and (torch.version.cuda or '').startswith('12.8')"
    else
        PYTORCH_INDEX="https://download.pytorch.org/whl/cu121"
        PYTORCH_LABEL="stable cu121"
        PYTORCH_CHECK="torch.cuda.is_available() and (torch.version.cuda or '').startswith('12.1')"
    fi
else
    PYTORCH_INDEX="https://download.pytorch.org/whl/cpu"
    PYTORCH_LABEL="CPU only"
    PYTORCH_CHECK="not torch.cuda.is_available()"
fi

echo "  Selected: PyTorch $PYTORCH_LABEL"

# Only install if not already correct — avoids 3-4GB re-download on reruns
PYTORCH_OK=0
python3 -c "
import torch, sys
try:
    ok = $PYTORCH_CHECK
    sys.exit(0 if ok else 1)
except Exception:
    sys.exit(1)
" 2>/dev/null && PYTORCH_OK=1 || PYTORCH_OK=0

if [ "$PYTORCH_OK" -eq 1 ]; then
    echo "  PyTorch $PYTORCH_LABEL already installed — skipping download"
else
    echo "  Installing PyTorch $PYTORCH_LABEL (this may take a few minutes)..."
    pip install --root-user-action=ignore torch torchvision torchaudio \
        --index-url "$PYTORCH_INDEX" \
        --break-system-packages --quiet
    echo "  PyTorch installed"
fi

# Verify
python3 - << 'PYEOF'
import torch
print(f"  PyTorch      : {torch.__version__}")
print(f"  CUDA         : {torch.version.cuda or 'N/A'}")
if torch.cuda.is_available():
    print(f"  GPU          : {torch.cuda.get_device_name(0)}")
    vram = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    print(f"  VRAM         : {vram} GB")
    cap = torch.cuda.get_device_capability()
    print(f"  Capability   : sm_{cap[0]}{cap[1]}")
else:
    print("  CUDA         : Not available (CPU mode)")
PYEOF

# ---------------------------------------------------------------------------
# [7b] Core packages from requirements.txt
# ---------------------------------------------------------------------------
echo "  [7b] Installing core packages..."
pip install --root-user-action=ignore -r "$REPO_ROOT/requirements.txt" \
    --break-system-packages --quiet
echo "  Core packages ready"

# ---------------------------------------------------------------------------
# [7c] cuML — GPU only, skip if already installed
# ---------------------------------------------------------------------------
if [ "$HAS_GPU" -eq 1 ]; then
    echo "  [7c] Checking cuML..."
    python3 -c "from cuml.ensemble import RandomForestClassifier" 2>/dev/null && \
        echo "  cuML already installed — skipping" || {
        echo "  Installing cuML..."
        pip install --root-user-action=ignore cuml-cu12 \
            --extra-index-url=https://pypi.nvidia.com \
            --break-system-packages --quiet
        echo "  cuML installed"
    }
else
    echo "  [7c] Skipping cuML (no GPU)"
fi

# ---------------------------------------------------------------------------
# [7d] tqdm — skip if already installed
# ---------------------------------------------------------------------------
echo "  [7d] Checking tqdm..."
python3 -c "import tqdm" 2>/dev/null && \
    echo "  tqdm already installed" || {
    pip install --root-user-action=ignore tqdm --break-system-packages --quiet
    echo "  tqdm installed"
}

echo "  All packages ready"

# =============================================================================
# [8/8] HMM models + environment verification
# =============================================================================
echo ""
echo "[8/8] Finalising environment..."

mkdir -p "$REPO_ROOT/logs"

# Download HMM model pickles from GCS backup if not already present locally
for asset in BTCUSDT ETHUSDT SOLUSDT; do
    LOCAL_PATH="$REPO_ROOT/logs/${asset}_hmm_model.pkl"
    if [ ! -f "$LOCAL_PATH" ]; then
        gsutil cp \
            "gs://${GCP_BUCKET}/v2/vm_backup_20260420_1709/logs/logs/${asset}_hmm_model.pkl" \
            "$LOCAL_PATH" 2>/dev/null && \
            echo "  Downloaded ${asset}_hmm_model.pkl" || \
            echo "  WARNING: Could not download ${asset}_hmm_model.pkl"
    else
        echo "  ${asset}_hmm_model.pkl already present — skipping download"
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
    vram = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    print(f"  VRAM         : {vram} GB")
else:
    print("  CUDA         : Not Available (CPU Mode)")
PYEOF

if [ "$HAS_GPU" -eq 1 ]; then
    python3 -c "
from cuml.ensemble import RandomForestClassifier
print('  cuML GPU RF  : OK')
" 2>/dev/null || echo "  cuML         : NOT AVAILABLE — check install"
fi

python3 -c "
import polars, xgboost, sklearn, numpy, tqdm, hmmlearn
print(f'  Polars       : {polars.__version__}')
print(f'  XGBoost      : {xgboost.__version__}')
print(f'  sklearn      : {sklearn.__version__}')
print(f'  NumPy        : {numpy.__version__}')
print(f'  tqdm         : {tqdm.__version__}')
print(f'  hmmlearn     : {hmmlearn.__version__}')
print('  All core packages OK')
"

# =============================================================================
# DONE
# =============================================================================
echo ""
chmod +x "$REPO_ROOT"/*.sh "$REPO_ROOT"/scripts/*.sh 2>/dev/null || true
echo "  Pipeline scripts marked executable"

echo ""
echo "========================================================"
echo "Setup complete. Environment ready."
echo "========================================================"
echo ""

if [ "$HAS_GPU" -eq 0 ]; then
    echo "  CPU mode — data prep and analysis scripts available."
    echo "  Run: bash cpu_pipeline.sh"
    echo ""
    echo "  NOTE: GPU instance required for training scripts (gpu_pipeline.sh)."
else
    echo "  Full pipeline available."
    echo "  1. bash cpu_pipeline.sh"
    echo "  2. bash gpu_pipeline.sh"
    echo "  3. bash cpu_post_gpu.sh"
fi
echo ""