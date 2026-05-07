#!/bin/bash
# =============================================================================
# MAIC — GPU Pipeline
# Runs all GPU-intensive training stages.
# Must run after cpu_pipeline.sh and before cpu_post_pipeline.sh.
#
# Usage:
#   bash gpu_pipeline.sh               # run all GPU stages
#   bash gpu_pipeline.sh --from=06d    # resume from a stage
#   bash gpu_pipeline.sh --only=06d    # run one stage only
#   bash gpu_pipeline.sh --dry-run     # print plan, execute nothing
#   bash gpu_pipeline.sh --check       # validate env/GPU/imports only
#
# Full pipeline order:
#   1. bash cpu_pipeline.sh
#   2. bash gpu_pipeline.sh            (this script)
#   3. bash cpu_post_pipeline.sh
#
# Recommended for a full unattended run:
#   nohup bash gpu_pipeline.sh >> logs/gpu_pipeline.log 2>&1 &
#
# Replicator requirements:
#   1. CUDA-capable GPU (Blackwell sm_120 → nightly cu128 auto-selected;
#      all others → stable cu121 auto-selected by setup.sh)
#   2. Minimum 20GB VRAM for fold 4 production run (34GB recommended)
#   3. GCP service account key as gcp-key.json
#   4. .env with GCP_PROJECT_ID, GCP_BUCKET, GCP_REGION
#   5. Run setup.sh before this script
# =============================================================================

set -euo pipefail

cd "$(dirname "$0")"

# -----------------------------------------------------------------------------
# ARGS
# -----------------------------------------------------------------------------
FROM_STAGE=""
ONLY_STAGE=""
DRY_RUN=0
CHECK=0
SKIP=1

for arg in "$@"; do
    case $arg in
        --from=*)   FROM_STAGE="${arg#*=}" ;;
        --only=*)   ONLY_STAGE="${arg#*=}" ;;
        --dry-run)  DRY_RUN=1 ;;
        --check)    CHECK=1 ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: bash gpu_pipeline.sh [--from=STAGE] [--only=STAGE] [--dry-run] [--check]"
            exit 1
            ;;
    esac
done

# -----------------------------------------------------------------------------
# COLOURS
# -----------------------------------------------------------------------------
mkdir -p logs

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"; }
ok()   { echo -e "${GREEN}[$(date '+%H:%M:%S')] ✓ $1${NC}"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] ⚠ $1${NC}"; }
fail() { echo -e "${RED}[$(date '+%H:%M:%S')] ✗ $1${NC}"; exit 1; }
info() { echo -e "${CYAN}                    $1${NC}"; }

# -----------------------------------------------------------------------------
# STAGE DEFINITIONS
# GPU stages — use CUDA, cuML, or GPU-accelerated XGBoost/PyTorch
# -----------------------------------------------------------------------------
STAGE_IDS=("06a" "06b" "06c" "06d")
STAGE_NAMES=(
    "Baseline training"
    "Extended training"
    "Ablation study (fractional differencing)"
    "Production run (5 seeds x 4 folds)"
)
STAGE_SCRIPTS=(
    "scripts/06a_train_models.py"
    "scripts/06b_train_models.py"
    "scripts/06c_train_ablation.py"
    "scripts/06d_train_production.py"
)
STAGE_TIMES=(
    "~1-2 hours"
    "~2-4 hours"
    "~2-3 hours"
    "~6-10 hours"
)

NUM_STAGES=${#STAGE_IDS[@]}

# -----------------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  MAIC GPU Pipeline"
echo "  $(date)"
if   [ "$CHECK"   -eq 1 ]; then echo "  Mode: CHECK (no computation)"
elif [ "$DRY_RUN" -eq 1 ]; then echo "  Mode: DRY RUN (no execution)"
elif [ -n "$ONLY_STAGE" ];  then echo "  Mode: ONLY stage $ONLY_STAGE"
elif [ -n "$FROM_STAGE" ];  then echo "  Mode: RESUME from $FROM_STAGE"
else echo "  Mode: FULL RUN"
fi
echo "============================================================"
echo ""

# -----------------------------------------------------------------------------
# PREFLIGHT
# -----------------------------------------------------------------------------
log "Preflight checks..."

command -v nvidia-smi &>/dev/null || fail "No GPU detected — nvidia-smi not found"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
ok "GPU: $GPU_NAME ($GPU_MEM)"

[ ! -f ".env" ]         && fail ".env not found"
[ ! -f "gcp-key.json" ] && fail "gcp-key.json not found"

set -a; source .env; set +a
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/gcp-key.json"

[ -z "${GCP_PROJECT_ID:-}" ] && fail "GCP_PROJECT_ID not set in .env"
[ -z "${GCP_BUCKET:-}"     ] && fail "GCP_BUCKET not set in .env"

# Guard — refuse to run if pre-GPU CPU pipeline is not done
if [ ! -f "logs/.done_05b_verify_features" ] && [ "$CHECK" -eq 0 ] && [ "$DRY_RUN" -eq 0 ]; then
    fail "Pre-GPU CPU pipeline not complete — run cpu_pipeline.sh first (logs/.done_05b_verify_features not found)"
fi

ok "Preflight — credentials and env OK"

# -----------------------------------------------------------------------------
# CHECK MODE
# -----------------------------------------------------------------------------
if [ "$CHECK" -eq 1 ]; then
    echo ""
    log "CHECK MODE — validating GPU environment without any training"
    echo ""
    FAIL=0

    log "1/6  Checking script files exist..."
    for i in $(seq 0 $((NUM_STAGES - 1))); do
        script="${STAGE_SCRIPTS[$i]}"
        [ -f "$script" ] && ok "  Found: $script" || { warn "  MISSING: $script"; FAIL=1; }
    done

    echo ""
    log "2/6  Checking PyTorch and CUDA..."
    python3 - << 'PYEOF'
import torch, sys
if not torch.cuda.is_available():
    print("ERROR: CUDA not available")
    sys.exit(1)
cap = torch.cuda.get_device_capability()
print(f"  PyTorch  : {torch.__version__}")
print(f"  CUDA     : {torch.version.cuda}")
print(f"  Device   : {torch.cuda.get_device_name(0)}")
print(f"  Compute  : sm_{cap[0]}{cap[1]}")
PYEOF
    [ $? -eq 0 ] && ok "  PyTorch + CUDA OK" || { warn "  PyTorch/CUDA check failed"; FAIL=1; }

    echo ""
    log "3/6  Checking cuML..."
    python3 -c "
import cuml
from cuml.ensemble import RandomForestClassifier
print(f'  cuML version: {cuml.__version__}')
" 2>/dev/null && ok "  cuML OK" || { warn "  cuML not available — RF GPU training will fail"; FAIL=1; }

    echo ""
    log "4/6  Checking XGBoost GPU support..."
    python3 -c "
import xgboost as xgb, numpy as np
X = np.random.rand(100, 5).astype(np.float32)
y = (X[:, 0] > 0.5).astype(np.int32)
xgb.XGBClassifier(n_estimators=2, tree_method='hist', device='cuda', verbosity=0).fit(X, y)
print('  XGBoost GPU OK')
" 2>/dev/null && ok "  XGBoost GPU OK" || { warn "  XGBoost GPU not available"; FAIL=1; }

    echo ""
    log "5/6  Checking GCS bucket..."
    python3 -c "
import os; os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '$(pwd)/gcp-key.json'
from google.cloud import storage
print('EXISTS' if storage.Client().bucket('$GCP_BUCKET').exists() else 'NOT_FOUND')
" 2>/dev/null | grep -q "EXISTS" \
        && ok "  GCS bucket '$GCP_BUCKET' accessible" \
        || { warn "  GCS bucket '$GCP_BUCKET' not accessible"; FAIL=1; }

    echo ""
    log "6/6  Checking GPU script imports..."
    for i in $(seq 0 $((NUM_STAGES - 1))); do
        script="${STAGE_SCRIPTS[$i]}"; [ ! -f "$script" ] && continue
        name="${STAGE_NAMES[$i]}"
        result=$(python3 -c "
import ast, sys
try:
    with open('$script') as f:
        tree = ast.parse(f.read())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names: imports.append(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module: imports.append(node.module.split('.')[0])
    stdlib = {'os','sys','io','re','gc','json','csv','time','math','datetime',
              'pathlib','logging','warnings','typing','collections','itertools',
              'functools','dataclasses','abc','copy','enum','random','string',
              'struct','traceback','contextlib','threading','multiprocessing'}
    third_party = [m for m in set(imports) if m not in stdlib and not m.startswith('_')]
    actually_failed = []
    for m in third_party:
        try: __import__(m)
        except ImportError: actually_failed.append(m)
    if actually_failed:
        print('MISSING: ' + ', '.join(actually_failed)); sys.exit(1)
    print('OK')
except Exception as e:
    print(f'PARSE ERROR: {e}'); sys.exit(1)
" 2>&1)
        echo "$result" | grep -q "^OK" && ok "  $name — imports OK" || { warn "  $name — $result"; FAIL=1; }
    done

    echo ""
    echo "============================================================"
    if [ "$FAIL" -eq 0 ]; then
        ok "ALL GPU CHECKS PASSED — safe to run training"
        echo ""
        info "Run: nohup bash gpu_pipeline.sh >> logs/gpu_pipeline.log 2>&1 &"
    else
        warn "SOME CHECKS FAILED — fix before training"
    fi
    echo "============================================================"
    echo ""
    exit $FAIL
fi

# -----------------------------------------------------------------------------
# DRY RUN
# -----------------------------------------------------------------------------
should_run() {
    local id="$1"
    [ -n "$ONLY_STAGE" ] && { [ "$id" = "$ONLY_STAGE" ] && return 0 || return 1; }
    if [ -n "$FROM_STAGE" ] && [ "$id" = "$FROM_STAGE" ]; then SKIP=0; fi
    [ -n "$FROM_STAGE" ] && [ "$SKIP" -eq 1 ] && return 1
    return 0
}

if [ "$DRY_RUN" -eq 1 ]; then
    echo ""
    log "DRY RUN — execution plan"
    echo ""
    for i in $(seq 0 $((NUM_STAGES - 1))); do
        id="${STAGE_IDS[$i]}"; name="${STAGE_NAMES[$i]}"
        script="${STAGE_SCRIPTS[$i]}"; est="${STAGE_TIMES[$i]}"
        done_marker="logs/.done_gpu_${id}"
        if ! should_run "$id"; then
            warn "  $id ($name) — SKIP"
        elif [ -f "$done_marker" ]; then
            ok "  $id ($name) — SKIP (already done)"
        else
            log "  $id ($name) — WOULD RUN: python -u $script  [$est]"
        fi
    done
    echo ""
    exit 0
fi

# -----------------------------------------------------------------------------
# RUN STAGES
# -----------------------------------------------------------------------------
run_stage() {
    local id="$1" name="$2" script="$3" est="$4"
    local log_file="logs/${id}_$(basename $script .py).log"
    local done_marker="logs/.done_gpu_${id}"

    if ! should_run "$id"; then
        warn "Stage $id ($name) — skipped"
        return
    fi

    [ -f "$done_marker" ] && { ok "Stage $id ($name) — already done, skipping"; return; }
    [ ! -f "$script" ] && fail "Script not found: $script"

    echo ""
    log "Stage $id ($name)"
    info "Script    : $script"
    info "Log       : $log_file"
    info "Estimated : $est"

    echo "" >> "$log_file"
    echo "=== GPU STATE AT START ===" >> "$log_file"
    nvidia-smi --query-gpu=name,memory.used,memory.free,temperature.gpu \
        --format=csv,noheader >> "$log_file" 2>/dev/null || true
    echo "==========================" >> "$log_file"

    if python3 -u "$script" 2>&1 | tee -a "$log_file"; then
        touch "$done_marker"
        ok "Stage $id ($name) — complete"
    else
        fail "Stage $id ($name) — FAILED. See $log_file"
    fi
}

for i in $(seq 0 $((NUM_STAGES - 1))); do
    run_stage "${STAGE_IDS[$i]}" "${STAGE_NAMES[$i]}" "${STAGE_SCRIPTS[$i]}" "${STAGE_TIMES[$i]}"
done

echo ""
echo "============================================================"
ok "GPU pipeline complete — training done"
echo ""
info "Next: bash cpu_post_pipeline.sh"
echo "============================================================"
echo ""
