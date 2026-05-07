#!/bin/bash
# =============================================================================
# MAIC — CPU Pipeline (Pre-GPU)
# Runs all data preparation stages before GPU training.
# Must complete before running gpu_pipeline.sh.
#
# Usage:
#   bash cpu_pipeline.sh             # run all stages
#   bash cpu_pipeline.sh --from=4    # resume from stage 4
#   bash cpu_pipeline.sh --dry-run   # print plan, execute nothing
#   bash cpu_pipeline.sh --check     # validate env/paths/imports only
#
# Full pipeline order:
#   1. bash cpu_pipeline.sh          (this script)
#   2. bash gpu_pipeline.sh
#   3. bash cpu_post_pipeline.sh
#
# Replicator requirements:
#   1. A GCP project with a Cloud Storage bucket containing the MAIC data
#   2. A service account key saved as gcp-key.json in the repo root
#   3. A .env file with: GCP_PROJECT_ID, GCP_BUCKET, GCP_REGION
#   4. Python 3.11+ with requirements installed (run setup.sh first)
#   The bucket name is the only thing that differs between environments.
# =============================================================================

set -euo pipefail

cd "$(dirname "$0")"

# -----------------------------------------------------------------------------
# ARGS
# -----------------------------------------------------------------------------
FROM_STAGE=0
DRY_RUN=0
CHECK=0

for arg in "$@"; do
    case $arg in
        --from=*)   FROM_STAGE="${arg#*=}" ;;
        --dry-run)  DRY_RUN=1 ;;
        --check)    CHECK=1 ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: bash cpu_pipeline.sh [--from=N] [--dry-run] [--check]"
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
# STAGE DEFINITIONS — pre-GPU only
# -----------------------------------------------------------------------------
declare -a STAGE_NUMS=(1 2 3 4 5 6 7)
declare -a STAGE_NAMES=(
    "01_download"
    "02_csv_to_parquet"
    "03_quality_audit"
    "04a_feature_engineering"
    "04b_stationarity"
    "05a_label_generation"
    "05b_verify_features"
)
declare -a STAGE_SCRIPTS=(
    "scripts/01_download.py"
    "scripts/02_csv_to_parquet.py"
    "scripts/03_quality_audit.py"
    "scripts/04a_feature_engineering.py"
    "scripts/04b_stationarity_fracdiff.py"
    "scripts/05a_label_generation.py"
    "scripts/05b_verify_features.py"
)

NUM_STAGES=${#STAGE_NUMS[@]}

# -----------------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  MAIC CPU Pipeline (Pre-GPU)"
echo "  $(date)"
if   [ "$CHECK"   -eq 1 ]; then echo "  Mode: CHECK (no computation)"
elif [ "$DRY_RUN" -eq 1 ]; then echo "  Mode: DRY RUN (no execution)"
elif [ "$FROM_STAGE" -gt 0 ]; then echo "  Mode: RESUME from stage $FROM_STAGE"
else echo "  Mode: FULL RUN"
fi
echo "============================================================"
echo ""

# -----------------------------------------------------------------------------
# PREFLIGHT
# -----------------------------------------------------------------------------
log "Preflight checks..."

[ ! -f ".env" ]         && fail ".env not found — copy .env.example and fill in your values"
[ ! -f "gcp-key.json" ] && fail "gcp-key.json not found — add your GCP service account key"
[ ! -f "config.py" ]    && fail "config.py not found — are you in the repo root?"

set -a; source .env; set +a
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/gcp-key.json"

[ -z "${GCP_PROJECT_ID:-}" ] && fail "GCP_PROJECT_ID not set in .env"
[ -z "${GCP_BUCKET:-}"     ] && fail "GCP_BUCKET not set in .env"

ok "Preflight — .env and credentials found"
info "Bucket : $GCP_BUCKET"
info "Project: $GCP_PROJECT_ID"

# -----------------------------------------------------------------------------
# CHECK MODE
# -----------------------------------------------------------------------------
if [ "$CHECK" -eq 1 ]; then
    echo ""
    log "CHECK MODE — validating environment without running any computation"
    echo ""
    FAIL=0

    log "1/4  Checking script files exist..."
    for i in $(seq 0 $((NUM_STAGES - 1))); do
        script="${STAGE_SCRIPTS[$i]}"
        [ -f "$script" ] && ok "  Found: $script" || { warn "  MISSING: $script"; FAIL=1; }
    done

    echo ""
    log "2/4  Checking Python imports..."
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
    failed = [m for m in third_party if not __import__(m) is not None or False]
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
    log "3/4  Checking GCS bucket access..."
    python3 -c "
import os; os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '$(pwd)/gcp-key.json'
from google.cloud import storage
client = storage.Client()
print('EXISTS' if client.bucket('$GCP_BUCKET').exists() else 'NOT_FOUND')
" 2>/dev/null | grep -q "EXISTS" \
        && ok "  GCS bucket '$GCP_BUCKET' is accessible" \
        || { warn "  GCS bucket '$GCP_BUCKET' not accessible"; FAIL=1; }

    echo ""
    log "4/4  Checking config.py..."
    python3 -c "
import sys; sys.path.insert(0, '.')
from config import BUCKET, ASSETS, WINDOWS, ASSET_D_VALUES
assert len(ASSETS) == 3 and len(WINDOWS) == 5
print('OK')
" 2>/dev/null && ok "  config.py — OK" || { warn "  config.py import failed"; FAIL=1; }

    echo ""
    echo "============================================================"
    if [ "$FAIL" -eq 0 ]; then
        ok "ALL CHECKS PASSED — safe to run pre-GPU pipeline"
        echo ""
        info "Next: bash cpu_pipeline.sh"
        info "Then: bash gpu_pipeline.sh"
        info "Then: bash cpu_post_pipeline.sh"
    else
        warn "SOME CHECKS FAILED — fix the above before running"
    fi
    echo "============================================================"
    echo ""
    exit $FAIL
fi

# -----------------------------------------------------------------------------
# DRY RUN
# -----------------------------------------------------------------------------
if [ "$DRY_RUN" -eq 1 ]; then
    echo ""
    log "DRY RUN — execution plan (nothing will run)"
    echo ""
    for i in $(seq 0 $((NUM_STAGES - 1))); do
        num="${STAGE_NUMS[$i]}"; name="${STAGE_NAMES[$i]}"; script="${STAGE_SCRIPTS[$i]}"
        done_marker="logs/.done_${name}"
        if [ "$num" -lt "$FROM_STAGE" ]; then
            warn "  Stage $num ($name) — SKIP (before --from=$FROM_STAGE)"
        elif [ -f "$done_marker" ]; then
            ok "  Stage $num ($name) — SKIP (already done)"
        else
            log "  Stage $num ($name) — WOULD RUN: python -u $script"
        fi
    done
    echo ""
    exit 0
fi

# -----------------------------------------------------------------------------
# RUN STAGES
# -----------------------------------------------------------------------------
run_stage() {
    local stage_num="$1" stage_name="$2" script="$3"
    local log_file="logs/${stage_name}.log"
    local done_marker="logs/.done_${stage_name}"

    [ "$stage_num" -lt "$FROM_STAGE" ] && { warn "Stage $stage_num ($stage_name) — skipped"; return; }
    [ -f "$done_marker" ] && { ok "Stage $stage_num ($stage_name) — already done, skipping"; return; }
    [ ! -f "$script" ] && fail "Script not found: $script"

    log "Stage $stage_num ($stage_name) — starting"
    info "Log: $log_file"

    if python3 -u "$script" 2>&1 | tee "$log_file"; then
        touch "$done_marker"
        ok "Stage $stage_num ($stage_name) — complete"
    else
        fail "Stage $stage_num ($stage_name) — FAILED. See $log_file"
    fi
}

for i in $(seq 0 $((NUM_STAGES - 1))); do
    run_stage "${STAGE_NUMS[$i]}" "${STAGE_NAMES[$i]}" "${STAGE_SCRIPTS[$i]}"
done

# -----------------------------------------------------------------------------
# DONE
# -----------------------------------------------------------------------------
echo ""
echo "============================================================"
ok "Pre-GPU CPU pipeline complete"
echo ""
info "Features and labels are ready for training."
info "Next step: bash gpu_pipeline.sh"
echo "============================================================"
echo ""
