#!/bin/bash
# =============================================================================
# MAIC — CPU Pipeline (Post-GPU)
# Runs aggregation, figure generation, lead-time analysis, HMM robustness,
# external crisis validation, and optionally the live inference system.
#
# Usage:
#   bash cpu_post_gpu.sh             # run all stages
#   bash cpu_post_gpu.sh --from=4    # resume from stage 4
#   bash cpu_post_gpu.sh --dry-run   # print plan, execute nothing
#   bash cpu_post_gpu.sh --check     # validate env/paths/imports only
#
# Full pipeline order:
#   1. bash cpu_pipeline.sh
#   2. bash gpu_pipeline.sh
#   3. bash cpu_post_gpu.sh          (this script)
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
            echo "Usage: bash cpu_post_gpu.sh [--from=N] [--dry-run] [--check]"
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
# -----------------------------------------------------------------------------
declare -a STAGE_NUMS=(1 2 3 4 5 6 7 8 9 10 11)
declare -a STAGE_NAMES=(
    "07a_aggregate_baseline"
    "07b_aggregate_ablation"
    "07c_aggregate_production"
    "08_generate_paper_figures"
    "09_lead_time_analysis"
    "10_hmm_robustness_check"
    "11a_hmm_stability_local"
    "11b_crisis_validation_full"
    "13a_persistence_baseline"
    "13b_lead_time_external"
    "13c_block_bootstrap_ztest"
)
declare -a STAGE_SCRIPTS=(
    "scripts/07a_aggregate_results.py"
    "scripts/07b_aggregate_ablation.py"
    "scripts/07c_aggregate_production.py"
    "scripts/08_generate_paper_figures.py"
    "scripts/09_lead_time_analysis.py"
    "scripts/10_hmm_robustness_check.py"
    "scripts/11a_local_global_hmm.py"
    "scripts/11b_crisis_validation_full.py"
    "scripts/13a_persistence_baseline.py"
    "scripts/13b_lead_time_external.py"
    "scripts/13c_block_bootstrap_ztest.py"
)

NUM_STAGES=${#STAGE_NUMS[@]}

# -----------------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  MAIC CPU Pipeline (Post-GPU)"
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

[ ! -f ".env" ]         && fail ".env not found"
[ ! -f "gcp-key.json" ] && fail "gcp-key.json not found"
[ ! -f "config.py" ]    && fail "config.py not found — are you in the repo root?"

set -a; source .env; set +a
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/gcp-key.json"

[ -z "${GCP_PROJECT_ID:-}" ] && fail "GCP_PROJECT_ID not set in .env"
[ -z "${GCP_BUCKET:-}"     ] && fail "GCP_BUCKET not set in .env"

# Guard — refuse to run if GPU training is not done
if [ ! -f "logs/.done_gpu_06d" ] && [ "$CHECK" -eq 0 ] && [ "$DRY_RUN" -eq 0 ]; then
    fail "GPU training not complete — run gpu_pipeline.sh first (logs/.done_gpu_06d not found)"
fi

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
print('EXISTS' if storage.Client().bucket('$GCP_BUCKET').exists() else 'NOT_FOUND')
" 2>/dev/null | grep -q "EXISTS" \
        && ok "  GCS bucket '$GCP_BUCKET' is accessible" \
        || { warn "  GCS bucket '$GCP_BUCKET' not accessible"; FAIL=1; }

    echo ""
    log "4/4  Checking GPU training prerequisite..."
    if [ -f "logs/.done_gpu_06d" ]; then
        ok "  GPU training complete (logs/.done_gpu_06d found)"
    else
        warn "  GPU training not yet complete — run gpu_pipeline.sh first"
        FAIL=1
    fi

    echo ""
    echo "============================================================"
    if [ "$FAIL" -eq 0 ]; then
        ok "ALL CHECKS PASSED — safe to run post-GPU pipeline"
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
    log "  Stage 12 (12_inference) — WOULD PROMPT: run live inference? [y/N]"
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
    [ ! -f "$script" ] && { warn "Stage $stage_num ($stage_name) — script not found, skipping"; return; }

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
# STAGE 10 — LIVE INFERENCE (optional, prompted)
# -----------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  OPTIONAL: Live Inference System"
echo "============================================================"
echo ""
echo "  The early warning system can now run against live Binance"
echo "  data using the production XGBoost model."
echo ""
echo "  Note: Stress = liquidity conditions, NOT a price prediction."
echo "  Price impact is not guaranteed — liquid markets may absorb"
echo "  stress without significant price movement."
echo ""
echo -n "  Run live inference monitoring? [y/N]: "
read -r run_inference

if [[ "$run_inference" =~ ^[Yy]$ ]]; then
    echo ""
    echo -n "  Monitor which asset? [BTCUSDT/ETHUSDT/SOLUSDT/all] (default: all): "
    read -r asset_choice
    asset_choice=${asset_choice:-all}

    echo ""
    echo -n "  Run continuously (--loop) or single snapshot? [loop/single] (default: loop): "
    read -r mode_choice
    mode_choice=${mode_choice:-loop}

    echo ""
    log "Starting inference: asset=$asset_choice mode=$mode_choice"
    info "Logs → logs/inference/inference_log.csv"
    info "Outcomes → logs/inference/outcome_log.csv (resolved 30 min after each WARNING)"
    info "Press Ctrl+C to stop continuous monitoring"
    echo ""

    if [[ "$mode_choice" == "loop" ]]; then
        python3 scripts/12_inference.py --asset "$asset_choice" --loop
    else
        python3 scripts/12_inference.py --asset "$asset_choice"
    fi
else
    echo ""
    info "Inference skipped. Run manually at any time:"
    info "  python3 scripts/12_inference.py --asset all --loop"
    info "  python3 scripts/12_inference.py --asset BTCUSDT"
fi

# -----------------------------------------------------------------------------
# DONE
# -----------------------------------------------------------------------------
echo ""
echo "============================================================"
ok "Post-GPU CPU pipeline complete"
echo ""
info "Results      : production_results.csv"
info "Lead times   : lead_time_results.csv"
info "Robustness   : hmm_robustness_*.csv"
info "Stability    : global_hmm_stability.csv"
info "Validation   : crisis_validation_*.csv + global_vs_local_kappa.csv"
info "Figures      : paper_figures/"
info "Inference    : logs/inference/ (if run)"
echo "============================================================"
echo ""