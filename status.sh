#!/bin/bash
# =============================================================================
# MAIC — Pipeline Status
# Shows which stages are complete, running, or pending.
# Usage: bash status.sh
# =============================================================================

cd "$(dirname "$0")"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREY='\033[0;37m'
BLUE='\033[0;34m'
NC='\033[0m'

stage_status() {
    local label="$1"
    local done_marker="$2"
    local log_file="$3"

    printf "  %-48s" "$label"

    if [ -f "$done_marker" ]; then
        echo -e "${GREEN}✓ done${NC}"
        return
    fi

    if [ -f "$log_file" ] && [ "$(find "$log_file" -mmin -1 2>/dev/null)" ]; then
        echo -e "${YELLOW}⟳ running${NC}"
        return
    fi

    if [ -f "$log_file" ]; then
        echo -e "${RED}✗ failed${NC}"
        return
    fi

    echo -e "${GREY}- pending${NC}"
}

check_file() {
    local label="$1"
    local path="$2"
    printf "  %-48s" "$label"
    if [ -f "$path" ]; then
        SIZE=$(du -sh "$path" 2>/dev/null | cut -f1)
        echo -e "${GREEN}✓ present (${SIZE})${NC}"
    else
        echo -e "${GREY}- missing${NC}"
    fi
}

echo ""
echo "========================================================"
echo "  MAIC Pipeline Status — $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"

if command -v nvidia-smi &>/dev/null; then
    GPU=$(nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total \
        --format=csv,noheader 2>/dev/null | head -1)
    echo ""
    echo -e "  ${BLUE}GPU:${NC} $GPU"
fi

echo ""
echo -e "  ${BLUE}── CPU STAGES ────────────────────────────────────────${NC}"
stage_status "01 · Download raw trades"            "logs/.done_01_download"               "logs/01_download.log"
stage_status "02 · CSV → Parquet"                  "logs/.done_02_csv_to_parquet"         "logs/02_csv_to_parquet.log"
stage_status "03 · Quality audit"                  "logs/.done_03_quality_audit"          "logs/03_quality_audit.log"
stage_status "04a · Feature engineering"          "logs/.done_04a_feature_engineering"   "logs/04a_feature_engineering.log"
stage_status "04b · Stationarity / fracdiff"      "logs/.done_04b_stationarity"          "logs/04b_stationarity.log"
stage_status "05a · HMM label generation"         "logs/.done_05a_label_generation"      "logs/05a_label_generation.log"
stage_status "05b · Feature verification"         "logs/.done_05b_verify_features"       "logs/05b_verify_features.log"

echo ""
echo -e "  ${BLUE}── GPU STAGES (training only) ────────────────────────${NC}"
stage_status "06a · Baseline training"            "logs/.done_gpu_06a"  "logs/06a_train_models.log"
stage_status "06b · Extended training"            "logs/.done_gpu_06b"  "logs/06b_train_models.log"
stage_status "06c · Ablation study"               "logs/.done_gpu_06c"  "logs/06c_train_ablation.log"
stage_status "06d · Production run (5s × 4f)"     "logs/.done_gpu_06d"  "logs/06d_train_production.log"

echo ""
echo -e "  ${BLUE}── CPU — AGGREGATION & ANALYSIS ──────────────────────${NC}"
stage_status "07a · Aggregate baseline"            "logs/.done_07a_aggregate_baseline"    "logs/07a_aggregate_baseline.log"
stage_status "07b · Aggregate ablation"            "logs/.done_07b_aggregate_ablation"    "logs/07b_aggregate_ablation.log"
stage_status "07c · Aggregate production"          "logs/.done_07c_aggregate_production"  "logs/07c_aggregate_production.log"
stage_status "08  · Paper figure generation"       "logs/.done_08_generate_paper_figures" "logs/08_generate_paper_figures.log"
stage_status "09  · Lead-time analysis"            "logs/.done_09_lead_time_analysis"     "logs/09_lead_time_analysis.log"
stage_status "10  · HMM robustness check"          "logs/.done_10_hmm_robustness_check"   "logs/10_hmm_robustness_check.log"
stage_status "11  · Crisis validation (T1 & T2)"   "logs/.done_11_crisis_validation"      "logs/11_crisis_validation.log"

echo ""
echo -e "  ${BLUE}── OUTPUT FILES ──────────────────────────────────────${NC}"
check_file "production_results.csv"                "production_results.csv"
check_file "production_results.parquet"            "production_results.parquet"
check_file "lead_time_results.csv"                 "lead_time_results.csv"
check_file "hmm_robustness_label_agreement.csv"    "hmm_robustness_label_agreement.csv"
check_file "hmm_robustness_xgb_comparison.csv"     "hmm_robustness_xgb_comparison.csv"
check_file "crisis_validation_summary.csv"         "crisis_validation_summary.csv"
check_file "crisis_validation_stats.csv"           "crisis_validation_stats.csv"
check_file "crisis_validation_silent_events.csv"   "crisis_validation_silent_events.csv"

echo ""
echo -e "  ${BLUE}── RECENT LOG ACTIVITY ───────────────────────────────${NC}"
if ls logs/*.log 2>/dev/null | head -1 > /dev/null 2>&1; then
    ls -t logs/*.log 2>/dev/null | head -3 | while read logf; do
        LAST=$(tail -1 "$logf" 2>/dev/null | cut -c1-70)
        printf "  %-30s  %s\n" "$(basename $logf)" "$LAST"
    done
else
    echo "  No logs yet"
fi

echo ""
echo -e "  ${BLUE}── QUICK ACTIONS ─────────────────────────────────────${NC}"
echo "  bash cpu_pipeline.sh --check       # validate pre-GPU environment"
echo "  bash gpu_pipeline.sh --check       # validate GPU environment"
echo "  bash cpu_post_gpu.sh --check  # validate post-GPU environment"
echo "  bash gpu_pipeline.sh --dry-run     # preview GPU execution plan"
echo "========================================================"
echo ""
