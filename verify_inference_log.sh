#!/usr/bin/env bash
# verify_inference_log.sh -- sanity-check a fresh post-fix inference run
# before syncing to Hugging Face or referencing it in correspondence.
set -e

LOG="logs/inference/inference_log.csv"
OUT="logs/inference/outcome_log.csv"

echo "=== inference_log.csv ==="
wc -l "$LOG"
head -1 "$LOG"
echo "--- first row ---"
sed -n '2p' "$LOG"
echo "--- last row ---"
tail -1 "$LOG"

echo ""
echo "=== stress_prob distribution ==="
awk -F, 'NR>1 {print $4}' "$LOG" | sort -n > /tmp/probs.txt
echo "min:  $(head -1 /tmp/probs.txt)"
echo "max:  $(tail -1 /tmp/probs.txt)"
awk -F, 'NR>1 {sum+=$4; n++} END {print "mean: " sum/n}' "$LOG"
echo "count stress_prob > 0.85 (warning threshold): $(awk -F, 'NR>1 && $4>0.85 {c++} END {print c+0}' "$LOG")"
echo "count stress_prob > 0.95 (near-certain, flag if frequent): $(awk -F, 'NR>1 && $4>0.95 {c++} END {print c+0}' "$LOG")"

echo ""
echo "=== sample rows ==="
head -4 "$LOG"
echo "..."
tail -3 "$LOG"

echo ""
echo "=== outcome_log.csv ==="
if [ -f "$OUT" ]; then
    wc -l "$OUT"
else
    echo "does not exist (fine if no warnings fired)"
fi

echo ""
echo "=== warnings fired (col 6 == 1) ==="
awk -F, 'NR>1 && $6==1 {c++} END {print c+0}' "$LOG"
