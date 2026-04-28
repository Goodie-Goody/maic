#!/bin/bash
set -e

echo "=== MAIC RunPod Training Setup ==="

# Step 1 - Auth (Strictly JSON Key)
echo "Step 1: Authenticating to GCP..."
export GOOGLE_APPLICATION_CREDENTIALS="/workspace/maic/gcp-key.json"
gcloud auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS
gcloud config set project numeric-marker-478907-m5

# Step 2 - Clone or update repo
echo "Step 2: Getting repo..."
if [ ! -d "/workspace/maic" ]; then
  git clone https://github.com/Goodie-Goody/maic.git /workspace/maic
else
  cd /workspace/maic && git pull
fi

cd /workspace/maic

# Step 3 - Create .env
echo "Step 3: Creating .env..."
cat > .env << 'INNER_EOF'
GCP_PROJECT_ID=numeric-marker-478907-m5
GCP_BUCKET=fe-binance-data-2025
GCP_REGION=us-central1
BQ_DATASET=binance_trade_analytics
BQ_TABLE=trades_history
INNER_EOF

# Step 4 - Install dependencies
echo "Step 4: Installing dependencies..."
pip install -q --root-user-action=ignore \
  captum pyts shap polars hmmlearn pyarrow \
  google-cloud-storage google-cloud-bigquery \
  python-dotenv psutil xgboost scikit-learn matplotlib

# Step 5 - Download pkl files
echo "Step 5: Downloading HMM models from GCS..."
mkdir -p logs
for asset in BTCUSDT ETHUSDT SOLUSDT; do
  gsutil cp "gs://fe-binance-data-2025/v2/vm_backup_20260420_1709/logs/logs/${asset}_hmm_model.pkl" logs/
done

# Step 6 - Verify GPU
echo "Step 6: Verifying GPU..."
cat > /tmp/check_gpu.py << 'INNER_EOF'
import torch
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")
INNER_EOF
python3 /tmp/check_gpu.py

echo ""
echo "=== Setup complete ==="
echo "To start the pooled training, run:"
echo "  nohup python3 scripts/06b_train_models.py > logs/06b_runpod.log 2>&1 &"
echo "  tail -f logs/06b_runpod.log"
