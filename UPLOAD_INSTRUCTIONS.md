# Upload Instructions - Copy & Paste Commands

**Total time**: 30 minutes
**Total cost**: ~$19

## Prerequisites

- [ ] Google Cloud account
- [ ] Hugging Face token (get from https://huggingface.co/settings/tokens)

## Step-by-Step Commands

### 1. Create VM (2 minutes)

Open terminal and run:

```bash
# Create L4 VM
gcloud compute instances create budget-trainer-l4 \
  --zone=us-central1-a \
  --machine-type=g2-standard-4 \
  --accelerator=type=nvidia-l4,count=1 \
  --image-family=common-cu121-debian-11-py310 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-ssd \
  --maintenance-policy=TERMINATE \
  --metadata="install-nvidia-driver=True"
```

**Wait 2-3 minutes** for VM to be ready.

If you get "zone does not have enough resources", try these zones:
- `us-central1-b`
- `us-west1-a`
- `us-east1-c`

### 2. Connect to VM (1 minute)

```bash
gcloud compute ssh budget-trainer-l4 --zone=us-central1-a
```

**You're now on the VM!**

### 3. Setup VM (5 minutes)

Run these commands **on the VM**:

```bash
# Check GPU
nvidia-smi
# Should show: NVIDIA L4 (24GB)

# Install Python packages
pip3 install torch transformers peft accelerate tqdm numpy scikit-learn

# Verify CUDA
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
# Should print: CUDA: True

# Install tmux
sudo apt-get update && sudo apt-get install -y tmux

# Login to Hugging Face
pip3 install huggingface_hub
huggingface-cli login
```

When prompted, paste your Hugging Face token (get from https://huggingface.co/settings/tokens)

### 4. Upload Files (10 minutes)

**Open a NEW terminal on your computer** (keep VM SSH open).

```bash
# Navigate to this folder
cd "/Users/snehalgore/Documents/Web Tech Saty/HW2/blackberry/gcp_training"

# Compress files (faster upload)
tar -czf training.tar.gz train_sft_l4.py data/

# Upload to VM
gcloud compute scp training.tar.gz budget-trainer-l4:~/ --zone=us-central1-a
```

**Wait for upload to complete** (~5-10 minutes for 145 MB).

### 5. Extract Files on VM (1 minute)

**Go back to your VM SSH terminal** and run:

```bash
# Extract files
tar -xzf training.tar.gz

# Verify
ls -lh
ls -lh data/
```

You should see:
- `train_sft_l4.py`
- `data/train.jsonl` (127M)
- `data/val.jsonl` (14M)
- `data/test.jsonl` (2M)

### 6. Start Training (1 minute)

**On the VM**:

```bash
# Start tmux session
tmux new -s training

# Run training
python3 train_sft_l4.py
```

You'll see output like:
```
======================================================================
SFT Budget Model Training - L4 GPU Optimized
======================================================================

Start time: 2024-11-20 15:30:00

✓ GPU: NVIDIA L4
  Memory: 24.0 GB
...
```

**To detach from tmux** (training continues):
1. Press **Ctrl+B**
2. Then press **D**

**You can now close your terminal!** Training runs in the background.

### 7. Monitor (Optional)

Check progress anytime:

```bash
# Reconnect to VM
gcloud compute ssh instance-20251122-013633 --zone=us-central1-a

# Reattach to training
tmux attach -t training

# Or check GPU
watch -n 5 nvidia-smi

# Or check files
ls -lh checkpoints/
cat logs/training_history.json
```

To detach again: **Ctrl+B** then **D**

### 8. Wait for Training (~20-24 hours)

Training will complete in 20-24 hours. You'll see:

```
======================================================================
Training Complete!
======================================================================

Best validation MAE: 742.35 tokens

✅ GOOD: MAE 742.35 < 750 tokens
✅ GOOD: Underrun 68.23% < 70%
```

### 9. Download Results (2 minutes)

**From your computer**:

```bash
# Download best model
gcloud compute scp budget-trainer-l4:~/checkpoints/best_model.pt ./ --zone=us-central1-a

# Optional: Download all checkpoints
gcloud compute scp --recurse budget-trainer-l4:~/checkpoints ./ --zone=us-central1-a

# Optional: Download logs
gcloud compute scp --recurse budget-trainer-l4:~/logs ./ --zone=us-central1-a
```

### 10. Stop VM (IMPORTANT!)

```bash
# Stop VM to save money
gcloud compute instances stop budget-trainer-l4 --zone=us-central1-a
```

**Done!** You now have your trained model.

## All Commands in One Place

```bash
# ========== ON YOUR COMPUTER ==========

# Create VM
gcloud compute instances create budget-trainer-l4 \
  --zone=us-central1-a \
  --machine-type=g2-standard-4 \
  --accelerator=type=nvidia-l4,count=1 \
  --image-family=common-cu121-debian-11-py310 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-ssd \
  --maintenance-policy=TERMINATE \
  --metadata="install-nvidia-driver=True"

# Wait 2-3 minutes

# Connect
gcloud compute ssh budget-trainer-l4 --zone=us-central1-a

# ========== ON THE VM ==========

# Setup
nvidia-smi
pip3 install torch transformers peft accelerate tqdm numpy scikit-learn
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
sudo apt-get update && sudo apt-get install -y tmux
pip3 install huggingface_hub
huggingface-cli login  # Paste your HF token

# Keep VM terminal open, open NEW terminal on computer

# ========== ON YOUR COMPUTER (new terminal) ==========

# Upload
cd "/Users/snehalgore/Documents/Web Tech Saty/HW2/blackberry/gcp_training"
tar -czf training.tar.gz train_sft_l4.py data/
gcloud compute scp training.tar.gz budget-trainer-l4:~/ --zone=us-central1-a

# ========== BACK ON THE VM ==========

# Extract and run
tar -xzf training.tar.gz
ls -lh data/
tmux new -s training
python3 train_sft_l4.py

# Detach: Ctrl+B, then D

# ========== WAIT 20-24 HOURS ==========

# ========== ON YOUR COMPUTER ==========

# Download
gcloud compute scp budget-trainer-l4:~/checkpoints/best_model.pt ./ --zone=us-central1-a

# Stop VM
gcloud compute instances stop budget-trainer-l4 --zone=us-central1-a
```

## Troubleshooting

### Upload is slow
Already using compression - just wait, it should take 5-10 minutes.

### "CUDA: False"
```bash
# Check driver
nvidia-smi
# Should show L4 GPU

# Reinstall PyTorch with CUDA
pip3 install torch --index-url https://download.pytorch.org/whl/cu121
```

### Lost connection
```bash
# Reconnect
gcloud compute ssh budget-trainer-l4 --zone=us-central1-a

# Check if training is running
tmux attach -t training
```

### "No such file: data/train.jsonl"
```bash
# Make sure extraction worked
ls -lh
ls -lh data/

# If needed, re-extract
tar -xzf training.tar.gz
```

## Cost Tracking

- L4 GPU: $0.60/hour
- Running for 24 hours: $14.40
- Storage: $0.17/day
- **Total**: ~$19

Check your spend:
```bash
gcloud billing accounts list
```

## What Files Get Created

During training:
```
checkpoints/
├── best_model.pt
├── epoch_1.pt
├── epoch_2.pt
└── epoch_3.pt

logs/
└── training_history.json
```

## After Training

1. Download `checkpoints/best_model.pt`
2. Stop the VM
3. Check `logs/training_history.json` for metrics
4. Verify MAE < 750 and underrun < 70%

---

**That's it!** Just follow these steps and you'll have your model in ~24 hours for ~$19.
