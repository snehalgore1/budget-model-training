# Budget Model Training - SFT for Large Reasoning Models

**Supervised Fine-Tuning (SFT) for Difficulty-Aware Token Budget Prediction**

Train a neural model to predict optimal reasoning token budgets for Large Reasoning Models (LRMs) based on query difficulty. This implementation uses LoRA-adapted Llama-3.2-3B with a custom regression head, optimized for L4 GPU (24GB VRAM).

---

## 🎯 Project Overview

Large Reasoning Models (LRMs) generate extended chains of thought before answering queries. However, different problems require different amounts of reasoning:
- Simple arithmetic: ~100 tokens
- Complex math proofs: ~10,000 tokens
- Advanced AIME problems: ~32,000 tokens

This project trains a **Budget Model** that learns to predict the optimal token budget by analyzing query difficulty across multiple domains (math, reasoning, code).

### Key Features

- **Efficient Training**: LoRA fine-tuning reduces trainable parameters by 99%
- **GPU Optimized**: Fits 3B parameter model on single L4 GPU (24GB)
- **Mixed Precision**: bfloat16 for 2x memory efficiency
- **Production Ready**: Gradient checkpointing, gradient accumulation, warmup scheduling
- **Comprehensive Metrics**: MAE, RMSE, MAPE, underrun/overrun rates

---
## 📊 Dataset

Download the training data from Google Drive:

**Link:** [Download Data (143 MB)](https://drive.google.com/your-share-link-here)

After downloading, extract to `data/` folder:
```bash
mkdir data
# Extract your downloaded files to data/
# Should have: data/train.jsonl, data/val.jsonl, data/test.jsonl
```
---

## 📊 Architecture

```
Input Query
    ↓
Llama-3.2-3B-Instruct (frozen base + LoRA adapters)
    ↓
Last Token Hidden State (3072-dim)
    ↓
3-Layer MLP Regression Head
    ├─ Linear(3072 → 1024) + LayerNorm + ReLU + Dropout
    ├─ Linear(1024 → 256) + LayerNorm + ReLU + Dropout
    └─ Linear(256 → 1)
    ↓
Predicted Token Budget (scalar)
```

**LoRA Configuration:**
- Rank: 32
- Alpha: 64
- Dropout: 0.05
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

**Training:**
- Loss: Huber Loss (δ=1000)
- Optimizer: AdamW (lr=2e-4, weight_decay=0.01)
- Scheduler: Linear warmup (500 steps) + linear decay
- Batch Size: 2 (physical) × 16 (gradient accumulation) = 32 (effective)

---

## 📁 Project Structure

```
budget-model-training/
├── train_sft_l4.py           # Main training script (self-contained)
├── UPLOAD_INSTRUCTIONS.md     # Step-by-step GCP deployment guide
├── README.md                  # This file
│
├── data/                      # Your training data (not included)
│   ├── train.jsonl           # Training set
│   ├── val.jsonl             # Validation set
│   └── test.jsonl            # Test set (optional)
│
├── checkpoints/               # Created during training
│   ├── best_model.pt         # Best model by validation MAE
│   ├── epoch_1.pt
│   ├── epoch_2.pt
│   └── epoch_3.pt
│
└── logs/                      # Created during training
    └── training_history.json
```

---

## 🚀 Quick Start

### Prerequisites

- **Hardware**: CUDA GPU with 24GB+ VRAM (e.g., NVIDIA L4, A10, RTX 4090)
- **Software**: Python 3.10+, CUDA 12.1+

### Installation

```bash
# Clone repository
git clone https://github.com/snehalgore1/budget-model-training.git
cd budget-model-training

# Install dependencies
pip install torch transformers peft accelerate tqdm numpy scikit-learn

# Verify CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Data Format

Your data files should be in JSONL format with one example per line:

```jsonl
{"query": "Solve: 2x + 5 = 13", "optimal_budget": 150}
{"query": "Prove that sqrt(2) is irrational", "optimal_budget": 8500}
{"query": "Write a function to merge two sorted arrays", "optimal_budget": 2300}
```

**Required fields:**
- `query` (string): The input problem/question
- `optimal_budget` (int): Ground truth token budget (can also be named `budget`)

### Training

```bash
# Local training
python train_sft_l4.py

# With tmux (recommended for long training runs)
tmux new -s training
python train_sft_l4.py
# Detach: Ctrl+B, then D
```

**Expected Output:**
```
======================================================================
SFT Budget Model Training - L4 GPU Optimized
======================================================================

✓ GPU: NVIDIA L4
  Memory: 24.0 GB

Loading Data...
  ✓ Loaded 15,000 examples from train.jsonl
  ✓ Loaded 1,500 examples from val.jsonl

Training...
Epoch 1/3 [████████████████████████████████████] 100%
  Train Loss: 0.8234
  Val MAE: 856.42 tokens

Epoch 2/3 [████████████████████████████████████] 100%
  Train Loss: 0.5621
  Val MAE: 742.35 tokens ← New best!

Epoch 3/3 [████████████████████████████████████] 100%
  Train Loss: 0.4903
  Val MAE: 738.12 tokens ← New best!

======================================================================
Training Complete!
======================================================================
Best validation MAE: 738.12 tokens
✅ GOOD: MAE < 750 tokens
✅ GOOD: Underrun rate 68.23% < 70%
```

---

## ☁️ Google Cloud Platform Training

For cloud-based training on GCP L4 instances, see **[UPLOAD_INSTRUCTIONS.md](UPLOAD_INSTRUCTIONS.md)** for a complete step-by-step guide.

**Quick summary:**
```bash
# 1. Create L4 VM
gcloud compute instances create budget-trainer-l4 \
  --zone=us-central1-a \
  --machine-type=g2-standard-4 \
  --accelerator=type=nvidia-l4,count=1 \
  ...

# 2. Upload files
gcloud compute scp train_sft_l4.py data/ budget-trainer-l4:~/ 

# 3. SSH and train
gcloud compute ssh budget-trainer-l4
python3 train_sft_l4.py
```

**Costs:** ~$19 for 24-hour training run ($0.60/hour for L4 GPU)

---

## 📈 Evaluation Metrics

The model is evaluated on:

| Metric | Description | Target |
|--------|-------------|--------|
| **MAE** | Mean Absolute Error (tokens) | < 750 (good), < 500 (excellent) |
| **RMSE** | Root Mean Squared Error (tokens) | Lower is better |
| **MAPE** | Mean Absolute Percentage Error | < 15% |
| **Underrun Rate** | % of predictions below optimal | < 70% (avoid starvation) |
| **Overrun Rate** | % of predictions above optimal | - |
| **Avg Underrun** | Average underrun amount (tokens) | Minimize |

**Success Criteria:**
- ✅ **Good**: MAE < 750, Underrun < 70%
- ✅ **Excellent**: MAE < 500, Underrun < 60%
- ⚠️ **Needs Improvement**: MAE ≥ 1500 or Underrun ≥ 80%

---

## 🔧 Configuration

Edit the `CONFIG` dictionary in `train_sft_l4.py`:

```python
CONFIG = {
    # Model
    'model_name': 'meta-llama/Llama-3.2-3B-Instruct',
    
    # Training
    'batch_size': 2,              # Adjust for your GPU
    'grad_accum_steps': 16,       # Effective batch = 32
    'num_epochs': 3,
    'learning_rate': 2e-4,
    'max_length': 512,            # Max input sequence length
    
    # LoRA
    'lora_r': 32,                 # LoRA rank
    'lora_alpha': 64,             # LoRA alpha
    'lora_dropout': 0.05,
    
    # Loss
    'huber_delta': 1000.0,        # Huber loss threshold
}
```

**Memory optimization tips:**
- Reduce `batch_size` if OOM errors occur
- Increase `grad_accum_steps` to maintain effective batch size
- Lower `max_length` for shorter sequences
- Reduce `lora_r` for fewer trainable parameters

---

## 📦 Model Checkpoints

Each checkpoint contains:
```python
{
    'epoch': 2,
    'model_state_dict': {...},           # Model weights
    'optimizer_state_dict': {...},        # Optimizer state
    'scheduler_state_dict': {...},        # LR scheduler state
    'val_mae': 742.35,                    # Validation MAE
    'val_metrics': {...},                 # All validation metrics
    'config': {...}                       # Training configuration
}
```

**Loading a checkpoint:**
```python
import torch
from train_sft_l4 import BudgetRegressionModel, CONFIG

# Load checkpoint
checkpoint = torch.load('checkpoints/best_model.pt')

# Recreate model
model = BudgetRegressionModel(CONFIG)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Loaded model from epoch {checkpoint['epoch']}")
print(f"Validation MAE: {checkpoint['val_mae']:.2f} tokens")
```

---

## 🧪 Inference Example

```python
import torch
from transformers import AutoTokenizer
from train_sft_l4 import BudgetRegressionModel, CONFIG, PROMPT_TEMPLATE

# Load model
model = BudgetRegressionModel(CONFIG)
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to('cuda').eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Predict
query = "Solve the differential equation: dy/dx = x^2 + y"
prompt = PROMPT_TEMPLATE.format(query=query)

encoding = tokenizer(
    prompt,
    max_length=512,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
).to('cuda')

with torch.no_grad():
    prediction = model(
        encoding['input_ids'],
        encoding['attention_mask']
    ).item()

print(f"Predicted budget: {prediction:.0f} tokens")
# Output: Predicted budget: 3420 tokens
```

---

## 📊 Training Data Requirements

**Recommended dataset size:**
- Minimum: 5,000 examples
- Good: 15,000+ examples
- Excellent: 50,000+ examples

**Domain distribution** (example):
- Mathematical reasoning: 40% (GSM8K, MATH, AIME)
- General reasoning: 30% (MMLU-Pro)
- Code generation: 30% (LiveCodeBench)

**Budget distribution:**
- Simple (< 500 tokens): 30%
- Medium (500-5000 tokens): 50%
- Hard (> 5000 tokens): 20%

---

## 🐛 Troubleshooting

### Out of Memory (OOM)

```python
# Reduce batch size
CONFIG['batch_size'] = 1

# Increase gradient accumulation
CONFIG['grad_accum_steps'] = 32  # Keep effective batch = 32

# Shorter sequences
CONFIG['max_length'] = 384
```

### Slow Training

```python
# Check GPU utilization
watch -n 1 nvidia-smi

# Increase batch size if GPU < 90% utilized
CONFIG['batch_size'] = 4
CONFIG['grad_accum_steps'] = 8

# Use more workers
CONFIG['num_workers'] = 4
```

### Poor Performance (High MAE)

- **Check data quality**: Ensure budgets are realistic
- **Increase training epochs**: Try 5-10 epochs
- **Adjust learning rate**: Try 1e-4 or 5e-4
- **Larger LoRA rank**: Increase `lora_r` to 64
- **More training data**: Aim for 50K+ examples

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{budgetmodel2024,
  author = {Your Name},
  title = {Budget Model Training for Large Reasoning Models},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/snehalgore1/budget-model-training}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Meta AI** for Llama-3.2 models
- **Hugging Face** for Transformers and PEFT libraries
- **Project Blackberry** academic midterm project

---

## 📞 Contact

For questions or issues:
- Open an issue on [GitHub Issues](https://github.com/snehalgore1/budget-model-training/issues)
- Email: ssgore@usc.edu or snehal.gore1@gmail.com

---

## 🔄 Future Work

- [ ] Multi-task learning across domains
- [ ] Classification-based approach (budget bins)
- [ ] Uncertainty quantification
- [ ] Distillation to smaller models
- [ ] Online learning from LRM feedback

---

**Happy Training! 🚀**
