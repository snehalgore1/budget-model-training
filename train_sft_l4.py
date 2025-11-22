#!/usr/bin/env python3
"""
Self-contained SFT training script for L4 GPU (24GB).
Optimized for Google Cloud Platform L4 instances.

This script contains EVERYTHING needed - just upload this + data files!

Usage:
    python3 train_sft_l4.py

Requirements:
    - CUDA-capable GPU (24GB+ recommended)
    - PyTorch, Transformers, PEFT
    - Data files: data/train.jsonl, data/val.jsonl, data/test.jsonl
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import numpy as np
import os
from datetime import datetime

# ============================================================================
# CONFIGURATION (Optimized for L4 24GB VRAM)
# ============================================================================

CONFIG = {
    # Model
    'model_name': 'meta-llama/Llama-3.2-3B-Instruct',

    # Paths
    'data_dir': 'data',
    'output_dir': 'checkpoints',
    'log_dir': 'logs',

    # Training (L4 24GB optimized)
    'batch_size': 4,              # Small batch for 24GB
    'grad_accum_steps': 8,        # Effective batch = 32
    'max_length': 512,
    'num_epochs': 3,
    'learning_rate': 2e-4,
    'warmup_steps': 500,
    'weight_decay': 0.01,
    'max_grad_norm': 1.0,

    # LoRA parameters
    'lora_r': 32,
    'lora_alpha': 64,
    'lora_dropout': 0.05,

    # Loss function
    'huber_delta': 1000.0,

    # Device settings
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'mixed_precision': True,  # Use bfloat16 for efficiency
    'num_workers': 2,
}

# ============================================================================
# DATA LOADING
# ============================================================================

def load_jsonl(filepath):
    """
    Load data from JSONL file.

    Each line should be: {"query": "...", "budget": 123} or {"query": "...", "optimal_budget": 123}
    """
    data = []
    print(f"  Loading {filepath}...")

    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                example = json.loads(line)

                # Normalize field name (handle both 'budget' and 'optimal_budget')
                if 'budget' in example and 'optimal_budget' not in example:
                    example['optimal_budget'] = example['budget']

                if 'query' not in example or 'optimal_budget' not in example:
                    continue

                data.append(example)

            except json.JSONDecodeError:
                if line_num <= 10:  # Only warn for first few
                    print(f"    Warning: Skipping malformed line {line_num}")
                continue

    print(f"  ✓ Loaded {len(data):,} examples")
    return data

# ============================================================================
# DATASET
# ============================================================================

PROMPT_TEMPLATE = """Analyze the following query and predict the optimal number of reasoning tokens needed.

Query: {query}

Optimal token budget:"""

class BudgetDataset(Dataset):
    """PyTorch Dataset for budget prediction."""

    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        query = example['query']
        budget = example['optimal_budget']

        # Format prompt
        prompt = PROMPT_TEMPLATE.format(query=query)

        # Tokenize
        encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'budget': torch.tensor(budget, dtype=torch.float32)
        }

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class RegressionHead(nn.Module):
    """3-layer MLP regression head."""

    def __init__(self, hidden_size=3072, dropout=0.1):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.layers(x).squeeze(-1)

class BudgetRegressionModel(nn.Module):
    """Complete model: LLM with LoRA + Regression Head."""

    def __init__(self, config):
        super().__init__()

        # Load base model
        print(f"\nLoading {config['model_name']}...")
        model_config = AutoConfig.from_pretrained(config['model_name'])
        self.hidden_size = model_config.hidden_size

        self.base_model = AutoModelForCausalLM.from_pretrained(
            config['model_name'],
            torch_dtype=torch.bfloat16 if config['mixed_precision'] else torch.float32,
            device_map='auto'
        )
        print(f"✓ Model loaded (hidden size: {self.hidden_size})")

        # Apply LoRA
        print("\nApplying LoRA adapters...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config['lora_r'],
            lora_alpha=config['lora_alpha'],
            lora_dropout=config['lora_dropout'],
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            bias="none"
        )

        self.base_model = get_peft_model(self.base_model, lora_config)
        self.base_model.print_trainable_parameters()

        # Add regression head
        print("\nAdding regression head...")
        self.regression_head = RegressionHead(self.hidden_size)

        regression_params = sum(p.numel() for p in self.regression_head.parameters())
        print(f"✓ Regression head parameters: {regression_params:,}")

    def forward(self, input_ids, attention_mask):
        # Get hidden states from LLM
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        # Extract last hidden state at final token position
        last_hidden_state = outputs.hidden_states[-1]
        batch_size = input_ids.shape[0]
        sequence_lengths = attention_mask.sum(dim=1) - 1

        last_token_hidden = last_hidden_state[
            torch.arange(batch_size, device=last_hidden_state.device),
            sequence_lengths
        ]

        # Pass through regression head
        predictions = self.regression_head(last_token_hidden)
        return predictions

# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(predictions, targets):
    """Compute evaluation metrics."""
    predictions = np.asarray(predictions).flatten()
    targets = np.asarray(targets).flatten()

    abs_errors = np.abs(predictions - targets)

    mae = abs_errors.mean()
    rmse = np.sqrt(((predictions - targets) ** 2).mean())
    mape = (abs_errors / (targets + 1e-8) * 100).mean()

    underrun_rate = (predictions < targets).mean() * 100
    overrun_rate = (predictions > targets).mean() * 100

    underruns = targets[predictions < targets] - predictions[predictions < targets]
    avg_underrun = underruns.mean() if len(underruns) > 0 else 0.0

    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'mape': float(mape),
        'underrun_rate': float(underrun_rate),
        'overrun_rate': float(overrun_rate),
        'avg_underrun': float(avg_underrun)
    }

# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, dataloader, optimizer, scheduler, device, config, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")

    for step, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['budget'].to(device)

        # Forward pass
        predictions = model(input_ids, attention_mask)

        # Compute loss (Huber loss)
        loss = F.huber_loss(predictions, targets, delta=config['huber_delta'])
        loss = loss / config['grad_accum_steps']

        # Backward pass
        loss.backward()

        # Gradient accumulation
        if (step + 1) % config['grad_accum_steps'] == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * config['grad_accum_steps']
        pbar.set_postfix({
            'loss': f"{loss.item() * config['grad_accum_steps']:.4f}",
            'lr': f"{scheduler.get_last_lr()[0]:.2e}"
        })

    return total_loss / len(dataloader)

def evaluate(model, dataloader, device, config):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['budget'].to(device)

            predictions = model(input_ids, attention_mask)
            loss = F.huber_loss(predictions, targets, delta=config['huber_delta'])

            total_loss += loss.item()
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    metrics = compute_metrics(all_predictions, all_targets)
    metrics['loss'] = total_loss / len(dataloader)

    return metrics

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("SFT Budget Model Training - L4 GPU Optimized")
    print("="*70)
    print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check GPU
    if torch.cuda.is_available():
        print(f"\n✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("\n⚠️  WARNING: No GPU found! Training will be very slow.")
        print("   This script is designed for GPU training.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return

    # Create directories
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    os.makedirs(CONFIG['log_dir'], exist_ok=True)

    # Load data
    print(f"\n{'='*70}")
    print("Loading Data")
    print(f"{'='*70}")

    train_data = load_jsonl(f"{CONFIG['data_dir']}/train.jsonl")
    val_data = load_jsonl(f"{CONFIG['data_dir']}/val.jsonl")

    if not train_data or not val_data:
        print("\n✗ Error: No data loaded! Check that data files exist in 'data/' directory.")
        return

    print(f"\n✓ Total training examples: {len(train_data):,}")
    print(f"✓ Total validation examples: {len(val_data):,}")

    # Initialize tokenizer
    print(f"\n{'='*70}")
    print("Loading Tokenizer")
    print(f"{'='*70}")

    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"✓ Tokenizer loaded")

    # Create datasets
    print(f"\n{'='*70}")
    print("Creating Datasets")
    print(f"{'='*70}")

    train_dataset = BudgetDataset(train_data, tokenizer, CONFIG['max_length'])
    val_dataset = BudgetDataset(val_data, tokenizer, CONFIG['max_length'])

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )

    print(f"✓ Train batches: {len(train_loader):,}")
    print(f"✓ Val batches:   {len(val_loader):,}")

    # Initialize model
    print(f"\n{'='*70}")
    print("Initializing Model")
    print(f"{'='*70}")

    model = BudgetRegressionModel(CONFIG)

    # Setup optimizer and scheduler
    print(f"\n{'='*70}")
    print("Setup Training")
    print(f"{'='*70}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )

    total_steps = len(train_loader) * CONFIG['num_epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=CONFIG['warmup_steps'],
        num_training_steps=total_steps
    )

    print(f"✓ Optimizer: AdamW (lr={CONFIG['learning_rate']})")
    print(f"✓ Scheduler: Linear warmup + decay")
    print(f"✓ Total training steps: {total_steps:,}")
    print(f"✓ Warmup steps: {CONFIG['warmup_steps']}")

    # Training loop
    print(f"\n{'='*70}")
    print("Starting Training")
    print(f"{'='*70}")
    print(f"Configuration:")
    print(f"  Epochs: {CONFIG['num_epochs']}")
    print(f"  Batch size: {CONFIG['batch_size']}")
    print(f"  Gradient accumulation: {CONFIG['grad_accum_steps']}")
    print(f"  Effective batch size: {CONFIG['batch_size'] * CONFIG['grad_accum_steps']}")
    print(f"  Max sequence length: {CONFIG['max_length']}")
    print(f"  Huber delta: {CONFIG['huber_delta']}")

    best_val_mae = float('inf')
    history = []

    for epoch in range(CONFIG['num_epochs']):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch + 1}/{CONFIG['num_epochs']}")
        print(f"{'='*70}")

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            CONFIG['device'], CONFIG, epoch
        )

        print(f"\n✓ Train Loss: {train_loss:.4f}")

        # Validate
        print("\nValidating...")
        val_metrics = evaluate(model, val_loader, CONFIG['device'], CONFIG)

        print(f"\nValidation Metrics:")
        print(f"{'='*70}")
        print(f"  Loss:        {val_metrics['loss']:.4f}")
        print(f"  MAE:         {val_metrics['mae']:.2f} tokens")
        print(f"  RMSE:        {val_metrics['rmse']:.2f} tokens")
        print(f"  MAPE:        {val_metrics['mape']:.2f}%")
        print(f"  Underrun:    {val_metrics['underrun_rate']:.2f}%")
        print(f"  Overrun:     {val_metrics['overrun_rate']:.2f}%")
        print(f"  Avg Underrun: {val_metrics['avg_underrun']:.2f} tokens")
        print(f"{'='*70}")

        # Save history
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_metrics['loss'],
            'val_mae': val_metrics['mae'],
            'val_rmse': val_metrics['rmse'],
            'val_underrun_rate': val_metrics['underrun_rate']
        })

        # Save best model
        if val_metrics['mae'] < best_val_mae:
            best_val_mae = val_metrics['mae']
            checkpoint_path = f"{CONFIG['output_dir']}/best_model.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_mae': best_val_mae,
                'val_metrics': val_metrics,
                'config': CONFIG
            }, checkpoint_path)
            print(f"\n✅ Saved new best model (MAE: {best_val_mae:.2f} tokens)")

        # Save epoch checkpoint
        epoch_path = f"{CONFIG['output_dir']}/epoch_{epoch+1}.pt"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_mae': val_metrics['mae'],
            'config': CONFIG
        }, epoch_path)
        print(f"✓ Saved epoch checkpoint: {epoch_path}")

    # Save training history
    history_path = f"{CONFIG['log_dir']}/training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n✓ Training history saved to: {history_path}")

    # Final summary
    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}")
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nBest validation MAE: {best_val_mae:.2f} tokens")
    print(f"\nCheckpoints saved to: {CONFIG['output_dir']}/")
    print(f"Logs saved to: {CONFIG['log_dir']}/")

    # Success criteria
    print(f"\n{'='*70}")
    print("Success Criteria Check")
    print(f"{'='*70}")

    if best_val_mae < 500:
        print(f"✅ EXCELLENT: MAE {best_val_mae:.2f} < 500 tokens")
    elif best_val_mae < 750:
        print(f"✅ GOOD: MAE {best_val_mae:.2f} < 750 tokens")
    elif best_val_mae < 1500:
        print(f"✓  MINIMUM VIABLE: MAE {best_val_mae:.2f} < 1500 tokens")
    else:
        print(f"⚠️  NEEDS IMPROVEMENT: MAE {best_val_mae:.2f} >= 1500 tokens")

    final_metrics = history[-1]
    underrun = final_metrics['val_underrun_rate']

    if underrun < 60:
        print(f"✅ EXCELLENT: Underrun {underrun:.2f}% < 60%")
    elif underrun < 70:
        print(f"✅ GOOD: Underrun {underrun:.2f}% < 70%")
    elif underrun < 80:
        print(f"✓  MINIMUM VIABLE: Underrun {underrun:.2f}% < 80%")
    else:
        print(f"⚠️  NEEDS IMPROVEMENT: Underrun {underrun:.2f}% >= 80%")

    print(f"{'='*70}\n")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
