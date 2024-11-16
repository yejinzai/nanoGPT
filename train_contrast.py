"""
Modified train.py for enwik8 with contrastive learning (in-batch negatives)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

import random
import matplotlib.pyplot as plt  # Added for plotting
import pandas as pd  # Added for saving loss values
# -----------------------------------------------------------------------------
# Configuration values
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'
alpha = 0.05  # weight for contrastive loss
# data
dataset = 'enwik8'
gradient_accumulation_steps = 5 * 8
batch_size = 12
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False
# optimizer
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
# learning rate decay
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5
# DDP settings
backend = 'nccl'
# system
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

# Distributed Data Parallel (DDP) setup
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Load dataset
data_dir = os.path.join('data', dataset)
def get_batch(split):
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# Contrastive learning helper functions
def generate_positive(x):
    min_substring_ratio = 0.95
    batch_size, seq_len = x.size()
    x_pos_list = []
    for seq in x:
        seq = seq.tolist()
        # Determine minimum substring length
        min_substring_len = int(seq_len * min_substring_ratio)
        if min_substring_len < 1:
            min_substring_len = 1
        # Randomly select substring length
        substring_len = random.randint(min_substring_len, seq_len)
        # Randomly select start index for the substring
        start_idx = random.randint(0, seq_len - substring_len)
        # Extract substring
        substring = seq[start_idx:start_idx + substring_len]
        # Pad substring to match seq_len if necessary
        if len(substring) < seq_len:
            padding = [0] * (seq_len - len(substring))
            substring.extend(padding)
        x_pos_list.append(substring)
    x_pos = torch.tensor(x_pos_list, dtype=torch.long, device=x.device)
    return x_pos

def contrastive_loss(anchor, positive, temperature=0.1):
    batch_size = anchor.size(0)
    anchor_positive_sim = F.cosine_similarity(anchor, positive) / temperature
    anchor_negative_sim = torch.matmul(anchor, anchor.T) / temperature
    mask = torch.eye(batch_size, device=anchor.device).bool()
    anchor_negative_sim = anchor_negative_sim.masked_fill(mask, -float('inf'))
    logits = torch.cat([anchor_positive_sim.unsqueeze(1), anchor_negative_sim], dim=1)
    labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)
    loss = F.cross_entropy(logits, labels)
    return loss

# Learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

iter_num = 0

# Initialize model
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)
meta_path = os.path.join(data_dir, 'meta2.pkl')
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    model_args['vocab_size'] = meta['vocab_size']
else:
    model_args['vocab_size'] = 50304  # default GPT-2 vocab size
print(f"Using vocab_size: {model_args['vocab_size']}")
gptconf = GPTConfig(**model_args)
model = GPT(gptconf).to(device)

# Initialize optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# Compile the model
if compile:
    model = torch.compile(model)

# Register a hook function to capture embeddings
embedding_output = None
def hook_fn(module, input, output):
    global embedding_output
    embedding_output = output

# Register the hook on the last transformer block
model.transformer.h[-1].register_forward_hook(hook_fn)

# Initialize timing and logging variables
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model  # unwrap DDP container if needed
running_mfu = -1.0

# Initialize lists to store losses for plotting
lm_losses = []
contrastive_losses = []
iterations = []

# Training loop
while True:
    # Get batch and generate positive pairs
    X, Y = get_batch('train')
    x_pos = generate_positive(X)

    # Forward pass with context and capture embeddings
    with ctx:
        # Forward pass for original batch
        logits, lm_loss = model(X, Y)  # Pass Y to calculate lm_loss
        anchor_embeddings = embedding_output.mean(dim=1)  # Mean pooling over tokens

        # Forward pass for positive batch
        logits_pos, _ = model(x_pos)  # Only need embeddings for contrastive loss
        positive_embeddings = embedding_output.mean(dim=1)  # Mean pooling over tokens

        # Compute contrastive loss with pooled embeddings
        cont_loss = contrastive_loss(anchor_embeddings, positive_embeddings, temperature=0.1)

        # Combine language modeling loss and contrastive loss
        total_loss = lm_loss + alpha * cont_loss

    # Record the losses for plotting
    if master_process:
        lm_losses.append(lm_loss.item())
        contrastive_losses.append(cont_loss.item())
        iterations.append(iter_num)

    # Backward pass and optimizer step
    scaler.scale(total_loss).backward()
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # Timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # Calculate total loss as float and print the log
        lossf = total_loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:  # Let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, lm_loss {lm_loss.item():.4f}, cont_loss {cont_loss.item():.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%")

    # Learning rate decay
    if decay_lr:
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()

# Plot and save the loss curves after training
if master_process:
    # Save losses to a CSV file
    loss_data = pd.DataFrame({
        'Iteration': iterations,
        'Language_Modeling_Loss': lm_losses,
        'Contrastive_Loss': contrastive_losses
    })
    loss_data.to_csv(os.path.join(out_dir, 'losses.csv'), index=False)

    # Plot the loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, lm_losses, label='Language Modeling Loss')
    plt.plot(iterations, contrastive_losses, label='Contrastive Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, 'loss_curves.png'))
    plt.close()