"""
Sample from a trained enwik8 model at the character level
"""
import os
import pickle
import time
import math
import numpy as np
from contextlib import nullcontext
import torch
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume'  # Resume from an enwik8 training checkpoint
out_dir = 'out-enwik8-char'  # Set the directory where your enwik8 checkpoints are saved
start = "The quick brown fox"  # Starting prompt text
num_samples = 10  # Number of samples to draw
max_new_tokens = 500  # Number of tokens to generate in each sample
temperature = 0.8  # Sampling temperature
top_k = 200  # Only consider top_k tokens for sampling
seed = 1337
device = 'cpu'  # Use 'cpu' or 'cuda'
dtype = 'float32'  # Use 'float32' for CPU
compile = False  # Set to False for compatibility
# -----------------------------------------------------------------------------

# Set random seed for reproducibility
torch.manual_seed(seed)
device_type = 'cpu' if 'cpu' in device else 'cuda'
ptdtype = torch.float32 if dtype == 'float32' else torch.float16
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

eval_iters = 100  # Number of iterations for evaluation

# Load the model
if init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    model.load_state_dict(checkpoint['model'])
else:
    raise ValueError("For enwik8, set init_from to 'resume' to load from a saved checkpoint.")

model.eval().to(device)
if compile:
    model = torch.compile(model)  # Optional PyTorch 2.0 feature

# Load vocabulary from meta.pkl
meta_path = os.path.join(out_dir, 'meta.pkl')
if os.path.exists(meta_path):
    print(f"Loading character-level encoding from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    raise FileNotFoundError(f"meta.pkl not found in {out_dir}. Ensure it exists for character-level encoding.")

# Encode the starting prompt
start_ids = encode(start)
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

# Number of Parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params / 1e6:.2f}M")

# Inference Time Measurement
start_time = time.time()
with torch.no_grad():
    with ctx:
        for _ in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            generated_text = decode(y[0].tolist())
            print(generated_text)
            print('---------------')
end_time = time.time()
elapsed_time = end_time - start_time
tokens_per_second = (num_samples * max_new_tokens) / elapsed_time
print(f"Inference time for {num_samples} samples of {max_new_tokens} tokens each: {elapsed_time:.4f} seconds")
print(f"Tokens per second: {tokens_per_second:.2f}")

# Bits per Byte (BPB) Calculation
# Bits per Byte (BPB) Calculation
def calculate_bpb(model, val_data_path='data/enwik8/val.bin', block_size=None, batch_size=12):
    val_data = np.fromfile(val_data_path, dtype=np.uint16)
    num_chars = len(val_data)
    total_nll = 0.0

    if block_size is None:
        block_size = model.config.block_size  # Use model's block_size

    num_batches = num_chars // (batch_size * block_size)

    for i in range(num_batches):
        start_idx = i * block_size * batch_size
        x_data = val_data[start_idx:start_idx + block_size * batch_size]
        y_data = val_data[start_idx + 1:start_idx + 1 + block_size * batch_size]

        if len(x_data) < block_size * batch_size:
            break  # End of data reached

        x = torch.tensor(x_data, dtype=torch.long).view(batch_size, block_size).to(device)
        y = torch.tensor(y_data, dtype=torch.long).view(batch_size, block_size).to(device)

        with torch.no_grad():
            _, loss = model(x, y)
        num_tokens = x.numel()
        total_nll += loss.item() * num_tokens  # Scale by total tokens in batch

    total_tokens = num_batches * batch_size * block_size
    avg_nll = total_nll / total_tokens
    bpb = avg_nll / math.log(2)  # Convert from nats to bits per byte
    print(f"Validation loss: {avg_nll:.4f} nats per token, Bits per byte: {bpb:.4f}")
    return bpb

# Calculate BPB and print
bpb = calculate_bpb(model)
print(f"Bits per byte (bpb): {bpb:.4f}")

# Function to load the validation dataset
def load_validation_data():
    data_dir = os.path.join('data', 'enwik8')
    val_data_path = os.path.join(data_dir, 'val.bin')
    if not os.path.exists(val_data_path):
        raise FileNotFoundError(f"Validation data not found at {val_data_path}")
    with open(val_data_path, 'rb') as f:
        val_data = np.frombuffer(f.read(), dtype=np.uint16)
    return val_data

# Function to get a batch of validation data
def get_batch(val_data, idx, block_size):
    # Grab a chunk of data starting from idx
    data = val_data[idx: idx + block_size + 1]
    if len(data) < block_size + 1:
        # If not enough data, wrap around
        data = np.concatenate((data, val_data[:block_size + 1 - len(data)]))
    x = torch.tensor(data[:-1], dtype=torch.long, device=device).unsqueeze(0)
    y = torch.tensor(data[1:], dtype=torch.long, device=device).unsqueeze(0)
    return x, y

# Function to evaluate the model and compute bits per character
def evaluate_model(model):
    model.eval()
    val_data = load_validation_data()
    n = val_data.size
    total_nll = 0.0
    total_tokens = 0
    block_size = model.config.block_size  # Use the model's block_size
    with torch.no_grad():
        for _ in range(eval_iters):
            idx = np.random.randint(0, n - block_size - 1)
            x, y = get_batch(val_data, idx, block_size)
            num_tokens = x.numel()  # Total tokens in the batch
            with ctx:
                logits, loss = model(x, y)
            # Multiply loss by the number of tokens to get total NLL for this batch
            total_nll += loss.item() * num_tokens
            total_tokens += num_tokens
    avg_loss_nats = total_nll / total_tokens  # Average NLL per token
    bpc = avg_loss_nats / math.log(2)  # Convert from nats to bits per character
    print(f"Validation loss: {avg_loss_nats:.4f} nats per token, Bits per character: {bpc:.4f}")
    return bpc

# Compute bits per character
bpc = evaluate_model(model)