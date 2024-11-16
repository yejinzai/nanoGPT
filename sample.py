"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
import time
import numpy as np
import math

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

eval_iters = 100  # Number of iterations for evaluation

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

start_time = time.time()
# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')

end_time = time.time()
elapsed_time = end_time - start_time
tokens_per_second = (num_samples * max_new_tokens) / elapsed_time
print(f"Inference time for {num_samples} samples of {max_new_tokens} tokens each: {elapsed_time:.4f} seconds")
print(f"Tokens per second: {tokens_per_second:.2f}")

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

