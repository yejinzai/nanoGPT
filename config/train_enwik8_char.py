# train a character-level enwik8 model
out_dir = 'out-enwik8-char'
eval_interval = 200 # evaluate model every 200 steps
eval_iters = 20
log_interval = 10

# checkpoint saving settings
always_save_checkpoint = False

# dataset and experiment setup
dataset = 'enwik8'
gradient_accumulation_steps = 1
batch_size = 12  # lower batch size for MacBook
block_size = 64  # smaller context window

# model size (reduced for MacBook training)
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0

# training hyperparameters
learning_rate = 1e-3
max_iters = 2000
lr_decay_iters = 2000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

# system and logging
device = 'cpu'
compile = False
