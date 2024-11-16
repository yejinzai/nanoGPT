# prepare.py
import os
import numpy as np
import pickle

# Define paths for input and output
input_file_path = 'data/enwik8/enwik8'
output_dir = 'data/enwik8'
os.makedirs(output_dir, exist_ok=True)

# Read the entire enwik8 file into a single string
with open(input_file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Create a character-level vocabulary
chars = sorted(set(text))
vocab_size = len(chars)
print(f'Vocab size: {vocab_size}')

# Create mappings from characters to integers and vice versa
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Encode the entire dataset into integers
data = np.array([stoi[ch] for ch in text], dtype=np.uint16)  # Save as uint16 array
data_size = len(data)
print(f'Dataset size (in characters): {data_size}')

# Define the split: 90M for training and split the remainder equally for val and test
train_size = 90_000_000
remaining_size = data_size - train_size
val_size = remaining_size // 2
test_size = remaining_size - val_size  # Ensure the remaining characters go to test

assert train_size + val_size + test_size == data_size, "Split sizes do not match the dataset size."

# Split the data
train_data = data[:train_size]
val_data = data[train_size:train_size + val_size]
test_data = data[train_size + val_size:]

# Save each split as a binary file in uint16 format
train_data.tofile(os.path.join(output_dir, 'train.bin'))
val_data.tofile(os.path.join(output_dir, 'val.bin'))
test_data.tofile(os.path.join(output_dir, 'test.bin'))

# Save vocabulary mappings
# Save vocabulary mappings as a dictionary
with open(os.path.join(output_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump({'stoi': stoi, 'itos': itos, 'vocab_size': vocab_size}, f)

print(f"Saved train, val, and test splits to {output_dir}")
print(f"Train size: {len(train_data)}, Val size: {len(val_data)}, Test size: {len(test_data)}")
