import torch
from torch.utils.data import Dataset

class MemDataset(Dataset):
    def __init__(self, data, tokenizer, seq_len=20):
        self.data = data
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        tokens = self.tokenizer.encode(text, bos=True, eos=True) # bos, eos
        encoded = tokens[:self.seq_len] + [self.tokenizer.padding_id] * (self.seq_len - len(tokens))  # Padding/truncating
        input_seq = torch.tensor(encoded[:-1], dtype=torch.long)
        target_seq = torch.tensor(encoded[1:], dtype=torch.long)
        return input_seq, target_seq


training_dataset = [
    f'{i} x {j} = {i*j}'
    for j in range(10)
    for i in range(10)
]