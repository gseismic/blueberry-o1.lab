import torch
from torch.utils.data import Dataset

class MemDataset(Dataset):
    def __init__(self, data, tokenizer, seq_len):
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

    @classmethod 
    def from_file(cls, filename, tokenizer, seq_len, pretrain_text_sep):
        def get_data():
            with open(filename) as f:
                text = f.read()
                lines = text.strip().split(pretrain_text_sep)
            return lines
        data = get_data()
        dataset = cls(data, tokenizer=tokenizer, seq_len=seq_len)
        return dataset
