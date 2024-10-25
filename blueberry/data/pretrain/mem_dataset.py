import torch
from torch.utils.data import Dataset

class MemDataset(Dataset):
    def __init__(self, data, tokenizer, seq_len,
                 use_bos=True, use_eos=True, 
                 if_unknown='encode',
                 allowed_special_tokens=None, 
                 disallowed_special_tokens=None,
                 output_mask=True):
        self.data = data
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.use_bos = use_bos
        self.use_eos = use_eos
        self.if_unknown = if_unknown
        self.allowed_special_tokens = allowed_special_tokens
        self.disallowed_special_tokens = disallowed_special_tokens
        self.output_mask = output_mask
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        tokens = self.tokenizer.encode(text, bos=self.use_bos, eos=self.use_eos, 
                                       if_unknown=self.if_unknown,
                                       allowed_special_tokens=self.allowed_special_tokens,
                                       disallowed_special_tokens=self.disallowed_special_tokens) # bos, eos
        assert len(tokens) <= self.seq_len + 1, f'Too long text will cause information loss: ```{text}```, with len(tokens)={len(tokens)=}, max_len={self.seq_len}'
        encoded = tokens[:self.seq_len+1] + [self.tokenizer.padding_id] * (self.seq_len + 1 - len(tokens[:self.seq_len+1]))  # Padding/truncating
        input_seq = torch.tensor(encoded[:-1], dtype=torch.long)
        target_seq = torch.tensor(encoded[1:], dtype=torch.long)
        if self.output_mask:
            mask_seq = torch.ones_like(input_seq)
            mask_seq[target_seq == self.tokenizer.padding_id] = 0
            return input_seq, target_seq, mask_seq
        else:
            return input_seq, target_seq

    @classmethod 
    def from_file(cls, filename, tokenizer, seq_len, pretrain_text_sep, if_unknown='encode',
                 allowed_special_tokens=None, disallowed_special_tokens=None, output_mask=True,
                 use_bos=True, use_eos=True, filter=None):
        def get_data():
            with open(filename) as f:
                text = f.read()
                lines = text.strip().split(pretrain_text_sep)
            return lines
        data = get_data()
        if filter is not None:
            data = [line for line in data if filter(line)]
        dataset = cls(data, tokenizer=tokenizer, seq_len=seq_len,
                      if_unknown=if_unknown,
                      allowed_special_tokens=allowed_special_tokens,
                      disallowed_special_tokens=disallowed_special_tokens,
                      output_mask=output_mask,
                      use_bos=use_bos, use_eos=use_eos)
        return dataset
