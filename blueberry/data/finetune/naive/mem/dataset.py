import json
import torch
from torch.utils.data import Dataset
from blueberry.tokenizer.naive import ChatFormat, Role, Message, Dialog

class FinetuneDataset(Dataset):
    def __init__(self, data, tokenizer, max_seq_len, output_mask=True, mask_first_token=True, 
                 system_role='系统', user_role='用户', assistant_role='助手',
                 system_content='',
                 ):
        """
        data: list 
            Each element is a dict with keys:
                'question': str
                'answer': str
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.output_mask = output_mask
        self.mask_first_token = mask_first_token
        self.system_role = system_role
        self.user_role = user_role
        self.assistant_role = assistant_role
        self.system_content = system_content

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        chat_format = ChatFormat(self.tokenizer)
        question = item['question']
        answer = item['answer']
        tokens, info = chat_format.encode_dialog([
            Message(role=self.system_role, content=self.system_content),
            Message(role=self.user_role, content=question),
            Message(role=self.assistant_role, content=answer)
        ])
        message_seps = info['message_seps']
        assert len(tokens) <= self.max_seq_len + 1, f'Too long text will cause information loss: ```{question=}, {answer=}```, with len(tokens)={len(tokens)=}, max_len={self.max_seq_len}'
        encoded = tokens[:self.max_seq_len+1] + [self.tokenizer.padding_id] * (self.max_seq_len + 1 - len(tokens[:self.max_seq_len+1]))  # Padding/truncating
        input_seq = torch.tensor(encoded[0:-1], dtype=torch.long)
        target_seq = torch.tensor(encoded[1:], dtype=torch.long)
        if self.output_mask:
            finetune_mask = torch.zeros_like(target_seq)
            sep_assistant = message_seps[-2] # fixed: -1 -> -2
            finetune_mask[sep_assistant:] = 1 # 只对助手回答部分计算微调损失
            finetune_mask[target_seq == self.tokenizer.padding_id] = 0 # 不计算padding部分的损失

            pretrain_mask = torch.ones_like(input_seq)
            pretrain_mask[sep_assistant:] = 0
            pretrain_mask[target_seq == self.tokenizer.padding_id] = 0

            if self.mask_first_token:
                pretrain_mask[0] = 0

            return input_seq, target_seq, (finetune_mask, pretrain_mask)
        else:
            return input_seq, target_seq
    
    @staticmethod
    def read_file(filename, data_sep):
        with open(filename) as f:
            text = f.read()
            lines = [json.loads(line.strip()) for line in text.strip().split(data_sep)]
        return lines

    @classmethod 
    def from_file(cls, filename, tokenizer, max_seq_len, data_sep, output_mask=True, filter=None):
        data = cls.read_file(filename, data_sep)
        data = [item for item in data if filter(item)] if filter else data
        dataset = cls(data, tokenizer=tokenizer, max_seq_len=max_seq_len, output_mask=output_mask)
        return dataset
