import json
import torch
from torch.utils.data import Dataset
from blueberry.tokenizer.naive import ChatFormat, Role, Message, Dialog

class DifftuneDataset(Dataset):
    """
    差异微调数据集
    
    原理：通过answer的diff数据，模型知道奖励的确切来源，通过question的不同问法，让模型理解同义词。
    本质：对比学习、强化学习，让模型学会区分哪些回答是好的，哪些回答是不好的，让模型学会选择好的回答
    
    所有chosen_answer和rejected_answer都是list，每个元素是一个回答
    任意的chosen_answer和rejected_answer的组合都可以形成一个训练样本
    数据格式：
    ```
    [
        {
            'question': str
            'chosen_answer': [str, ...]
            'rejected_answer': [str, ...]
        }
    ]
    ```
    """
    def __init__(self, data, tokenizer, max_seq_len, output_mask=True, mask_first_token=True, 
                 system_role='系统', user_role='用户', assistant_role='助手',
                 system_content='',
                 ):
        """
        data: list 
            Each element is a dict with keys:
                'question': str
                'chosen_answer': [str, ...]
                'rejected_answer': [str, ...]
        """
        self.data = self.flatten_data(data)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.output_mask = output_mask
        self.mask_first_token = mask_first_token
        self.system_role = system_role
        self.user_role = user_role
        self.assistant_role = assistant_role
        self.system_content = system_content
    
    def flatten_data(self, data):
        """
        将数据展平，每个样本包含一个question和一对chosen_answer和rejected_answer
        """
        flattened_data = []
        for item in data:
            question = item['question']
            chosen_answers = item['chosen_answer']
            rejected_answers = item['rejected_answer']
            for chosen_answer in chosen_answers:
                for rejected_answer in rejected_answers:
                    flattened_data.append({
                        'question': question,
                        'chosen_answer': chosen_answer,
                        'rejected_answer': rejected_answer
                    })
        return flattened_data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        print(f'item: {item}')
        chat_format = ChatFormat(self.tokenizer)
        question = item['question']
        chosen_answer = item['chosen_answer']
        rejected_answer = item['rejected_answer']
        chosen_tokens, chosen_info = chat_format.encode_dialog([
            Message(role=self.system_role, content=self.system_content),
            Message(role=self.user_role, content=question),
            Message(role=self.assistant_role, content=chosen_answer)
        ])
        rejected_tokens, rejected_info = chat_format.encode_dialog([
            Message(role=self.system_role, content=self.system_content) ,
            Message(role=self.user_role, content=question),
            Message(role=self.assistant_role, content=rejected_answer)
        ])
        chosen_message_seps = chosen_info['message_seps']
        rejected_message_seps = rejected_info['message_seps']
        # chosen/rejected答案不一定长度相同
        # assert chosen_message_seps[-1] == rejected_message_seps[-1]
        assert len(chosen_tokens) <= self.max_seq_len + 1, f'Too long text will cause information loss: ```{question=}, {chosen_answer=}```, with len(tokens)={len(chosen_tokens)=}, max_len={self.max_seq_len}'
        chosen_encoded = chosen_tokens[:self.max_seq_len+1] + [self.tokenizer.padding_id] * (self.max_seq_len + 1 - len(chosen_tokens[:self.max_seq_len+1]))  # Padding/truncating
        chosen_input_seq = torch.tensor(chosen_encoded[0:-1], dtype=torch.long)
        chosen_target_seq = torch.tensor(chosen_encoded[1:], dtype=torch.long)
        rejected_encoded = rejected_tokens[:self.max_seq_len+1] + [self.tokenizer.padding_id] * (self.max_seq_len + 1 - len(rejected_tokens[:self.max_seq_len+1]))  # Padding/truncating
        rejected_input_seq = torch.tensor(rejected_encoded[0:-1], dtype=torch.long)
        rejected_target_seq = torch.tensor(rejected_encoded[1:], dtype=torch.long)
        if self.output_mask:
            chosen_finetune_mask = torch.zeros_like(chosen_target_seq)
            # 只对助手回答部分计算微调损失: 只计算assistant的token的概率乘积
            chosen_finetune_mask[chosen_message_seps[-1]:] = 1
            chosen_finetune_mask[chosen_target_seq == self.tokenizer.padding_id] = 0 # 不计算padding部分的损失

            rejected_finetune_mask = torch.zeros_like(rejected_target_seq)
            rejected_finetune_mask[rejected_message_seps[-1]:] = 1
            rejected_finetune_mask[rejected_target_seq == self.tokenizer.padding_id] = 0 # 不计算padding部分的损失

            # 对问题部分的预训练，本质是：要求问题部分为合理的对话序列
            # assert sep == rejected_message_seps[-1]
            chosen_pretrain_mask = torch.ones_like(chosen_input_seq)
            chosen_pretrain_mask[chosen_message_seps[-1]:] = 0 
            chosen_pretrain_mask[chosen_target_seq == self.tokenizer.padding_id] = 0
            if self.mask_first_token:
                chosen_pretrain_mask[0] = 0
            rejected_pretrain_mask = torch.ones_like(rejected_input_seq)
            rejected_pretrain_mask[rejected_message_seps[-1]:] = 0
            rejected_pretrain_mask[rejected_target_seq == self.tokenizer.padding_id] = 0
            if self.mask_first_token:
                rejected_pretrain_mask[0] = 0

            # return (chosen_input_seq, chosen_target_seq), (rejected_input_seq, rejected_target_seq)
            return (chosen_input_seq, chosen_target_seq, (chosen_finetune_mask, chosen_pretrain_mask)), (rejected_input_seq, rejected_target_seq, (rejected_finetune_mask, rejected_pretrain_mask))
        else:
            return (chosen_input_seq, chosen_target_seq), (rejected_input_seq, rejected_target_seq)
    
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
