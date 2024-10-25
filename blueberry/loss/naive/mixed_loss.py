import torch
import torch.nn as nn
from transformers import AutoTokenizer

class MixedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.finetune_loss = nn.CrossEntropyLoss(ignore_index=tokenizer.padding_id, reduction='none')
        self.pretrain_loss = nn.CrossEntropyLoss(ignore_index=tokenizer.padding_id, reduction='none')
    
    def forward(self, logits, targets, finetune_mask, pretrain_mask):
        ft_loss = self.finetune_loss(logits.view(-1, logits.size(-1)), targets.view(-1))
        ft_loss = (ft_loss * finetune_mask.view(-1)).sum() / finetune_mask.sum()
        
        pt_loss = self.pretrain_loss(logits.view(-1, logits.size(-1)), targets.view(-1))
        pt_loss = (pt_loss * pretrain_mask.view(-1)).sum() / pretrain_mask.sum()
        
        return self.alpha * ft_loss + (1 - self.alpha) * pt_loss
