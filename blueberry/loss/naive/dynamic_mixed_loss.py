import torch
import torch.nn as nn

class DynamicMixedLoss(nn.Module):
    def __init__(self, epochs, initial_alpha=0.5):
        super().__init__()
        self.initial_alpha = initial_alpha
        self.epochs = epochs
        self.current_epoch = 0
        
    def update_alpha(self):
        self.current_epoch += 1
        return self.initial_alpha + (1 - self.initial_alpha) * (self.current_epoch / self.epochs)
    
    def forward(self, logits, targets, finetune_mask, pretrain_mask):
        alpha = self.update_alpha()
        return alpha * ft_loss + (1 - alpha) * pt_loss

