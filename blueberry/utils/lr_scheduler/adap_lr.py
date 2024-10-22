import random
import numpy as np
from collections import defaultdict

class AdapLR:
    """
    Adaptative learning rate scheduler
    """
    def __init__(self, optimizer,
                 initial_lr: float, 
                 enlarge_ratio: float = 1.01,
                 shrink_ratio: float = 0.99,
                 loss_eps: float = 1e-8
                 ):
        assert 0.8 < shrink_ratio < 1, 'shrink_ratio must be less than 1 and greater than 0.8'
        assert 1 < enlarge_ratio < 1.2, 'enlarge_ratio must be greater than 1 and less than 1.2'
        assert loss_eps > 0, 'loss_eps must be greater than 0'
        # assert loss_ratio_eps > 0, 'loss_ratio_eps must be greater than 0'
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.enlarge_ratio = enlarge_ratio
        self.shrink_ratio = shrink_ratio
        self.loss_eps = loss_eps
        # self.loss_ratio_eps = loss_ratio_eps
        # 每个epoch一变
        self._enlarged_lr = self.initial_lr * enlarge_ratio
        self._shrinked_lr = self.initial_lr * shrink_ratio
        self._original_lr = self.initial_lr
        # 每个epoch一变，记录每个lr的各batch的loss
        self._loss_dict = defaultdict(list)
        # 每个epoch一变，记录当前状态
        self._next_state = 0
        self.apply_lr(self._original_lr)
        
    def apply_lr(self, lr):
        self._last_lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_last_lr(self):
        return self._last_lr
    
    def step_batch(self, batch_loss):
        self._loss_dict[self._next_state].append(batch_loss)
        # self._next_state = (self._next_state + 1) % 3 - 1
        self._next_state = random.choice([-1, 0, 1])
        if self._next_state == 0:
            lr = self._original_lr
        elif self._next_state == 1:
            lr = self._enlarged_lr
        else:
            lr = self._shrinked_lr

        self.apply_lr(lr)

    def step(self):
        loss_enlarge, loss_shrink, loss_original = None, None, None
        if self._loss_dict[-1]:
            loss_shrink = np.mean(self._loss_dict[-1])
        if self._loss_dict[1]:
            loss_enlarge = np.mean(self._loss_dict[1])
        if self._loss_dict[0]:
            loss_original = np.mean(self._loss_dict[0])
        
        self._loss_dict.clear()
        # 如果loss_dict中没有数据，则不更新lr，说明数据量比较少
        if loss_enlarge is None or loss_shrink is None or loss_original is None:
            return
        
        # 也可以根据p值来更新
        if loss_shrink - self.loss_eps < loss_original < loss_enlarge + self.loss_eps:
            self._shrinked_lr = self._original_lr
            self._original_lr = self._enlarged_lr
            self._enlarged_lr = self.enlarge_ratio * self._enlarged_lr
        elif loss_enlarge + self.loss_eps > loss_original > loss_shrink - self.loss_eps:
            self._enlarged_lr = self._original_lr
            self._original_lr = self._shrinked_lr
            self._shrinked_lr = self.shrink_ratio * self._shrinked_lr
        else:
            pass
