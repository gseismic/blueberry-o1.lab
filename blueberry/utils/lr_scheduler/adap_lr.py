import random
import numpy as np
from collections import defaultdict
from collections import deque
from ...logger import user_logger

class AdapLR:
    """
    Adaptative learning rate scheduler
    """
    def __init__(self, optimizer,
                 initial_lr: float, 
                 enlarge_ratio: float = 1.01, # 1.01,
                 shrink_ratio: float = 0.99, # 0.99,
                 adjust_period: int = 180,
                 reduce_size: int = 100,
                 loss_eps: float = 1e-8,
                 patience: int = 3,
                 logger = None
                 ):
        """
        Args:
            patience 连续3次没有下降，认为需要shrink lr
        """
        assert 0.1 < shrink_ratio < 1, 'shrink_ratio must be less than 1 and greater than 0.0'
        assert 1 < enlarge_ratio < 10.0, 'enlarge_ratio must be greater than 1 and less than 10.0'
        assert loss_eps > 0, 'loss_eps must be greater than 0'
        # assert loss_ratio_eps > 0, 'loss_ratio_eps must be greater than 0'
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.enlarge_ratio = enlarge_ratio
        self.shrink_ratio = shrink_ratio
        self.loss_eps = loss_eps
        self.patience = patience
        self.logger = logger or user_logger
        # self.loss_ratio_eps = loss_ratio_eps
        # 每个epoch一变
        self._enlarged_lr = self.initial_lr * enlarge_ratio
        self._shrinked_lr = self.initial_lr * shrink_ratio
        self._original_lr = self.initial_lr
        self.adjust_period = adjust_period
        self.reduce_size = reduce_size
        # 每个epoch一变，记录每个lr的各batch的loss
        self._loss_decr_dict = defaultdict(lambda: deque(maxlen=self.reduce_size))
        # 每个epoch一变，记录当前状态
        self._i_batch = 0
        self._prev_batch_loss = None
        self._next_state = 0
        self.patience_used = 0
        self.apply_lr(self._original_lr)
        
    def apply_lr(self, lr):
        self._last_lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_last_lr(self):
        return [self._last_lr]
    
    def step_batch(self, batch_loss):
        self._i_batch += 1
        if self._prev_batch_loss is None:
            self._prev_batch_loss = batch_loss
            return

        self._loss_decr_dict[self._next_state].append(self._prev_batch_loss - batch_loss)
        # self._next_state = (self._next_state + 1 + 1) % 3 - 1
        self._next_state = random.choice([-1, 0, 1])
        if self._next_state == 0:
            lr = self._original_lr
        elif self._next_state == 1:
            lr = self._enlarged_lr
        else:
            lr = self._shrinked_lr

        self.apply_lr(lr)
        self._prev_batch_loss = batch_loss
        if self._i_batch % self.adjust_period == 0:
            self.try_adjust_lr()
        
    def try_adjust_lr(self):
        loss_enlarge, loss_shrink, loss_original = None, None, None
        fn_reduce = np.mean # np.median
        # fn_reduce = lambda x: np.mean(x)/np.std(x) 导致步稳定，不能确保往loss下降方向精细
        if self._loss_decr_dict[-1]:
            loss_shrink = fn_reduce(list(self._loss_decr_dict[-1]))
        if self._loss_decr_dict[1]:
            loss_enlarge = fn_reduce(list(self._loss_decr_dict[1]))
        if self._loss_decr_dict[0]:
            loss_original = fn_reduce(list(self._loss_decr_dict[0]))
        
        print(len(self._loss_decr_dict[-1]))
        print(len(self._loss_decr_dict[0]))
        print(len(self._loss_decr_dict[1]))
        print(f'loss_shrink: {loss_shrink:>20.9f}')
        print(f'loss_original: {loss_original:>20.8f}')
        print(f'loss_enlarge: {loss_enlarge:>20.8f}')
        # self._loss_decr_dict.clear()
        # 如果loss_dict中没有数据，则不更新lr，说明数据量比较少
        if loss_enlarge is None or loss_shrink is None or loss_original is None:
            return
        
        loss_sum = loss_shrink + loss_original + loss_enlarge
        print(f'{loss_sum=}')
        i_max = np.argmax([loss_shrink, loss_original, loss_enlarge])
        if loss_sum < 0:
            self.patience_used += 1
        else:
            self.patience_used = 0
        
        if self.patience_used >= self.patience or i_max == 0:
            self._shrinked_lr *= self.shrink_ratio
            self._original_lr *= self.shrink_ratio
            self._enlarged_lr *= self.shrink_ratio
            self.patience_used = 0
            self.logger.debug(f'Shrink lr to {self._original_lr}')
        elif loss_sum > 0 and i_max == 2:
            self._shrinked_lr *= self.enlarge_ratio
            self._original_lr *= self.enlarge_ratio
            self._enlarged_lr *= self.enlarge_ratio
            self.logger.debug(f'Enlarge lr to {self._original_lr}')

        return
        # （1）loss-sum如果小于0，则应减小lr
        # （2）如果loss和为负或最大值为负（全部为负），不能增大lr
        # 也可以根据p值来更新
        if loss_shrink + self.loss_eps < loss_original < loss_enlarge - self.loss_eps:
            if loss_shrink > 0:
                self._shrinked_lr = self._original_lr
                self._original_lr = self._enlarged_lr
                self._enlarged_lr = self.enlarge_ratio * self._enlarged_lr
            elif loss_enlarge < 0:
                # 所有loss都在上涨
                self._enlarged_lr *= self.shrink_ratio
                self._original_lr *= self.shrink_ratio
                self._shrinked_lr *= self.shrink_ratio
                pass
            print('**enlarge')
        elif loss_enlarge + self.loss_eps < loss_original < loss_shrink - self.loss_eps:
            if loss_enlarge > 0:
                self._enlarged_lr = self._original_lr
                self._original_lr = self._shrinked_lr
                self._shrinked_lr = self.shrink_ratio * self._shrinked_lr
            elif loss_shrink < 0:
                self._enlarged_lr *= self.shrink_ratio
                self._original_lr *= self.shrink_ratio
                self._shrinked_lr *= self.shrink_ratio
                pass
            print('**shrink')
        else:
            print('**keep same')
            pass
    
    def step(self):
        pass
