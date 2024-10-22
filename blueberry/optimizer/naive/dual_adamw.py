import torch
from torch.optim import Optimizer

class DualAdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=((0.8, 0.999), (0.95, 0.999)),
                 weight_decay=0.01, epsilon=1e-8, weights=(1.0, 1.0)):
        """
        :param params: 模型参数
        :param lr: 学习率
        :param betas: (大动量动量， 小动量动量)
        :param weight_decay: 权重衰减
        :param epsilon: 防止除零的常量
        :param weights: (大动量权重，小动量权重)
        """
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, epsilon=epsilon)
        super().__init__(params, defaults)

        self.weights = weights  # (大动量权重, 小动量权重)

        self.state = {}

        for param in params:
            self.state[param] = {
                'step': 0,
                'm_big': torch.zeros_like(param.data),  # 大动量
                'm_small': torch.zeros_like(param.data),  # 小动量
                'v_big': torch.zeros_like(param.data),  # 大动量的二次矩
                'v_small': torch.zeros_like(param.data)  # 小动量的二次矩
            }

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                if p not in self.state:
                    self.state[p] = {
                        'step': 0,
                        'm_big': torch.zeros_like(p.data),
                        'm_small': torch.zeros_like(p.data),
                        'v_big': torch.zeros_like(p.data),
                        'v_small': torch.zeros_like(p.data)
                    }

                grad = p.grad.data
                state = self.state[p]

                state['step'] += 1

                # 更新大动量和小动量
                m_big, m_small = state['m_big'], state['m_small']
                v_big, v_small = state['v_big'], state['v_small']

                # 大动量更新
                beta1_big, beta2_big = group['betas'][0]
                m_big.mul_(beta1_big).add_(grad, alpha=1 - beta1_big)

                # 小动量更新
                beta1_small, beta2_small = group['betas'][1]
                m_small.mul_(beta1_small).add_(grad, alpha=1 - beta1_small)

                # 二次矩更新
                v_big.mul_(beta2_big).addcmul_(grad, grad, value=1 - beta2_big)
                v_small.mul_(beta2_small).addcmul_(grad, grad, value=1 - beta2_small)

                # 计算偏差修正
                m_big_hat = m_big / (1 - beta1_big ** state['step'])
                m_small_hat = m_small / (1 - beta1_small ** state['step'])
                v_big_hat = v_big / (1 - beta2_big ** state['step'])
                v_small_hat = v_small / (1 - beta2_small ** state['step'])

                # 对二阶矩估计值开方并加上 epsilon
                v_big_hat_sqrt = v_big_hat.sqrt().add(group['epsilon'])
                v_small_hat_sqrt = v_small_hat.sqrt().add(group['epsilon'])

                # 计算更新
                update_big = m_big_hat / v_big_hat_sqrt
                update_small = m_small_hat / v_small_hat_sqrt

                update = (update_big * self.weights[0] + update_small * self.weights[1])
                p.data.add_(update, alpha=-group['lr'])

                # 应用权重衰减
                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['weight_decay'] * group['lr'])

        return self

# 示例用法
if __name__ == "__main__":
    model = torch.nn.Linear(10, 1)
    optimizer = DualAdamW(model.parameters(), lr=1e-3, weight_decay=0.01, weights=(0.8, 0.2))

    # 模拟训练过程
    for epoch in range(100):
        inputs = torch.randn(32, 10)
        targets = torch.randn(32, 1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = torch.nn.functional.mse_loss(outputs, targets)
        loss.backward()
        print(f'{epoch=}, {loss.item()=}')
        optimizer.step()
