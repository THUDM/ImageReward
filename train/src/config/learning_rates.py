'''
@File       :   learning_rates.py
@Time       :   2023/02/09 20:47:00
@Auther     :   Jiazheng Xu
@Contact    :   xjz22@mails.tsinghua.edu.cn
@Description:   learning_rates.

* Based on https://github.com/THUDM/GLM/blob/main/learning_rates.py
'''
"""PyTorch DataLoader for TFRecords"""

import torch
from torch.optim.lr_scheduler import _LRScheduler
import math

def get_learning_rate_scheduler(optimizer, args):
    """Build the learning rate scheduler."""

    # Add linear learning rate scheduler.
    if args.lr_decay_iters is not None:
        num_iters = args.lr_decay_iters
    else:
        num_iters = args.train_iters // args.accumulation_steps
    
    num_iters = max(1, num_iters)
    init_step = -1
    warmup_iter = args.warmup * num_iters
    lr_scheduler = AnnealingLR(optimizer,
                               start_lr=args.lr,
                               warmup_iter=warmup_iter,
                               num_iters=num_iters - warmup_iter,
                               decay_style=args.lr_decay_style,
                               last_iter=init_step,
                               decay_ratio=args.lr_decay_ratio)

    return lr_scheduler

class AnnealingLR(_LRScheduler):
    """Anneals the learning rate from start to zero along a cosine curve."""

    DECAY_STYLES = ['linear', 'cosine', 'exponential', 'constant', "inverse_square_root", 'None']

    def __init__(self, optimizer, start_lr, warmup_iter, num_iters, decay_style=None, last_iter=-1, decay_ratio=0.5):
        assert warmup_iter <= num_iters
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.warmup_iter = warmup_iter
        self.num_iters = last_iter + 1
        self.end_iter = num_iters
        self.decay_style = decay_style.lower() if isinstance(decay_style, str) else None
        self.decay_ratio = decay_ratio
        self.step(self.num_iters)
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print(f'learning rate decaying style {self.decay_style}, ratio {self.decay_ratio}')

    def get_lr(self):
        # https://openreview.net/pdf?id=BJYwwY9ll pg. 4
        if self.decay_style == "inverse_square_root":
            return self.start_lr * math.sqrt(self.warmup_iter) / math.sqrt(max(self.warmup_iter, self.num_iters))
        elif self.decay_style == "constant":
            return self.start_lr
        else:
            if self.warmup_iter > 0 and self.num_iters <= self.warmup_iter:
                return float(self.start_lr) * self.num_iters / self.warmup_iter
            else:
                if self.decay_style == "linear":
                    decay_step_ratio = min(1.0, (self.num_iters - self.warmup_iter) / self.end_iter)
                    return self.start_lr - self.start_lr * (1 - self.decay_ratio) * decay_step_ratio
                elif self.decay_style == "cosine":
                    decay_step_ratio = min(1.0, (self.num_iters - self.warmup_iter) / self.end_iter)
                    return self.start_lr * (
                            (math.cos(math.pi * decay_step_ratio) + 1) / 2 * (1 - self.decay_ratio) + self.decay_ratio)
                elif self.decay_style == "exponential":
                    # TODO: implement exponential decay
                    raise NotImplementedError
                else:
                    raise NotImplementedError

    def step(self, step_num=None):
        if step_num is None:
            step_num = self.num_iters + 1
        self.num_iters = step_num
        new_lr = self.get_lr()
        for group in self.optimizer.param_groups:
            group['lr'] = new_lr

    def state_dict(self):
        sd = {
            # 'start_lr': self.start_lr,
            'warmup_iter': self.warmup_iter,
            'num_iters': self.num_iters,
            'decay_style': self.decay_style,
            'end_iter': self.end_iter,
            'decay_ratio': self.decay_ratio
        }
        return sd

    def load_state_dict(self, sd):
        # self.start_lr = sd['start_lr']
        self.warmup_iter = sd['warmup_iter']
        self.num_iters = sd['num_iters']
        self.end_iter = sd['end_iter']
        self.decay_style = sd['decay_style']
        if 'decay_ratio' in sd:
            self.decay_ratio = sd['decay_ratio']
        self.step(self.num_iters)

    def switch_linear(self, args):
        current_lr = self.get_lr()
        self.start_lr = current_lr
        self.end_iter = args.epochs - self.num_iters
        self.num_iters = 0
        self.decay_style = "linear"
