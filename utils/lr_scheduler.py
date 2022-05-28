from paddle.optimizer.lr import LRScheduler
import math


class LR_Scheduler(LRScheduler):
    """Learning Rate Scheduler

    Step mode: ``lr = baselr * 0.1 ^ {floor(epoch-1 / lr_step)}``

    Cosine mode: ``lr = baselr * 0.5 * (1 + cos(iter/maxiter))``

    Poly mode: ``lr = baselr * (1 - iter/maxiter) ^ 0.9``

    Args:
        args:
          :attr:`args.lr_scheduler` lr scheduler mode (`cos`, `poly`),
          :attr:`args.lr` base learning rate, :attr:`args.epochs` number of epochs,
          :attr:`args.lr_step`

        iters_per_epoch: number of iterations per epoch
    """

    def __init__(
        self,
        mode,
        base_lr,
        num_epochs,
        iters_per_epoch=0,
        lr_step=0,
        warmup_epochs=0
    ):
        self.mode = mode
        print('Using {} LR Scheduler!'.format(self.mode))
        if mode == 'step':
            assert lr_step
        self.lr_step = lr_step
        self.iters_per_epoch = iters_per_epoch
        self.N = num_epochs * iters_per_epoch
        self.warmup_iters = warmup_epochs * iters_per_epoch
        if mode == 'step':
            self.scale_fn = self.step_scale
        elif mode == 'poly':
            self.scale_fn = self.poly_scale
        elif mode == 'cos':
            self.scale_fn = self.cos_scale
        else:
            raise NotImplementedError(
                'Unknown LR scheduler: {}'.format(self.mode))
        super().__init__(base_lr)

    def step_scale(self, epoch):
        return self.base_lr * (0.1 ** (epoch // self.lr_step))

    def cos_scale(self):
        T = self.last_epoch
        return 0.5 * self.base_lr * (1 + math.cos(1.0 * T / self.N * math.pi))

    def poly_scale(self):
        T = self.last_epoch
        return 0.5 * self.base_lr * (1 - 1.0 * T / self.N) ** 0.9

    def get_lr(self):
        lr = self.scale_fn()
        if self.warmup_iters > 0 and self.last_epoch < self.warmup_iters:
            lr = lr * 1.0 * self.last_epoch / self.warmup_iters
        return lr
