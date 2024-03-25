import os
from typing import Dict
import torch
from torch import distributed as dist
import numpy as np
import random
from torch.optim.lr_scheduler import _LRScheduler
from transformers import AutoTokenizer, GPT2Tokenizer
from matplotlib import pyplot as plt
import math
import albumentations as A


def init_distributed_setup(args):
    args.global_rank = int(os.environ['RANK'])
    args.local_rank = int(os.environ['LOCAL_RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(args.local_rank)

    dist.init_process_group(backend='nccl')
    dist.barrier()

    setup_for_distributed(args.global_rank==0)

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)

    
class LinearwarmupCosineScheduler(_LRScheduler):
    def __init__(self, optimizer, total_iterations: int, warmup_ratio: float):
        assert 0 <= warmup_ratio < 1, f"Unexpected warmup ratio. The ratio must be [0, 1]. Current : {warmup_ratio}"

        self.total_iterations = total_iterations
        self.warmup_thre = int(total_iterations * warmup_ratio)
        self.cur_iter = -1
        super(LinearwarmupCosineScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.warmup_thre > self.cur_iter:
            new_lrs = []
            for lr in self.base_lrs:
                new_lrs.append(lr * (self.cur_iter/self.warmup_thre))
            return new_lrs
        else:
            norm_iter = (self.cur_iter - self.warmup_thre) / (self.total_iterations - self.warmup_thre)
            new_lrs = []
            for lr in self.base_lrs:
                new_lr = 0.5 * (lr + lr * math.cos(math.pi * norm_iter))
                new_lrs.append(new_lr)
            return new_lrs
    
    def step(self):
        self.cur_iter += 1
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


def setup_for_distributed(is_master: bool):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def save_ddp_model(model, save_path):
    torch.save(model.module.state_dict(), save_path)
    print(f"The model checkpoint is saved at {save_path}")

def save_tokenizer(tokenizer: AutoTokenizer, save_dir: str):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"The tokenizer is saved at {save_dir}")


def get_transform_train(size:int = 384):
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]

    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Resize(size, size),
        A.Normalize(image_mean, image_std),

    ], bbox_params=A.BboxParams(format='pascal_voc'))


# tknizer = GPT2Tokenizer.from_pretrained('gpt2')
# sp = "/data/cad-recruit-02_814/kilee/NextChat/"
# save_tokenizer(tknizer, sp)
# from torchvision import models

# md = models.resnet18()
# opt = torch.optim.AdamW(params=md.parameters(), lr=1e-2)
# opt_scheduler = LinearwarmupCosineScheduler(optimizer=opt, 
#                                             total_iterations=1000, 
#                                             warmup_ratio=0.1)

# lrs = []
# for i in range(1000):
#     for idx, param_group in enumerate(opt.param_groups):
#         lrs.append(param_group['lr'])
#         if idx > 1:
#             break
#     opt_scheduler.step()

# plt.plot(np.linspace(0, 1000, 1000), lrs)
# plt.savefig("./lr_curve.jpg")
# num_iter = 10000
# warm_rat = 0.1
# iteration_space = np.linspace(0, num_iter, num_iter+1)
# lr = 1e-1
# lrs = []

# for i in range(num_iter+1):
#     lrs.append(lin_warmup_cos(lr, i, num_iter, warm_rat))

# print(len(iteration_space), len(lrs))
# plt.plot(iteration_space, lrs)
# plt.savefig("./lr_curve.jpg")
