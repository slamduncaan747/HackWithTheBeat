import os
import json
import torch
import torch.nn as nn
from torch.nn import functional as F
import math


MAX_MODELS = 3

out_dir = "out-midi"
eval_interval = 250
log_interval = 50
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = "scratch"


checkpoint_dir = None
top_models = []
tracking_file = None


wandb_log = True
wandb_project = "nanogpt-midi"
wandb_run_name = "bach-classical"


dataset = "midi"
gradient_accumulation_steps = 4
batch_size = 32
block_size = 2048

use_data_augmentation = True
augmentation_transposes = [-5, -3, 2, 4]


n_layer = 16
n_head = 16
n_embd = 512
dropout = 0.3
bias = False
vocab_size = 3294


label_smoothing = 0.1

learning_rate = 3e-4
max_iters = 5000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0


explosion_threshold = 100.0


decay_lr = True
warmup_iters = 500
lr_decay_iters = 4000
min_lr = 3e-5


patience = 5
best_val_loss = float("inf")
patience_counter = 0


backend = "nccl"

device = "cuda"
dtype = "bfloat16"
compile = True
