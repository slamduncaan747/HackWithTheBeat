import os
import time
import math
import pickle
import json
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

out_dir = "out-midi"
eval_interval = 250
log_interval = 50
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = "scratch"
wandb_log = True
wandb_project = "nanogpt-midi"
wandb_run_name = "bach-classical-150M"
dataset = "midi"
gradient_accumulation_steps = 4
batch_size = 32
block_size = 1024
use_data_augmentation = True
augmentation_transposes = [-5, -3, 2, 4]

n_layer = 16
n_head = 16
n_embd = 512
dropout = 0.3
bias = False

label_smoothing = 0.1
explosion_threshold = 100.0
learning_rate = 3e-4
max_iters = 5000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

decay_lr = True
warmup_iters = 500
lr_decay_iters = 4000
min_lr = 3e-5


MAX_MODELS = 3
checkpoint_dir = None

backend = "nccl"

device = "cuda"
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)
compile = True
# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("configurator.py").read())
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------


def transpose_tokens(tokens, semitones):
    new_tokens = []
    for token in tokens:
        if token.startswith("NOTE_ON_") or token.startswith("NOTE_OFF_"):
            parts = token.split("_")
            if len(parts) >= 2 and parts[1].isdigit():
                note = int(parts[1])
                new_note = note + semitones
                if 0 <= new_note <= 127:
                    parts[1] = str(new_note)
                    new_tokens.append("_".join(parts))
                    continue
        new_tokens.append(token)
    return new_tokens


def setup_checkpoint_management():
    try:
        checkpoint_dir = "../drive/MyDrive/music_model_150M"
        os.makedirs(checkpoint_dir, exist_ok=True)

        top_models = []
        tracking_file = os.path.join(checkpoint_dir, "top_models.json")
        if os.path.exists(tracking_file):
            with open(tracking_file, "r") as f:
                top_models = json.load(f)

        return checkpoint_dir, top_models, tracking_file
    except ImportError:
        print("error, colab not detected.")
        return None, [], None


def manage_checkpoints(
    checkpoint, losses, iter_num, checkpoint_dir, top_models, tracking_file
):
    if not checkpoint_dir:
        return top_models

    filename = f"model_iter_{iter_num}_val_{losses['val']:.4f}.pt"
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(checkpoint, filepath)
    print(f"Saved model: {filename}")

    top_models.append((losses["val"], filename))
    top_models.sort()

    if tracking_file:
        with open(tracking_file, "w") as f:
            json.dump(top_models, f)

    if losses["val"] == top_models[0][0]:
        best_path = os.path.join(checkpoint_dir, "best.pt")
        torch.save(checkpoint, best_path)
        print(f"Updated best.pt to iter {iter_num}")

    return top_models


def check_loss_explosion(loss, explosion_threshold):
    if torch.isnan(loss) or loss.item() > explosion_threshold:
        return True
    return False


def log_training_metrics(iter_num, losses, lr, grad_norm, checkpoint_dir):
    print(
        f"iter {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, "
        f"grad norm: {grad_norm:.4f}, lr: {lr:.2e}"
    )

    # Log to file
    if checkpoint_dir:
        with open(os.path.join(checkpoint_dir, "training_log.txt"), "a") as f:
            f.write(
                f"Iter {iter_num}: train {losses['train']:.4f}, val {losses['val']:.4f}, "
                f"grad norm {grad_norm:.4f}, lr {lr:.2e}\n"
            )


# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

# poor man's data loader
data_dir = os.path.join("data", dataset)


def get_batch(split):
    if split == "train":
        data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
    else:
        data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))
            for i in ix
        ]
    )
    if device_type == "cuda":
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y


iter_num = 0
best_val_loss = 1e9

checkpoint_dir, top_models, tracking_file = setup_checkpoint_management()

grad_norm = 0.0

meta_path = os.path.join(data_dir, "meta.pkl")
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    meta_vocab_size = meta["vocab_size"]
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=None,
    dropout=dropout,
)  # start with model_args from command line
if init_from == "scratch":
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print(
            "defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)"
        )
    model_args["vocab_size"] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == "resume":
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]
elif init_from.startswith("gpt2"):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args["block_size"] = (
        block_size  # so that the checkpoint will have the right value
    )
model.to(device)


scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))


optimizer = model.configure_optimizers(
    weight_decay, learning_rate, (beta1, beta2), device_type
)
if init_from == "resume":
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None


if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0


if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def get_lr(it):

    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)

    if it > lr_decay_iters:
        return min_lr

    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


if wandb_log and master_process:
    import wandb

    wandb.init(project=wandb_project, name=wandb_run_name, config=config)


X, Y = get_batch("train")
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0
while True:

    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()

        log_training_metrics(iter_num, losses, lr, grad_norm, checkpoint_dir)

        if wandb_log:
            wandb.log(
                {
                    "iter": iter_num,
                    "train/loss": losses["train"],
                    "val/loss": losses["val"],
                    "lr": lr,
                    "grad_norm": grad_norm,
                    "mfu": running_mfu * 100,
                }
            )

        if iter_num > 0:
            checkpoint = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_args": model_args,
                "iter_num": iter_num,
                "best_val_loss": best_val_loss,
                "config": config,
                "train_loss": losses["train"],
                "val_loss": losses["val"],
                "grad_norm": grad_norm,
            }

            if losses["val"] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses["val"]
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))

            top_models = manage_checkpoints(
                checkpoint, losses, iter_num, checkpoint_dir, top_models, tracking_file
            )
    if iter_num == 0 and eval_only:
        break

    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (
                micro_step == gradient_accumulation_steps - 1
            )
        with ctx:
            logits, loss = model(X, Y)

            if label_smoothing > 0.0:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    Y.view(-1),
                    label_smoothing=label_smoothing,
                    ignore_index=-1,
                )

            if check_loss_explosion(loss, explosion_threshold):
                print(f"skipping batch")
                loss = torch.tensor(0.0, device=device, requires_grad=True)

            loss = loss / gradient_accumulation_steps
        X, Y = get_batch("train")
        scaler.scale(loss).backward()
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:  # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(
            f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%, grad_norm {grad_norm:.4f}"
        )
    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
