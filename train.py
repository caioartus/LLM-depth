"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext
from functools import partial

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
skip_val_loss = False # If True, will only measure train loss
always_save_checkpoint = True # if True, always save a checkpoint after each eval
never_save_checkpoint = False # if True, never save a checkpoint
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# csv logging
csv_log = False # If enabled, logs stats to a csv file
run_log = False # depth study: append train.csv / eval.csv in out_dir row by row (survive crashes, continue on resume)
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
sequential_train_data = False # depth study: if True, read train.bin front to back once instead of random windows
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
init_std = 0.02 # Initialization standard deviation for weights
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
adam_eps = 1e-12
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# mup settings
mup_enabled = False # Whether to use muP. If False then all other mup variables are ignored
mup_disable_attention_scaling = False # Uses 1/sqrt(d_head) attn scaling instead of 1/d_head (Only needed for the step-by-step coord check in the blog)
mup_disable_hidden_lr_scaling = False # Disables muP hidden LR adjustment (Only needed for the step-by-step coord check in the blog)
mup_width_multiplier = 1.0 # mup_width_multiplier = width / base_width where base_width is typically 256
mup_input_alpha = 1.0 # Optional tunable multiplier applied to input embedding forward pass output
mup_output_alpha = 1.0 # Optional tunable multiplier applied to output unembedding forward pass output
mup_enable_coord_check_logging = False # If True will track the output.abs().mean() of various layers throughout training
# Depth scaling settings
depth_alpha_enabled = False 
depth_multiplier = 1.0
depth_alpha_exp = 1.0
# seed
seed = 1337
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

assert not (never_save_checkpoint and always_save_checkpoint)
### Begin depth-study code ###
# config/depth/common.py sets learning_rate = -1.0 so that a run can't silently use a default LR
assert learning_rate > 0, "set the learning rate with --learning_rate=..."
### End depth-study code ###

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
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
torch.manual_seed(seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
train_batches_read = 0 # number of sequential train micro-batches read so far (per process)
def get_batch(split, generator=None):
    global train_batches_read
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ### Begin depth-study code ###
    if split == 'train' and sequential_train_data and generator is None:
        # one pass, no repeats: micro-batch k takes the next batch_size*block_size tokens of train.bin.
        # every run sees the same tokens in the same order (step s always reads tokens
        # [s*tokens_per_iter, (s+1)*tokens_per_iter)), whatever the micro-batch size.
        # (generator is not None for the train-loss estimate in estimate_loss, which stays random)
        k = train_batches_read * ddp_world_size + seed_offset # seed_offset = this process's DDP rank (0 without DDP)
        start = k * batch_size * block_size
        end = start + batch_size * block_size + 1 # +1 for the shifted targets
        assert end <= len(data), f"train.bin has only {len(data):,} tokens, the run needs more. prepare more documents"
        chunk = torch.from_numpy(data[start:end].astype(np.int64))
        x = chunk[:-1].view(batch_size, block_size)
        y = chunk[1:].view(batch_size, block_size)
        train_batches_read += 1
    ### End depth-study code ###
    else:
        ix = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout, mup_enabled=mup_enabled,
                  mup_disable_attention_scaling=mup_disable_attention_scaling,
                  mup_disable_hidden_lr_scaling=mup_disable_hidden_lr_scaling,
                  mup_width_multiplier=mup_width_multiplier, mup_input_alpha=mup_input_alpha,
                  mup_output_alpha=mup_output_alpha,
                  depth_alpha_enabled=depth_alpha_enabled, depth_alpha_exp=depth_alpha_exp, depth_multiplier=depth_multiplier) # start with model_args from command line

if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
    train_batches_read = iter_num * gradient_accumulation_steps # depth study: continue the sequential read where it stopped
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), adam_eps, device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    splits = ['train'] if skip_val_loss else ['train', 'val']
    for split in splits:
        losses = torch.zeros(eval_iters)
        ### Begin depth-study code ###
        # the same fixed windows at every eval and in every run (also across micro-batch sizes),
        # so losses are compared on identical text. own generator: the training batches are unaffected
        eval_generator = torch.Generator().manual_seed(1234)
        ### End depth-study code ###
        for k in range(eval_iters):
            X, Y = get_batch(split, eval_generator)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    if skip_val_loss:
        out['val'] = -1
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if master_process:
    if wandb_log:
        import wandb
        wandb_run = wandb.init(project=wandb_project, name=wandb_run_name, config=config)
    if csv_log:
        from csv_logging import CSVLogWrapper
        def log(log_dict):
            pass
        csv_logger = CSVLogWrapper(log, config=config, out_dir=out_dir)
    ### Begin depth-study code ###
    if run_log:
        import csv, json
        with open(os.path.join(out_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        def open_run_log(filename, columns):
            # out_dir/filename gets one row per call, flushed straight away, so it survives a crash.
            # rows from iter_num on are dropped first: when resuming they were written after the
            # checkpoint and are about to be redone, and a run from scratch (iter_num = 0) starts empty
            path = os.path.join(out_dir, filename)
            old_rows = []
            if os.path.exists(path):
                with open(path, newline='') as f:
                    old_rows = [r for r in csv.DictReader(f) if int(r['iter']) < iter_num]
            f = open(path, 'w', newline='')
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore', lineterminator='\n')
            writer.writeheader()
            writer.writerows(old_rows)
            f.flush()
            def write_row(row):
                writer.writerow(row)
                f.flush()
            return write_row
        coord_check_columns = [k + '_act_abs_mean' for k in ['token_embedding', 'attn', 'mlp', 'lm_head', 'last_layer']]
        # train.csv: every log_interval steps. loss = mean over the step's micro-batches, tokens = seen so far
        write_train_row = open_run_log('train.csv', ['iter', 'tokens', 'loss', 'lr', 'time_ms', 'mfu', 'peak_mem_gb'])
        # eval.csv: every eval, and after the last step. tokens = seen before this eval
        write_eval_row = open_run_log('eval.csv', ['iter', 'tokens', 'train_loss', 'val_loss', 'lr']
                                      + (coord_check_columns if mup_enable_coord_check_logging else []))
    ### End depth-study code ###

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
coord_check_dict = None
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group.get('lr_scale', 1.0)

    # evaluate the loss on train/val sets and write checkpoints
    ### Begin depth-study code ###
    # also evaluate after the last step (iter_num == max_iters), so every run logs its final val loss
    # even when max_iters is not a multiple of eval_interval. tuning picks the LR from this value
    if (iter_num % eval_interval == 0 or iter_num == max_iters) and master_process:
    ### End depth-study code ###
        losses = estimate_loss()
        if np.isnan(losses['train']):
            raise Exception('NaN loss')
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        log_dict = {
            "iter": iter_num,
            "train/loss": losses['train'],
            "val/loss": losses['val'],
            "lr": lr,
            "mfu": running_mfu*100, # convert to percentage
        }
        if mup_enable_coord_check_logging and coord_check_dict is not None:
            for key in coord_check_dict:
                log_dict[key + '_act_abs_mean'] = np.mean(coord_check_dict[key])
        if wandb_log:
            wandb_run.log(log_dict)
        if csv_log:
            csv_logger.log(log_dict)
            csv_logger.step()
        ### Begin depth-study code ###
        if run_log:
            write_eval_row({'iter': iter_num, 'tokens': iter_num * tokens_per_iter, 'train_loss': losses['train'],
                            'val_loss': losses['val'], 'lr': lr,
                            **{k: v for k, v in log_dict.items() if k.endswith('_act_abs_mean')}})
        ### End depth-study code ###
        if (not never_save_checkpoint) and (losses['val'] < best_val_loss or always_save_checkpoint):
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                ### Begin depth-study code ###
                # write to a temporary file, then rename: a crash during saving can't corrupt the last good checkpoint
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt.tmp'))
                os.replace(os.path.join(out_dir, 'ckpt.pt.tmp'), os.path.join(out_dir, 'ckpt.pt'))
                ### End depth-study code ###
    if iter_num == 0 and eval_only:
        break
    ### Begin depth-study code ###
    # stop right after the final eval, so a run trains exactly max_iters steps (iters 0 .. max_iters - 1).
    # (before, it also took an extra step at iter max_iters, which with min_lr = 0 only wasted tokens)
    if iter_num == max_iters:
        break
    ### End depth-study code ###

    if mup_enable_coord_check_logging:
        coord_check_dict = {
            'token_embedding': [],
            'attn': [],
            'mlp': [],
            'lm_head': [],
            'last_layer': []
        }
        def hook(module, input, output, key):
            with torch.no_grad():
                coord_check_dict[key].append(output.abs().mean().item())
        coord_check_handles = []
        for module_name, module in model.named_modules():
            if module_name == 'transformer.wte':
                coord_check_handles.append(module.register_forward_hook(partial(hook, key='token_embedding')))
            elif module_name.endswith('.attn'):
                coord_check_handles.append(module.register_forward_hook(partial(hook, key='attn')))
            elif module_name.endswith('.mlp'):
                coord_check_handles.append(module.register_forward_hook(partial(hook, key='mlp')))
            elif module_name == 'lm_head':
                coord_check_handles.append(module.register_forward_hook(partial(hook, key='lm_head')))
            elif module_name == f'transformer.h.{n_layer - 1}':
                coord_check_handles.append(module.register_forward_hook(partial(hook, key='last_layer')))
    else:
        coord_check_dict = None

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    step_loss = torch.zeros((), device=device) # depth study: sum of loss / grad_accum = mean loss of this step
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        step_loss += loss.detach()
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        ### Begin depth-study code ###
        # mean over all micro-batches of the step (was: the last micro-batch only)
        lossf = step_loss.item()
        peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9 if device_type == 'cuda' else 0.0
        ### End depth-study code ###
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%, peak mem {peak_mem_gb:.1f}GB")
        ### Begin depth-study code ###
        if run_log:
            write_train_row({'iter': iter_num, 'tokens': (iter_num + 1) * tokens_per_iter, 'loss': lossf, 'lr': lr,
                             'time_ms': dt * 1000, 'mfu': running_mfu * 100, 'peak_mem_gb': peak_mem_gb})
        ### End depth-study code ###
    iter_num += 1
    local_iter_num += 1

    if mup_enable_coord_check_logging:
        for handle in coord_check_handles:
            handle.remove()

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
