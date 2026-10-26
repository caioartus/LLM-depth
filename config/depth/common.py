# Shared settings for every run of the depth/width study (the full spec is in CLAUDE.md).
# A shape config (config/depth/{sp,completep}/<d>x<L>.py) sets n_embd, n_layer and batch_size,
# then execs this file, which fills in everything else.
#
# learning_rate is what the sweeps tune, so it is never set in a config file. It always comes
# from the command line (the scripts in jobs/depth/ do this):
#   python train.py config/depth/sp/512x16.py --learning_rate=6e-4                        # full run
#   python train.py config/depth/sp/512x16.py config/depth/tune.py --learning_rate=6e-4  # tuning run
learning_rate = -1.0  # placeholder, train.py refuses to start until --learning_rate=... is given

# data: SlimPajama with the GPT-2 tokenizer. there is no meta.pkl, so train.py pads the vocab to 50,304
dataset = "slimpajama"
block_size = 2048
# read train.bin once, front to back: no token is seen twice, and every run sees the same tokens
# in the same order. needs train.bin >= the longest run (384x28: 1.56B tokens)
sequential_train_data = True

# model: pre-LN, tied embeddings and RoPE are built into model.py. d_head = 64, FFN width 4d (model.py)
n_head = n_embd // 64
dropout = 0.0
bias = False
init_std = 0.02

# batch: 128 sequences x 2048 tokens = 262,144 tokens per step for every shape and condition.
# batch_size is the micro-batch (chosen per shape for memory), gradient accumulation makes up the rest.
# the shape files size it for a 40GB A100; on a GPU with >= 75GB (A100-80, H100-96) it is doubled.
# only the effective batch matters, so this changes speed, not results
if (not globals().get("_keep_micro_batch") and torch.cuda.is_available()
        and torch.cuda.get_device_properties(0).total_memory >= 75e9 and 128 % (2 * batch_size) == 0):
    batch_size *= 2
assert 128 % batch_size == 0
gradient_accumulation_steps = 128 // batch_size

# optimizer, taken from CompleteP and identical for SP and CompleteP.
# (CompleteP scales adam_eps per parameter group on top of this, see config/depth/completep_common.py)
weight_decay = 0.0
beta1 = 0.9
beta2 = 0.95
adam_eps = 1e-16
grad_clip = 1.0

# run length: every shape gets the same compute C = D_ref x C_tok(512x16) ~= 8.48e17 FLOPs, where
# D_ref = 20 tokens per total parameter of 512x16. tokens for this shape: D = C / C_tok(d, L)
#   C_tok = 6 P_ne + 6 d V + 6 L T d,  P_ne = 12 d^2 L,  P_total = P_ne + d V
_V = 50304
def _flops_per_token(d, L):
    return 6 * 12 * d * d * L + 6 * d * _V + 6 * L * block_size * d
_budget_flops = 20 * (12 * 512 * 512 * 16 + 512 * _V) * _flops_per_token(512, 16)
max_iters = round(_budget_flops / _flops_per_token(n_embd, n_layer) / (128 * block_size))

# schedule: linear warmup over 10% of the steps, then cosine decay to 0 at the last step
decay_lr = True
warmup_iters = round(0.1 * max_iters)
lr_decay_iters = max_iters
min_lr = 0.0

# precision: bf16 autocast, fp32 master weights and optimizer states
dtype = "bfloat16"
compile = True

# eval and logging. the val loss after the last step is always logged (train.py)
eval_interval = 250
eval_iters = 1600 // batch_size  # 1,600 fixed sequences (3.3M tokens) per split per eval, identical for every shape
log_interval = 10
run_log = True  # out_dir/train.csv (every 10 steps) and out_dir/eval.csv (every eval), see train.py
csv_log = False  # the old log.csv, replaced by run_log
wandb_log = False
wandb_project = "llm-depth"
# checkpoint at every eval (every 250 steps, ~1 GB, overwritten each time), so a crashed run
# can resume: resubmitting the same run name continues from it (jobs/train.sbatch)
always_save_checkpoint = True

seed = 0
