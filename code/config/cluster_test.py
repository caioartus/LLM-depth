# small, quick config to check that training works end to end on a single cluster GPU
# (a few minutes on an A100). uses the openwebtext subset made by data/openwebtext/prepare.py
# $ sbatch jobs/train.sbatch config/cluster_test.py

wandb_log = False
csv_log = True

# 32 batch size * 1024 block size * 1 gradaccum = 32,768 tokens/iter
# 2000 iters ~= 65M tokens, fits in the default 100k-doc subset (~113M tokens)
batch_size = 32
block_size = 1024
gradient_accumulation_steps = 1

# small model
n_layer = 6
n_head = 6
n_embd = 384

learning_rate = 1e-3
max_iters = 2000
lr_decay_iters = 2000
min_lr = 1e-4
warmup_iters = 100

eval_interval = 250
eval_iters = 50
log_interval = 10
always_save_checkpoint = False
