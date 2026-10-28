# CompleteP (alpha = 1) diagnostic shape, tuning only (not one of the 5 study shapes, never a final run).
# Width d = 512 like 512x16, depth L = 4 like 1024x4, so 512x16 -> 512x4 changes only depth and
# 512x4 -> 1024x4 only width: tells whether the 1024x4 LR drift comes from shallow depth or from
# width / the tied embedding (CLAUDE.md "Sweeps"). Run with: jobs/depth/tune.sh completep 512x4
n_embd = 512
n_layer = 4
batch_size = 16  # micro-batch
exec(open("config/depth/common.py").read())
exec(open("config/depth/completep_common.py").read())
# common.py sizes the run from the FLOP budget, which would give this 12.6M model ~3.3B tokens.
# use 512x16's token count instead (5,805 steps; config/depth/tune.py then takes 10%), so only the shape differs
max_iters = round(_budget_flops / _flops_per_token(512, 16) / (128 * block_size))
warmup_iters = round(0.1 * max_iters)
lr_decay_iters = max_iters
