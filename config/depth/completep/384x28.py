# CompleteP (alpha = 1), width d = 384, depth L = 28 (d/L = 13).
# Full run: 5932 steps = 1.55B tokens. With config/depth/tune.py: 10% of that.
n_embd = 384
n_layer = 28
batch_size = 8  # micro-batch, sized for an A100-40GB. if it runs out of memory, halve it
exec(open("config/depth/common.py").read())
exec(open("config/depth/completep_common.py").read())
