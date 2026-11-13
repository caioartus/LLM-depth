# CompleteP (alpha = 1), width d = 256, depth L = 64 (d/L = 4).
# Full run: 5571 steps = 1.46B tokens. With config/depth/tune.py: 10% of that.
n_embd = 256
n_layer = 64
batch_size = 8  # micro-batch, sized for an A100-40GB. if it runs out of memory, halve it
exec(open("config/depth/common.py").read())
exec(open("config/depth/completep_common.py").read())
