# CompleteP (alpha = 1), width d = 1024, depth L = 4 (d/L = 256).
# Full run: 4890 steps = 1.28B tokens. With config/depth/tune.py: 10% of that.
n_embd = 1024
n_layer = 4
batch_size = 16  # micro-batch, sized for an A100-40GB. if it runs out of memory, halve it
exec(open("config/depth/common.py").read())
exec(open("config/depth/completep_common.py").read())
