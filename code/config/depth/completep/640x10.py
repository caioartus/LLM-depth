# CompleteP (alpha = 1), width d = 640, depth L = 10 (d/L = 64).
# Full run: 5707 steps = 1.50B tokens. With config/depth/tune.py: 10% of that.
n_embd = 640
n_layer = 10
batch_size = 16  # micro-batch, sized for an A100-40GB. if it runs out of memory, halve it
exec(open("config/depth/common.py").read())
exec(open("config/depth/completep_common.py").read())
