# SP, width d = 512, depth L = 16 (d/L = 32).
# Full run: 5805 steps = 1.52B tokens. With config/depth/tune.py: 10% of that.
n_embd = 512
n_layer = 16
batch_size = 16  # micro-batch, sized for an A100-40GB. if it runs out of memory, halve it
exec(open("config/depth/common.py").read())
