# Turns a full run into a tuning run (CLAUDE.md "Tuning protocol"): tune_fraction of the shape's
# tokens with the schedule compressed to that length (still 10% warmup, cosine to 0), same batch.
# Pass it after the shape config:
#   python train.py config/depth/sp/512x16.py config/depth/tune.py --learning_rate=6e-4
tune_fraction = 0.1
max_iters = round(tune_fraction * max_iters)
warmup_iters = round(0.1 * max_iters)
lr_decay_iters = max_iters
always_save_checkpoint = False
never_save_checkpoint = True  # only the final val loss is needed, a crashed tuning run is just rerun
