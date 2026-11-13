# CompleteP coordinate check. The shape and micro-batch come from the command line, *before* this file:
#   python train.py --n_embd=256 --n_layer=8 --batch_size=8 --seed=1 config/depth/completep/coord_check.py
# Run them all with jobs/depth/coord_check.sbatch.
_keep_micro_batch = True  # the micro-batch is the check's batch (no accumulation), so never resize it
exec(open("config/depth/common.py").read())
exec(open("config/depth/completep_common.py").read())
exec(open("config/depth/coord_check_common.py").read())
