# Coordinate check settings (CLAUDE.md "Before sweeps"), execed by config/depth/{sp,completep}/coord_check.py.
# 10 training steps that record the mean |output| of the token embedding, every attention and MLP
# branch, the last block and lm_head at every step. Under a correct CompleteP these stay roughly
# the same as depth (and width) grow; under SP they drift. Run by jobs/depth/coord_check.sbatch.
max_iters = 10
learning_rate = 2**-8  # CompleteP's tuned eta_base, the same for SP so both are checked at one LR
decay_lr = False  # constant LR: the check is about scaling with shape, not the schedule
gradient_accumulation_steps = 1  # batch_size (8) sequences per step is enough here
mup_enable_coord_check_logging = True
eval_interval = 1  # the activation sizes are written to eval.csv at every eval
eval_iters = 1
log_interval = 1
always_save_checkpoint = False
never_save_checkpoint = True
compile = False  # the hooks find layers by module name, which torch.compile changes
