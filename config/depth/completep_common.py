# CompleteP (alpha = 1) settings, execed by every config in config/depth/completep/ after common.py.
# Multipliers are relative to CompleteP's base shape N_base = 256, L_base = 2 (CLAUDE.md, CompleteP Table 1),
# and --learning_rate is eta_base, the LR of that base shape.
mup_enabled = True  # width (muP): hidden init variance and hidden LR / m_N, logits / m_N, attention 1/d_head
mup_width_multiplier = n_embd / 256  # m_N
depth_alpha_enabled = True  # depth: residual branches x m_L^-alpha, hidden LR x m_L^(alpha - 1) = 1 at alpha = 1
depth_multiplier = n_layer / 2  # m_L
depth_alpha_exp = 1.0  # alpha
# adam_eps = 1e-16 from common.py is then scaled per group in model.configure_optimizers:
# hidden eps / (m_N m_L^alpha), embedding and final LN eps / m_N
