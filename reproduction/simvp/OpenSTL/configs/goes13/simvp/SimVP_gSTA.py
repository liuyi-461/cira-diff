# SimVP + gSTA configuration for the EDM GOES-16 ABI channel-13 smoke run.
#
# Task: 2 past frames -> 1 next frame (ordinary 2->1 sample,
# see docs/experiments/cira_diff/EXP-001-cira-diff-data-audit.md).
#
# These are SMOKE settings: the network is deliberately narrowed (hid_S=32,
# N_T=4) and the LR schedule is kept short so that a small subsample trains fast.
# Do NOT quote numbers produced with this config as reproduction results; the
# full-reproduction config should return to the reference values
# (hid_S=64, hid_T=512, N_S=4, N_T=8, see configs/mmnist/simvp/SimVP_gSTA.py).

method = 'SimVP'
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'gSTA'
hid_S = 32
hid_T = 256
N_S = 4
N_T = 4
# training
lr = 1e-3
batch_size = 8
drop_path = 0
sched = 'onecycle'
