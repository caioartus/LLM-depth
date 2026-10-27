# Tuned learning rates

Written by `analysis/collect_results.py` from the tuning runs (10% of each shape's tokens).
Optimum = lowest val loss after the last step (CLAUDE.md "Tuning protocol").

## completep 256x64

| LR | val loss | |
|---|---|---|
| 0.0009766 | 5.7827 |  |
| 0.001953 | 5.7533 |  |
| 0.003906 | 5.6663 |  |
| 0.007812 | 5.5612 | **best** |
| 0.01562 | 5.9570 |  |

## completep 512x16

| LR | val loss | |
|---|---|---|
| 0.0009766 | 5.6913 |  |
| 0.001953 | 5.5105 |  |
| 0.003906 | 5.4523 |  |
| 0.007812 | 5.4300 | **best** |
| 0.01562 | 5.8776 |  |

## completep 1024x4

| LR | val loss | |
|---|---|---|
| 0.0009766 | 5.7199 |  |
| 0.001953 | 5.4923 | **best** |
| 0.003906 | 5.6820 |  |
| 0.007812 | 5.6761 |  |
| 0.01562 | 5.7153 |  |

## sp 256x64

| LR | val loss | |
|---|---|---|
| 0.00015 | 6.6052 |  |
| 0.0003 | 6.2994 |  |
| 0.0006 | 6.0605 | **best** |
| 0.0012 | 6.1883 |  |
| 0.0024 | 6.2557 |  |

## sp 384x28

| LR | val loss | |
|---|---|---|
| 0.00015 | 6.1966 |  |
| 0.0003 | 5.9207 |  |
| 0.0006 | 5.7090 | **best** |
| 0.0012 | 5.8325 |  |
| 0.0024 | 5.8058 |  |

## sp 512x16

| LR | val loss | |
|---|---|---|
| 0.00015 | 5.9731 |  |
| 0.0003 | 5.7013 |  |
| 0.0006 | 5.5305 |  |
| 0.0012 | 5.5221 | **best** |
| 0.0024 | 5.7489 |  |

## sp 640x10

| LR | val loss | |
|---|---|---|
| 0.00015 | 5.7997 |  |
| 0.0003 | 5.5482 |  |
| 0.0006 | 5.3933 |  |
| 0.0012 | 5.3864 | **best** |
| 0.0024 | 5.5486 |  |

## sp 1024x4

| LR | val loss | |
|---|---|---|
| 0.00015 | 5.6723 |  |
| 0.0003 | 5.4453 |  |
| 0.0006 | 5.3972 |  |
| 0.0012 | 5.3963 | **best** |
| 0.0024 | 5.5421 |  |

## CompleteP transfer check

Optimum at the depth extremes vs 512x16 (more than one grid step = flag, CLAUDE.md "Sweeps").

- 256x64: +0 grid steps (ok)
- 1024x4: -2 grid steps ⚠ **FLAG: transfer failed or implementation bug**
