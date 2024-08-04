# mLSTM parallel fwbw triton kernels v1

This folder contains the first production ready triton kernels of the mLSTM parallel formulation.

TODOs until production readiness:

- [x] align naming of pure pytorch implementation.
- [x] Add a torch autograd function for triton kernels. Check numerical errors.
- [x] Add fwbw speed checks compare to pytorch.
- [x] Add bounds checking to support arbitrary sequence length and head dimension
  - we cannot do bounds checking this results in an error.
  - [ ]: follow up support arbitrary head dimensions.
- [x] Support float16 and bfloat16 dtypes. Check numerical errors.
- [ ] enable autmatic mixed precision see xLSTM repo for sLSTM kernels

## Open points

### Arbitrary head dimensions

Currently we only support head dimensions of power of 2.
Reason for this is that tl.zeros([BQ, DH]) only supports sizes of power of 2.
We could set it manually and pad with zeros, but this proably results in inefficiency.

### Enable automatic mixed precision

Somehow torch.cuda.amp.custom_fwd decorator is deprecated.
So leave that out for now.
Probably go to: Make a config and manually cast inputs to configured dtype.
We probably want to have float16 instead of bfloat16.

```
S=256, DH=64, Baselines in float32

Errors with vecF_cs in float16:
====== Triton -> PT Autograd ======
hs match: False, max diff: 0.19851431250572205
dQ match: False, max diff: 3.22531795501709
dK match: False, max diff: 3.1646175384521484
dV match: False, max diff: 1.4900684356689453
dI match: False, max diff: 6.1808013916015625
dF match: False, max diff: 3.6602001190185547
 ====== Triton -> PT Own backward ======
hs match: False, max diff: 0.19851431250572205
dQ match: False, max diff: 3.22531795501709
dK match: False, max diff: 3.16463565826416
dV match: False, max diff: 1.490067481994629
dI match: False, max diff: 6.180503845214844
dF match: False, max diff: 3.6602210998535156

Errors with vecF_cs in float32:
====== Triton -> PT Autograd ======
hs match: False, max diff: 0.008412718772888184
dQ match: False, max diff: 0.30786895751953125
dK match: False, max diff: 0.24111175537109375
dV match: False, max diff: 0.03338479995727539
dI match: False, max diff: 0.17911529541015625
dF match: False, max diff: 0.10850143432617188
 ====== Triton -> PT Own backward ======
hs match: False, max diff: 0.008412718772888184
dQ match: False, max diff: 0.30786895751953125
dK match: False, max diff: 0.2411041259765625
dV match: False, max diff: 0.033383846282958984
dI match: False, max diff: 0.1794586181640625
dF match: False, max diff: 0.10846734046936035
```

**Result**: The fgate cumsum plays a crucial role in numerical errors (as expected). Keep this in float32.

## Benchmark PyTorch compile vs. Triton

(on my laptop RTX 4060)

**Metric in milliseconds**

mlstm_parallel_fwbw_fw-batch1-head8-d64:
    N_CTX  mLSTM PT Autograd Compile FWBW  mLSTM Triton FWBW  mLSTM PT Autograd Compile FW  mLSTM Triton FW
0   256.0                        0.118374           0.519061                      0.040797         0.030254
1   512.0                        0.388854           0.277450                      0.131670         0.054715
2  1024.0                        1.378533           0.598020                      0.375019         0.124451
3  2048.0                        6.281271           1.618509                      1.598742         0.355323
4  4096.0                       32.691319           5.502702                      6.633487         1.245179

mlstm_parallel_fwbw_fw-batch1-head8-d128:
    N_CTX  mLSTM PT Autograd Compile FWBW  mLSTM Triton FWBW  mLSTM PT Autograd Compile FW  mLSTM Triton FW
0   256.0                        0.199337           0.207912                      0.057782         0.048669
1   512.0                        0.454639           0.450522                      0.165261         0.082363
2  1024.0                        1.601133           1.077663                      0.497016         0.254517
3  2048.0                        7.497245           3.267866                      1.690359         0.695027
4  4096.0                       33.677311          10.293648                      7.265906         2.473338

mlstm_parallel_fwbw_fw-batch1-head8-d256:
    N_CTX  mLSTM PT Autograd Compile FWBW  mLSTM Triton FWBW  mLSTM PT Autograd Compile FW  mLSTM Triton FW
0   256.0                        0.264804           0.374493                      0.072992         0.064823
1   512.0                        0.578312           0.774771                      0.178977         0.142496
2  1024.0                        2.159089           2.272552                      0.584149         0.425770
3  2048.0                        8.405284           8.191925                      2.285031         1.471647
4  4096.0                       43.880959          29.034836                      9.730020         6.753006
