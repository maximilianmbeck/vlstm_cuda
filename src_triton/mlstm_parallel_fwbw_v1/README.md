# mLSTM parallel fwbw triton kernels v1

This folder contains the first production ready triton kernels of the mLSTM parallel formulation.

TODOs until production readiness:

- [x] align naming of pure pytorch implementation.
- [x] Add a torch autograd function for triton kernels. Check numerical errors.
- [x] Add fwbw speed checks compare to pytorch.
- [ ] Add bounds checking to support arbitrary sequence length and head dimension
- [ ] Support float16 and bfloat16 dtypes. Check numerical errors.
- [ ] enable autmatic mixed precision see xLSTM repo for sLSTM kernels

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
