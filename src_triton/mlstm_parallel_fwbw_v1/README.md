# mLSTM parallel fwbw triton kernels v1

This folder contains the first production ready triton kernels of the mLSTM parallel formulation.

TODOs until production readiness:

- [x] align naming of pure pytorch implementation.
- [x] Add a torch autograd function for triton kernels. Check numerical errors.
- [x] Add fwbw speed checks compare to pytorch.
- [ ] Support float16 and bfloat16 dtypes. Check numerical errors.
- [ ] Add bounds checking to support arbitrary sequence length and head dimension
- [ ] enable autmatic mixed precision see xLSTM repo for sLSTM kernels

## Benchmark PyTorch compile vs. Triton

(on my laptop RTX 4060)

**Metric in milliseconds**

mlstm_parallel_fwbw-batch1-head8-d64:
    N_CTX  mLSTM PT Autograd Compile  mLSTM Triton
0   256.0                   0.134752      0.137492
1   512.0                   0.386205      0.259211
2  1024.0                   1.451759      0.595682
3  2048.0                   6.033095      1.607347
4  4096.0                  27.402861      5.174607

mlstm_parallel_fwbw-batch1-head8-d128:
    N_CTX  mLSTM PT Autograd Compile  mLSTM Triton
0   256.0                   0.154278      0.213272
1   512.0                   0.427126      0.397416
2  1024.0                   1.611108      1.037117
3  2048.0                   6.424649      3.276074
4  4096.0                  26.911402     10.022568

mlstm_parallel_fwbw-batch1-head8-d256:
    N_CTX  mLSTM PT Autograd Compile  mLSTM Triton
0   256.0                   0.220664      0.307998
1   512.0                   0.582464      0.732306
2  1024.0                   2.213279      2.358819
3  2048.0                   8.997788      7.414556
4  4096.0                  34.885422     27.908865
