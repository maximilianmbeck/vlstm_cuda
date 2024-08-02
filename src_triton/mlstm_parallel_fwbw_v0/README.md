# mLSTM parallel fwbw triton kernels v0

This folder contains the first version of the flash-attention like mLSTM kernel implementations for
the forward and backward pass.

We implement the purely parallel formulation.

Preliminary tuning on my laptop shows that this version in general is much faster  for smaller head dimensions.

Forward: Is about a factor of 2-3 faster for head dims <= 256

fused-mlstm-batch1-head8-d256:
    N_CTX   mLSTM PT  mLSTM PT Compile  mLSTM Triton
0   256.0   0.163215          0.077844      0.088321
1   512.0   0.354266          0.224933      0.173710
2  1024.0   1.578989          0.812083      0.456228
3  2048.0   7.949301          3.041348      1.928133
4  4096.0  31.774666         15.842171      6.356939

fused-mlstm-batch1-head8-d512:
    N_CTX   mLSTM PT  mLSTM PT Compile  mLSTM Triton
0   256.0   0.214969          0.126638      0.120540
1   512.0   0.471108          0.326504      0.454885
2  1024.0   1.758080          1.427305      1.357088
3  2048.0  10.854680          5.236983      4.066498
4  4096.0  34.558975         20.163006     15.434957

Backward: Is about a factor of 2 faster for head dims <= 128

fused-mlstm_bw-batch1-head8-d128:
    N_CTX  mLSTM PT Compile  mLSTM Triton
0   256.0          0.126860      0.126767
1   512.0          0.338630      0.248242
2  1024.0          1.324440      0.669022
3  2048.0          5.356736      2.381943
4  4096.0         25.103380      8.767787

fused-mlstm_bw-batch1-head8-d256:
    N_CTX  mLSTM PT Compile  mLSTM Triton
0   256.0          0.156252      0.232461
1   512.0          0.445820      0.607455
2  1024.0          1.774721      2.103677
3  2048.0          6.880968      7.720346
4  4096.0         31.294100     28.854507
