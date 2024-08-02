# mLSTM parallel fwbw triton kernels v1

This folder contains the first production ready triton kernels of the mLSTM parallel formulation.

TODOs:

- [x] align naming of pure pytorch implementation.
- [ ] Add a torch autograd function for triton kernels. Check numerical errors.
- [ ] Support float16 and bfloat16 dtypes. Check numerical errors.
- [ ] Add bounds checking to support arbitrary sequence length and head dimension
- [ ] enable autmatic mixed precision see xLSTM repo for sLSTM kernels
