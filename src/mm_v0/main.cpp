#include <ATen/ATen.h>
#include <iostream>
#include <torch/extension.h>

#include "interface.h"

#define CUDA_DEVICE 0

int main() {
  printf("Creating tensor:\n");
  auto options = torch::TensorOptions()
                     .dtype(torch::kFloat16)
                     //  .dtype(at::BFloat16)
                     .device(torch::kCUDA, CUDA_DEVICE);
  torch::Tensor tensor = torch::rand({2, 2}, options);
  std::cout << tensor << std::endl;

  printf("Calling copykernel:\n");
  torch::Tensor tensor2 = vlstm::interface::copykernel(tensor);
  std::cout << tensor2 << std::endl;

  // printf("Calling testkernel:\n");
  // torch::Tensor tensor3 = vlstm::interface::testkernel(tensor);
  // std::cout << tensor3 << std::endl;

  return 0;
}
