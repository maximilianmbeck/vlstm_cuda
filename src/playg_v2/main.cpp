# include <torch/torch.h>
# include <iostream>
# include "interface.h"

#define CUDA_DEVICE 5

int main() {
    printf("Creating tensor:\n");
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA, CUDA_DEVICE);
    torch::Tensor tensor = torch::rand({2, 3}, options);
    std::cout << tensor << std::endl;

    printf("Calling copykernel:\n");
    torch::Tensor tensor2 = vlstm::interface::copykernel(tensor);
    std::cout << tensor2 << std::endl;

    return 0;
}
