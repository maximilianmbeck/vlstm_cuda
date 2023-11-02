#pragma once

// #include <torch/extension.h>

namespace vlstm
{

    using torch::Tensor;

    namespace interface
    {

        Tensor testkernel(Tensor mat_A);

        Tensor testkernel2(Tensor mat_A);

    } // namespace interface

} // namespace vlstm