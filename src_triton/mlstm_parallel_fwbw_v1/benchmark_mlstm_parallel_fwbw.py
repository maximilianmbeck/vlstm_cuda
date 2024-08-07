import torch
import triton
from mlstm_parallel import mlstm_torch_autograd as mlstm_parallel_torch_autograd
from mlstm_parallel import mlstm_torch_ownbw as mlstm_parallel_torch_ownbw
from mlstm_parallel import mlstm_triton as mlstm_parallel_triton

BATCH, N_HEADS = 1, 8

HEAD_DIMS = [64, 128, 256]


configs = []
for HEAD_DIM in HEAD_DIMS:
    configs.append(
        triton.testing.Benchmark(
            x_names=["N_CTX"],
            x_vals=[256, 512, 1024, 2048, 4096],  # [2**i for i in range(10, 15)],
            line_arg="provider",
            line_vals=[
                "mlstm_parallel_pt_ag_compile_fwbw",
                "mlstm_parallel_triton_fwbw",
                "mlstm_parallel_pt_ag_compile_fw",
                "mlstm_parallel_triton_fw",
            ],
            line_names=[
                "mLSTM parallel PT Autograd Compile FWBW",
                "mLSTM parallel Triton FWBW",
                "mLSTM parallel PT Autograd Compile FW",
                "mLSTM parallel Triton FW",
            ],
            styles=[("red", "-"), ("blue", "-"), ("red", "--"), ("blue", "--")],
            ylabel="ms",
            plot_name=f"mlstm_fwbw_fw-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}",
            args={
                "H": N_HEADS,
                "BATCH": BATCH,
                "HEAD_DIM": HEAD_DIM,
            },
        )
    )


@triton.testing.perf_report(configs)
def bench_flash_mlstm_fwbw(BATCH, H, N_CTX, HEAD_DIM, provider, device="cuda"):
    warmup = 25
    rep = 100
    dtype = torch.float16

    q = torch.randn(
        (BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True
    )
    k = torch.randn(
        (BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True
    )
    v = torch.randn(
        (BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True
    )
    ig = torch.randn((BATCH, H, N_CTX), dtype=dtype, device=device, requires_grad=True)
    fg = torch.randn((BATCH, H, N_CTX), dtype=dtype, device=device, requires_grad=True)
    if "triton" in provider:
        if "parallel" in provider:
            fw_fn = lambda: mlstm_parallel_triton(q, k, v, ig, fg)
        else:
            raise ValueError(f"Unknown provider {provider}")

    if "pt" in provider:
        if "parallel" in provider:
            if "ag" in provider:
                mlstm_pt = mlstm_parallel_torch_autograd
            elif "obw" in provider:
                mlstm_pt = mlstm_parallel_torch_ownbw
            else:
                raise ValueError(f"Unknown provider {provider}")

        if "compile" in provider:
            mlstm_pt = torch.compile(mlstm_pt)

        fw_fn = lambda: mlstm_pt(
            q,
            k,
            v,
            ig,
            fg,
        )

    if "fwbw" in provider:
        fn = lambda: fw_fn().sum().backward()
    else:
        fn = fw_fn
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    # flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    # total_flops = 2 * flops_per_matmul
    return ms  # total_flops / ms * 1e-9


if __name__ == "__main__":
    bench_flash_mlstm_fwbw.run(save_path=".", print_data=True)
