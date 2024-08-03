import torch
import triton
from mlstm_parallel import mlstm_torch_autograd, mlstm_torch_ownbw, mlstm_triton

BATCH, N_HEADS, HEAD_DIM = 1, 8, 256
# vary seq length for fixed head and batch=4
configs = []
configs.append(
    triton.testing.Benchmark(
        x_names=["N_CTX"],
        x_vals=[256, 512, 1024, 2048, 4096],  # [2**i for i in range(10, 15)],
        line_arg="provider",
        line_vals=[
            "mlstm_pt_ag_compile",
            "mlstm_triton",
        ],  # ["mlstm_pt_obw_compile", "mlstm_pt_ag_compile", "mlstm_triton"],
        line_names=[
            "mLSTM PT Autograd Compile",
            "mLSTM Triton",
        ],  # ["mLSTM PT", "mLSTM PT Compile", "mLSTM Triton"],
        styles=[("red", "-"), ("blue", "-"), ("green", "-")],
        ylabel="ms",
        plot_name=f"mlstm_parallel_fwbw-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}",
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
        fn = lambda: mlstm_triton(q, k, v, ig, fg).sum().backward()
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

    if "pt" in provider:
        if "ag" in provider:
            mlstm_pt = mlstm_torch_autograd
        elif "obw" in provider:
            mlstm_pt = mlstm_torch_ownbw
        else:
            raise ValueError(f"Unknown provider {provider}")

        if "compile" in provider:
            mlstm_pt = torch.compile(mlstm_pt)

        fn = (
            lambda: mlstm_pt(
                q,
                k,
                v,
                ig,
                fg,
            )
            .sum()
            .backward()
        )
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    # flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    # total_flops = 2 * flops_per_matmul
    return ms  # total_flops / ms * 1e-9


if __name__ == "__main__":
    bench_flash_mlstm_fwbw.run(save_path=".", print_data=True)
