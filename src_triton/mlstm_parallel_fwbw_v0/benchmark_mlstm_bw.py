import torch
import triton
from mlstm_bw2 import mlstm_bw
from torch_impl import vlstm_parallel_w_groupnorm_torch_bw

BATCH, N_HEADS, HEAD_DIM = 1, 8, 256
# vary seq length for fixed head and batch=4
configs = []
configs.append(
    triton.testing.Benchmark(
        x_names=["N_CTX"],
        x_vals=[256, 512, 1024, 2048, 4096],  # [2**i for i in range(10, 15)],
        line_arg="provider",
        line_vals=[
            "mlstm_pt_compile",
            "mlstm_triton",
        ],  # ["mlstm_pt", "mlstm_pt_compile", "mlstm_triton"],
        line_names=[
            "mLSTM PT Compile",
            "mLSTM Triton",
        ],  # ["mLSTM PT", "mLSTM PT Compile", "mLSTM Triton"],
        styles=[("red", "-"), ("blue", "-"), ("green", "-")],
        ylabel="ms",
        plot_name=f"fused-mlstm_bw-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}",
        args={
            "H": N_HEADS,
            "BATCH": BATCH,
            "HEAD_DIM": HEAD_DIM,
        },
    )
)


@triton.testing.perf_report(configs)
def bench_flash_mlstm(BATCH, H, N_CTX, HEAD_DIM, provider, device="cuda"):
    warmup = 25
    rep = 100
    dtype = torch.float16
    dH = torch.randn(
        (BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True
    )
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
    n = torch.randn((BATCH, H, N_CTX), dtype=dtype, device=device, requires_grad=True)
    m = torch.randn((BATCH, H, N_CTX), dtype=dtype, device=device, requires_grad=True)
    if "triton" in provider:
        fn = lambda: mlstm_bw(dH, q, k, v, ig, fg, m, n)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

    if "pt" in provider:
        if "compile" in provider:
            mlstm_pt = torch.compile(vlstm_parallel_w_groupnorm_torch_bw)
        else:
            mlstm_pt = vlstm_parallel_w_groupnorm_torch_bw
        fn = lambda: mlstm_pt(
            dH,
            q,
            k,
            v,
            ig.unsqueeze(-1),
            fg.unsqueeze(-1),
            m.unsqueeze(-1),
            n.unsqueeze(-1),
        )
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    total_flops = 2 * flops_per_matmul
    return ms  # total_flops / ms * 1e-9


if __name__ == "__main__":
    # only works on post-Ampere GPUs right now
    bench_flash_mlstm.run(save_path=".", print_data=True)
