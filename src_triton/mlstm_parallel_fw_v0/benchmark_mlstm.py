import torch
import triton
from mlstm import mlstm_fw
from torch_impl import vlstm_fw_torch_ref

BATCH, N_HEADS, HEAD_DIM = 2, 8, 128
# vary seq length for fixed head and batch=4
configs = []
configs.append(
    triton.testing.Benchmark(
        x_names=["N_CTX"],
        x_vals=[256, 512, 1024, 2048, 4096],  # [2**i for i in range(10, 15)],
        line_arg="provider",
        line_vals=["mlstm_pt", "mlstm_pt_compile", "mlstm_triton"],
        line_names=["mLSTM PT", "mLSTM PT Compile", "mLSTM Triton"],
        styles=[("red", "-"), ("blue", "-"), ("green", "-")],
        ylabel="ms",
        plot_name=f"fused-mlstm-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}",
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
        fn = lambda: mlstm_fw(q, k, v, ig, fg)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

    if "pt" in provider:
        if "compile" in provider:
            mlstm_pt = torch.compile(vlstm_fw_torch_ref)
        else:
            mlstm_pt = vlstm_fw_torch_ref
        fn = lambda: mlstm_pt(q, k, v, ig.unsqueeze(-1), fg.unsqueeze(-1))
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    total_flops = 2 * flops_per_matmul
    return ms  # total_flops / ms * 1e-9


if __name__ == "__main__":
    # only works on post-Ampere GPUs right now
    bench_flash_mlstm.run(save_path=".", print_data=True)


# Current result on my RTX 4060 Laptop:

# version 1:
# fused-mlstm-batch2-head8-d128:
#     N_CTX   mLSTM PT  mLSTM PT Compile  mLSTM Triton
# 0   256.0   0.212689          0.107696      0.076507
# 1   512.0   0.578324          0.324955      0.162917
# 2  1024.0   3.295056          1.181538      0.488450
# 3  2048.0  13.671323          4.535624      1.675041
# 4  4096.0  54.495167         18.335104      5.996934
