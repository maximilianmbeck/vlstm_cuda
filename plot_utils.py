import torch
from matplotlib import pyplot as plt


def convert_to_diff_imarray(target: torch.Tensor, baseline: torch.Tensor = None):
    if baseline is None:
        imarr = target.float().abs().squeeze().cpu().numpy()
    else:
        imarr = (target.float() - baseline.float()).abs().squeeze().cpu().numpy()
    if imarr.ndim < 2:
        imarr = imarr[:, None]
    return imarr


def plot_numerical_diffs(
    pt_fp32_baseline,
    cu_fp32,
    cu_bf16,
    cu_half,
    title,
    vmin=0.0,
    vmax=1e-2,
    figsize=(10, 6),
):
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=figsize, ncols=3)

    pos1 = ax1.imshow(
        convert_to_diff_imarray(cu_fp32, pt_fp32_baseline),
        vmin=vmin,
        vmax=vmax,
    )
    ax1.set_title("float32")
    fig.colorbar(pos1, ax=ax1)
    pos2 = ax2.imshow(
        convert_to_diff_imarray(cu_bf16, pt_fp32_baseline),
        vmin=vmin,
        vmax=vmax,
    )
    ax2.set_title("bfloat16")
    fig.colorbar(pos2, ax=ax2)
    pos3 = ax3.imshow(
        convert_to_diff_imarray(cu_half, pt_fp32_baseline),
        vmin=vmin,
        vmax=vmax,
    )
    ax3.set_title("float16")
    fig.colorbar(pos3, ax=ax3)
    fig.suptitle(title)
    return fig


def plot_numerical_diffs_single(
    baseline, target=None, title="", vmin=0.0, vmax=1e-2, figsize=(10, 6)
):
    fig, ax1 = plt.subplots(figsize=figsize)
    pos1 = ax1.imshow(
        convert_to_diff_imarray(baseline, baseline=target),
        vmin=vmin,
        vmax=vmax,
    )
    ax1.set_title(title)
    fig.colorbar(pos1, ax=ax1)
    return fig
