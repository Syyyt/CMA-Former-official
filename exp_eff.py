"""
效率对比实验：FLOPs / 推理时延 / 推理峰值显存 / 训练峰值显存 / 训练耗时 / 参数量

口径说明（确保各模型公平对比，避免审稿人挑刺）：
  - FLOPs：B=1，纯前向，使用 calflops（若不可用自动回退到 thop）。MUSIC 不参与（标 N/A）。
  - 推理时延：B=1，GPU，纯前向。50 次 warm-up + 1000 次正式测。
              使用 torch.cuda.synchronize() 保证测量准确。
              报 mean / std / p50 / p95（单位 ms）。
  - 推理峰值显存：torch.cuda.reset_peak_memory_stats() 包夹 forward 调用。
  - 训练峰值显存 & 训练耗时：生成虚拟数据，固定 batch_size=64，跑 N 个 iteration，
                              Adam(lr=1e-3)，loss 与各模型推理语义一致：
                                * HMC-ViT 输出 logits → BCEWithLogitsLoss
                                * 其他网络输出已是概率/谱 → MSELoss
                              记录每 iteration 平均耗时与峰值显存。
  - MUSIC：完全跳过 FLOPs/训练相关项，只测推理时延（CPU 上更公平，因为它本身是非神经网络
            算法，但为与其他模型保持设备一致，主表仍报 GPU 同设备下的 CPU-numpy 实现耗时）。
  - 参数量：可学习参数总数。

输出：
  exp/efficiency_comparison/summary.txt   人类可读
  exp/efficiency_comparison/summary.md    论文用 Markdown 表
  exp/efficiency_comparison/summary.csv   原始数据
"""

import os
import time
import gc
import numpy as np

import gpu_manager
gpu_manager.auto_select_gpu()

import torch
import torch.nn as nn

from data import generate_received_signal
from exp_utils.load_model import (
    load_model, load_cnn_model, load_dct_model, load_dnn_model, load_hmc_model
)
from exp_utils.music import music_spectrum

# ============================================================
# 配置
# ============================================================
DEPTH = 4
EMBED = 128
MODEL_PATH = f"checkpoints/depth_{DEPTH}_embed_dim_{EMBED}/best_model.pt"

EXP_DIR = "exp/efficiency_comparison"
os.makedirs(EXP_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 时延测量
LATENCY_WARMUP = 50
LATENCY_TRIALS = 1000

# 训练时长测量
TRAIN_BATCH_SIZE = 64
TRAIN_ITERS = 200    # 200 个 iteration 足以稳定测出 per-iter 耗时
TRAIN_WARMUP_ITERS = 10  # 前 10 个 iter 不计入时间统计

# MUSIC 时延（次数少一些，因为它本身慢且不是重点）
MUSIC_TRIALS = 100
MUSIC_WARMUP = 5

# MUSIC 与生成数据用的参数（与原脚本一致）
N_ANTENNAS = 16
SNAPSHOT_NUM = 500
THETA_GRID = np.arange(-60, 61, 1)
D_SPACING = 0.5
MUSIC_TEST_ANGLES = [-12.89, -9.13, 8.09, 11.12]  # 取 4 源场景作时延基准
MUSIC_TEST_SNR = 0


# ============================================================
# 各模型输入构造（dummy，用于 FLOPs/时延/训练）
# ============================================================
# 输入形状均参考 exp_model_comparison.py 中的 preprocess_for_* 函数
# 注意：FLOPs / 时延用 batch=1，训练用 batch=TRAIN_BATCH_SIZE
def make_dummy_input(model_name: str, batch_size: int = 1) -> torch.Tensor:
    M = N_ANTENNAS
    L = M * (M - 1) // 2  # 上三角元素数 = 120
    if model_name == "Proposed":
        # (B, 120, 4)
        return torch.randn(batch_size, L, 4, device=device)
    if model_name == "CNN":
        # (B, 3, 16, 16)
        return torch.randn(batch_size, 3, M, M, device=device)
    if model_name == "DCT-ViT":
        # (B, 3, 16, 16)
        return torch.randn(batch_size, 3, M, M, device=device)
    if model_name == "DNN":
        # (B, 240) —— 实部+虚部拼接
        return torch.randn(batch_size, 2 * L, device=device)
    if model_name == "HMC-ViT":
        # (B, 3, 1, 120)
        return torch.randn(batch_size, 3, 1, L, device=device)
    raise ValueError(f"Unknown model: {model_name}")


def make_dummy_target(model_name: str, model_output: torch.Tensor) -> torch.Tensor:
    """
    生成与模型输出形状匹配的 target。
    HMC-ViT 用 BCE，其他用 MSE。target 为随机谱（[0,1] 之间）以模拟训练分布。
    """
    if model_name == "HMC-ViT":
        return torch.randint(0, 2, model_output.shape, device=device).float()
    return torch.rand_like(model_output)


def get_loss_fn(model_name: str):
    if model_name == "HMC-ViT":
        return nn.BCEWithLogitsLoss()
    return nn.MSELoss()


# ============================================================
# 工具函数
# ============================================================
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _unpack_output(out):
    """部分模型 (如 DNN) 的 forward 返回 tuple, 取第 0 个元素作为主输出"""
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_peak_mem():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def get_peak_mem_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


# ============================================================
# FLOPs 测量（自动选择可用后端）
# ============================================================
def _manual_flops_count(model: nn.Module, dummy_input: torch.Tensor):
    """
    使用 forward hooks 估算 FLOPs。覆盖 Linear / Conv1d/2d/3d / BatchNorm / LayerNorm。
    注意: 这是**兜底方案**, 对 ViT 中 attention 的 QK^T 和 score@V 矩阵乘
    无法捕获 (这部分不是 nn.Module 而是函数调用), 因此对 ViT 类模型会**低估**.
    强烈建议安装 calflops 或 thop 来获得准确结果.
    返回: (flops_G, macs_G)
    """
    macs_total = [0]

    def linear_hook(module, inp, out):
        out_t = out[0] if isinstance(out, (tuple, list)) else out
        # output shape: (..., out_features), MACs = N * in * out
        n_outputs = out_t.numel() // module.out_features
        macs_total[0] += n_outputs * module.in_features * module.out_features

    def conv_hook(module, inp, out):
        out_t = out[0] if isinstance(out, (tuple, list)) else out
        kernel_ops = module.in_channels // module.groups
        for k in module.kernel_size:
            kernel_ops *= k
        macs_total[0] += out_t.numel() * kernel_ops

    handles = []
    for m in model.modules():
        if isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))
        elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            handles.append(m.register_forward_hook(conv_hook))

    model.eval()
    try:
        with torch.no_grad():
            _ = model(dummy_input)
    finally:
        for h in handles:
            h.remove()

    macs = macs_total[0]
    return (2 * macs) / 1e9, macs / 1e9


def measure_flops(model: nn.Module, dummy_input: torch.Tensor):
    """
    返回 (flops_G, macs_G, backend_name)。若失败返回 (None, None, None)。
    优先级: calflops > thop > 手动 hook 兜底
    """
    model.eval()

    # 优先尝试 calflops（对 ViT 支持最准）
    try:
        from calflops import calculate_flops
        flops, macs, _params = calculate_flops(
            model=model,
            input_shape=tuple(dummy_input.shape),
            output_as_string=False,
            output_precision=4,
            print_results=False,
        )
        return flops / 1e9, macs / 1e9, "calflops"
    except ImportError:
        print(f"    [calflops 未安装, 尝试 thop ...]")
    except Exception as e:
        print(f"    [calflops 调用失败: {e}, 尝试 thop ...]")

    # 回退到 thop
    try:
        from thop import profile
        macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
        return (2 * macs) / 1e9, macs / 1e9, "thop"
    except ImportError:
        print(f"    [thop 未安装, 使用手动 hook 兜底 (ViT 类会低估) ...]")
    except Exception as e:
        print(f"    [thop 调用失败: {e}, 使用手动 hook 兜底 ...]")

    # 最终兜底
    try:
        flops_g, macs_g = _manual_flops_count(model, dummy_input)
        return flops_g, macs_g, "manual_hook"
    except Exception as e:
        print(f"    [手动兜底也失败: {e}]")
        return None, None, None


# ============================================================
# 推理时延 + 峰值显存
# ============================================================
@torch.no_grad()
def measure_inference(model: nn.Module, model_name: str):
    model.eval()
    dummy = make_dummy_input(model_name, batch_size=1)

    # warm-up
    for _ in range(LATENCY_WARMUP):
        _ = _unpack_output(model(dummy))
    cuda_sync()

    # 峰值显存（含一次完整 forward）
    reset_peak_mem()
    _ = _unpack_output(model(dummy))
    cuda_sync()
    peak_mb = get_peak_mem_mb()

    # 时延：使用 cuda Event 比 time.perf_counter 更准（GPU 异步）
    times_ms = []
    if torch.cuda.is_available():
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        for _ in range(LATENCY_TRIALS):
            starter.record()
            _ = model(dummy)
            ender.record()
            torch.cuda.synchronize()
            times_ms.append(starter.elapsed_time(ender))
    else:
        for _ in range(LATENCY_TRIALS):
            t0 = time.perf_counter()
            _ = model(dummy)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    times_ms = np.array(times_ms, dtype=np.float64)
    return {
        "mean_ms": float(times_ms.mean()),
        "std_ms": float(times_ms.std()),
        "p50_ms": float(np.percentile(times_ms, 50)),
        "p95_ms": float(np.percentile(times_ms, 95)),
        "peak_mem_mb": float(peak_mb),
    }


# ============================================================
# 训练耗时 + 训练峰值显存
# ============================================================
def measure_training(model: nn.Module, model_name: str):
    model.train()
    loss_fn = get_loss_fn(model_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    dummy_input = make_dummy_input(model_name, batch_size=TRAIN_BATCH_SIZE)
    # 先跑一次 forward 拿到输出形状以构造 target
    with torch.no_grad():
        out = _unpack_output(model(dummy_input))
    target = make_dummy_target(model_name, out)

    # warm-up（不计时间也不计显存）
    for _ in range(TRAIN_WARMUP_ITERS):
        optimizer.zero_grad()
        out = _unpack_output(model(dummy_input))
        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()
    cuda_sync()

    # 正式测量
    reset_peak_mem()
    iter_times = []
    for _ in range(TRAIN_ITERS):
        cuda_sync()
        t0 = time.perf_counter()
        optimizer.zero_grad()
        out = _unpack_output(model(dummy_input))
        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()
        cuda_sync()
        t1 = time.perf_counter()
        iter_times.append((t1 - t0) * 1000.0)  # ms

    peak_mb = get_peak_mem_mb()
    iter_times = np.array(iter_times)

    return {
        "iter_mean_ms": float(iter_times.mean()),
        "iter_std_ms": float(iter_times.std()),
        "total_s": float(iter_times.sum() / 1000.0),
        "iters": TRAIN_ITERS,
        "batch_size": TRAIN_BATCH_SIZE,
        "peak_mem_mb": float(peak_mb),
    }


# ============================================================
# MUSIC 时延（CPU + numpy，独立测量）
# ============================================================
def measure_music_latency():
    print("  [MUSIC] 测量推理时延（CPU/numpy 实现）...")
    # 用一个固定样本，避免数据生成耗时混入
    X = generate_received_signal(MUSIC_TEST_ANGLES, SNR=MUSIC_TEST_SNR, S=SNAPSHOT_NUM)
    K = len(MUSIC_TEST_ANGLES)

    # warm-up
    for _ in range(MUSIC_WARMUP):
        _ = music_spectrum(X, num_sources=K, theta_grid=THETA_GRID, N=N_ANTENNAS, d=D_SPACING)

    times_ms = []
    for _ in range(MUSIC_TRIALS):
        t0 = time.perf_counter()
        _ = music_spectrum(X, num_sources=K, theta_grid=THETA_GRID, N=N_ANTENNAS, d=D_SPACING)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)
    times_ms = np.array(times_ms)

    return {
        "mean_ms": float(times_ms.mean()),
        "std_ms": float(times_ms.std()),
        "p50_ms": float(np.percentile(times_ms, 50)),
        "p95_ms": float(np.percentile(times_ms, 95)),
    }


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 70)
    print("效率对比实验")
    print(f"设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    # 加载所有神经网络模型
    print("\n--- 加载模型 ---")
    proposed_model, _ = load_model(model_path=MODEL_PATH)
    print("Proposed 加载成功")
    cnn_model = load_cnn_model();          print("CNN 加载成功")
    dct_vit_model = load_dct_model();      print("DCT-ViT 加载成功")
    hmc_vit_model = load_hmc_model();      print("HMC-ViT 加载成功")
    dnn_model = load_dnn_model();          print("DNN 加载成功")

    nn_models = {
        "Proposed": proposed_model,
        "CNN": cnn_model,
        "DCT-ViT": dct_vit_model,
        "HMC-ViT": hmc_vit_model,
        "DNN": dnn_model,
    }

    results = {}

    # ---- 神经网络模型: 全部指标 ----
    for name, model in nn_models.items():
        print(f"\n=== {name} ===")
        model = model.to(device)

        # 1. 参数量
        n_params = count_parameters(model)
        print(f"  参数量: {n_params:,} ({n_params/1e6:.4f} M)")

        # 2. FLOPs
        dummy = make_dummy_input(name, batch_size=1)
        gflops, gmacs, backend = measure_flops(model, dummy)
        if gflops is not None:
            tag = f" [{backend}]" if backend else ""
            print(f"  FLOPs: {gflops:.4f} G  |  MACs: {gmacs:.4f} G{tag}")
            if backend == "manual_hook":
                print(f"  [警告] 手动 hook 兜底, ViT 类模型 attention 部分会低估, "
                      f"建议 pip install calflops")
        else:
            print(f"  FLOPs: 测量失败")

        # 3. 推理时延 + 推理峰值显存
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        inf = measure_inference(model, name)
        print(f"  推理时延 (B=1, GPU, 纯前向): "
              f"{inf['mean_ms']:.4f} ± {inf['std_ms']:.4f} ms  "
              f"(p50={inf['p50_ms']:.4f}, p95={inf['p95_ms']:.4f})")
        print(f"  推理峰值显存: {inf['peak_mem_mb']:.3f} MB")

        # 4. 训练耗时 + 训练峰值显存
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        try:
            tr = measure_training(model, name)
            print(f"  训练 per-iter: {tr['iter_mean_ms']:.3f} ± {tr['iter_std_ms']:.3f} ms  "
                  f"(B={tr['batch_size']}, {tr['iters']} iters, 总计 {tr['total_s']:.2f} s)")
            print(f"  训练峰值显存: {tr['peak_mem_mb']:.2f} MB")
        except Exception as e:
            print(f"  训练测量失败: {e}")
            tr = None

        results[name] = {
            "params": n_params,
            "gflops": gflops,
            "gmacs": gmacs,
            "flops_backend": backend,
            "inference": inf,
            "training": tr,
        }

    # ---- MUSIC: 仅推理时延 ----
    print(f"\n=== MUSIC ===")
    music_inf = measure_music_latency()
    print(f"  推理时延: {music_inf['mean_ms']:.3f} ± {music_inf['std_ms']:.3f} ms  "
          f"(p50={music_inf['p50_ms']:.3f}, p95={music_inf['p95_ms']:.3f})")
    results["MUSIC"] = {
        "params": None, "gflops": None, "gmacs": None, "flops_backend": None,
        "inference": music_inf, "training": None,
    }

    # ---- 输出汇总 ----
    write_summary(results)
    print(f"\n=== 完成 ===")
    print(f"结果文件:")
    print(f"  {EXP_DIR}/summary.txt")
    print(f"  {EXP_DIR}/summary.md")
    print(f"  {EXP_DIR}/summary.csv")


# ============================================================
# 输出
# ============================================================
def _fmt(val, fmt="{:.3f}"):
    return "N/A" if val is None else fmt.format(val)


def write_summary(results):
    # ---- summary.txt (人类可读) ----
    with open(f"{EXP_DIR}/summary.txt", "w", encoding="utf-8") as f:
        f.write("Efficiency Comparison Summary\n")
        f.write("=" * 80 + "\n")
        f.write(f"Device: {device}")
        if torch.cuda.is_available():
            f.write(f"  ({torch.cuda.get_device_name(0)})")
        f.write("\n")
        f.write(f"Latency: B=1, pure forward, {LATENCY_WARMUP} warm-up + {LATENCY_TRIALS} trials\n")
        f.write(f"Training: B={TRAIN_BATCH_SIZE}, {TRAIN_ITERS} iters (after {TRAIN_WARMUP_ITERS} warm-up)\n")
        f.write("=" * 80 + "\n\n")

        for name, r in results.items():
            f.write(f"[{name}]\n")
            f.write(f"  Params       : {_fmt(r['params'], '{:,}') if r['params'] else 'N/A'}\n")
            backend_tag = f" [{r['flops_backend']}]" if r.get('flops_backend') else ""
            f.write(f"  FLOPs (G)    : {_fmt(r['gflops'], '{:.4f}')}{backend_tag}\n")
            f.write(f"  MACs  (G)    : {_fmt(r['gmacs'],  '{:.4f}')}\n")
            inf = r["inference"]
            f.write(f"  Inference    : {inf['mean_ms']:.4f} ± {inf['std_ms']:.4f} ms "
                    f"(p50={inf['p50_ms']:.4f}, p95={inf['p95_ms']:.4f})\n")
            inf_mem = inf.get("peak_mem_mb", None)
            f.write(f"  Inf. Mem (MB): {_fmt(inf_mem, '{:.3f}')}\n")
            tr = r["training"]
            if tr is None:
                f.write(f"  Train iter   : N/A\n")
                f.write(f"  Train Mem(MB): N/A\n")
            else:
                f.write(f"  Train iter   : {tr['iter_mean_ms']:.3f} ± {tr['iter_std_ms']:.3f} ms "
                        f"(total {tr['total_s']:.2f} s for {tr['iters']} iters)\n")
                f.write(f"  Train Mem(MB): {tr['peak_mem_mb']:.2f}\n")
            f.write("\n")

    # ---- summary.md (论文表) ----
    with open(f"{EXP_DIR}/summary.md", "w", encoding="utf-8") as f:
        f.write("# Efficiency Comparison\n\n")
        f.write(f"Device: `{device}`")
        if torch.cuda.is_available():
            f.write(f" ({torch.cuda.get_device_name(0)})")
        f.write("\n\n")

        f.write("| Model | #Params | FLOPs (G) | Inference (ms) | Inf. Mem (MB) "
                "| Train iter (ms) | Train Mem (MB) |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for name, r in results.items():
            params = f"{r['params']:,}" if r['params'] else "N/A"
            gflops = _fmt(r['gflops'], "{:.3f}")
            inf = r["inference"]
            inf_lat = f"{inf['mean_ms']:.3f} ± {inf['std_ms']:.3f}"
            inf_mem = _fmt(inf.get("peak_mem_mb"), "{:.2f}")
            tr = r["training"]
            if tr is None:
                tr_iter, tr_mem = "N/A", "N/A"
            else:
                tr_iter = f"{tr['iter_mean_ms']:.2f} ± {tr['iter_std_ms']:.2f}"
                tr_mem = f"{tr['peak_mem_mb']:.2f}"
            f.write(f"| {name} | {params} | {gflops} | {inf_lat} | {inf_mem} | {tr_iter} | {tr_mem} |\n")

    # ---- summary.csv (原始数据) ----
    import csv
    with open(f"{EXP_DIR}/summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "model", "params", "gflops", "gmacs",
            "inf_mean_ms", "inf_std_ms", "inf_p50_ms", "inf_p95_ms", "inf_peak_mem_mb",
            "train_iter_mean_ms", "train_iter_std_ms", "train_total_s",
            "train_iters", "train_batch_size", "train_peak_mem_mb",
        ])
        for name, r in results.items():
            inf = r["inference"]
            tr = r["training"] or {}
            w.writerow([
                name,
                r["params"] if r["params"] is not None else "",
                r["gflops"] if r["gflops"] is not None else "",
                r["gmacs"] if r["gmacs"] is not None else "",
                inf["mean_ms"], inf["std_ms"], inf["p50_ms"], inf["p95_ms"],
                inf.get("peak_mem_mb", ""),
                tr.get("iter_mean_ms", ""), tr.get("iter_std_ms", ""), tr.get("total_s", ""),
                tr.get("iters", ""), tr.get("batch_size", ""), tr.get("peak_mem_mb", ""),
            ])


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"运行错误: {e}")
        import traceback
        traceback.print_exc()