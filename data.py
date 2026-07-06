import os
import h5py
import numpy as np
from itertools import combinations
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import threading

# 创建DATA文件夹
os.makedirs("DATA", exist_ok=True)

# 扩展参数配置
theta_grid = np.arange(-60, 61, 1)  # -60°到60°，间隔1° (121个角度)
SNR_list = [-20, -15, -10, -5, 0]   # 信噪比(dB)
snapshot_list = [1000]               # 变化的快拍数
N = 16                               # 天线数
d = 0.5                              # 天线间距(λ/2)

LABEL_TYPE = "hard"   # "hard" / "gaussian_soft"
GAUSSIAN_SIGMA = 2.0           # 可先用 2.0，后续可做 1.0 / 1.5 / 2.0 对比
OUTPUT_PATH = "DATA/doa_signal_dataset_lite_hard.h5"
DATA_RATIO = 1.0

def generate_steering_vector(theta):
    """生成导向矢量"""
    return np.exp(1j * 2 * np.pi * d * np.arange(N) * np.sin(np.deg2rad(theta)))

def generate_received_signal(thetas, SNR, S, seed=None):
    """生成接收信号矩阵 X (N x S)"""
    if seed is not None:
        np.random.seed(seed)
    
    A = np.column_stack([generate_steering_vector(theta) for theta in thetas])
    K = len(thetas)
    
    # 生成信号（单位功率）
    signal = (np.random.randn(K, S) + 1j * np.random.randn(K, S)) / np.sqrt(2)
    
    # 生成噪声
    noise_power = 10**(-SNR/10)
    noise = np.sqrt(noise_power/2) * (np.random.randn(N, S) + 1j * np.random.randn(N, S))
    
    # 接收信号
    return A @ signal + noise

def generate_label(thetas):
    """生成标签：
    - hard: 原始one-hot/multi-hot
    - gaussian_soft: 按审稿意见改成 Gaussian soft label
      s_i = max_k exp(-(g_i - theta_k)^2 / (2*sigma^2))
    """
    if LABEL_TYPE == "hard":
        label = np.zeros(len(theta_grid), dtype=np.float32)
        for theta in thetas:
            tt = np.argmin(np.abs(theta_grid - theta))
            if np.abs(theta_grid[tt] - theta) < 0.1:
                label[tt] = 1.0
            else:
                raise ValueError(f"角度{theta}超出网格范围")
        return label

    elif LABEL_TYPE == "gaussian_soft":
        thetas = np.asarray(thetas, dtype=np.float32)[:, None]          # [K, 1]
        grid = theta_grid.astype(np.float32)[None, :]                   # [1, 121]
        label = np.exp(-((grid - thetas) ** 2) / (2 * GAUSSIAN_SIGMA ** 2))
        label = label.max(axis=0).astype(np.float32)                    # 按图里公式取 max_k
        return label

    else:
        raise ValueError(f"Unsupported LABEL_TYPE: {LABEL_TYPE}")

def signal_to_network_input(X):
    """
    将接收信号矩阵转换为网络输入格式
    根据论文公式(11)和(12)，提取协方差矩阵的上三角非对角元素
    """
    # 计算协方差矩阵
    R = (X @ X.conj().T) / X.shape[1]
    
    # 提取上三角非对角元素
    r_complex = []
    for i in range(N):
        for j in range(i+1, N):
            r_complex.append(R[i, j])
    
    r_complex = np.array(r_complex)
    
    # 分离实部和虚部
    r_real = r_complex.real
    r_imag = r_complex.imag
    r_sin = np.sin(np.angle(r_complex))
    r_cos = np.cos(np.angle(r_complex))
    
    r = np.stack([r_real, r_imag, r_sin, r_cos], axis=-1)
    
    return r

def generate_balanced_combinations():
    """生成轻量化后的平衡组合"""
    np.random.seed(42)
    
    # 生成所有单/双/三源组合
    theta_combs = {
        1: list(combinations(theta_grid, 1)),  # 单源
        2: list(combinations(theta_grid, 2)),  # 双源
        3: list(combinations(theta_grid, 3))   # 三源
    }
    
    # 原始每类目标数量都是 287980，这里按比例缩减
    target_per_class = int(len(theta_combs[3]) * DATA_RATIO)
    
    balanced_combs = []
    
    # 1. 单源：按比例缩减重复次数，不足部分随机补齐
    single_repeat = target_per_class // len(theta_combs[1])
    balanced_combs.extend(theta_combs[1] * single_repeat)
    single_remain = target_per_class - single_repeat * len(theta_combs[1])
    if single_remain > 0:
        np.random.shuffle(theta_combs[1])
        balanced_combs.extend(theta_combs[1][:single_remain])
    
    # 2. 双源：按比例缩减完整重复次数，不足部分随机补齐
    dual_repeat = target_per_class // len(theta_combs[2])
    balanced_combs.extend(theta_combs[2] * dual_repeat)
    dual_remain = target_per_class - dual_repeat * len(theta_combs[2])
    if dual_remain > 0:
        np.random.shuffle(theta_combs[2])
        balanced_combs.extend(theta_combs[2][:dual_remain])
    
    # 3. 三源：随机抽取 target_per_class 个组合
    np.random.shuffle(theta_combs[3])
    balanced_combs.extend(theta_combs[3][:target_per_class])
    
    return balanced_combs

def generate_random_combination(k, n_samples):
    """高效生成k个角度的随机组合"""
    n_angles = len(theta_grid)
    indices = np.arange(n_angles)
    unique_combinations = set()
    
    while len(unique_combinations) < n_samples:
        selected = tuple(sorted(np.random.choice(indices, k, replace=False)))
        unique_combinations.add(selected)
    
    return [tuple(theta_grid[i] for i in comb) for comb in unique_combinations]

def generate_full_dataset():
    """高效生成完整数据集"""
    theta_combs = {
        1: list(combinations(theta_grid, 1)),  # 单源
        2: list(combinations(theta_grid, 2)),  # 双源
    }

    samples = []
    
    # 1. 单源场景（每个角度1个样本）
    print("生成单源组合...")
    samples.extend(theta_combs[1] * 50)
    
    # 2. 双源场景（所有组合）
    print("生成双源组合...")
    samples.extend(theta_combs[2] * 5)
    
    # 3. 三源场景（随机选择10000个组合）
    print("生成三源组合...")
    samples.extend(generate_random_combination(3, 5000))
    
    # 5. 五源场景（随机选择2000个组合）
    print("生成五源组合...")
    samples.extend(generate_random_combination(5, 0))
    
    return samples

def process_angle_combination(args):
    """处理单个角度组合的所有SNR和快拍数组合"""
    thetas, base_idx, base_seed = args
    
    # 为每个进程设置不同的随机种子
    np.random.seed(base_seed)
    
    # 生成标签（121维二进制向量）
    label = generate_label(thetas)
    
    results = []
    local_idx = 0
    
    for SNR in SNR_list:
        for S in snapshot_list:
            seed = base_seed + base_idx * 1000 + local_idx
            X = generate_received_signal(thetas, SNR, S, seed)
            r = signal_to_network_input(X)
            
            results.append({
                'r': r,
                'label': label,
                'SNR': SNR,
                'snapshots': S,
                'source_count': len(thetas)
            })
            local_idx += 1
    
    return results

def save_to_h5(X_data, labels, snr_values, snapshots, source_counts):
    """保存接收信号矩阵到HDF5文件"""
    with h5py.File(OUTPUT_PATH, "w") as f:
        f.create_dataset("X", data=X_data, dtype=np.float32)
        f.create_dataset("labels", data=labels, dtype=np.float32)
        f.create_dataset("SNR", data=snr_values, dtype=np.int32)
        f.create_dataset("snapshots", data=snapshots, dtype=np.int32)
        f.create_dataset("source_counts", data=source_counts, dtype=np.int32)

        f.attrs["theta_grid"] = theta_grid
        f.attrs["N"] = N
        f.attrs["d"] = d
        f.attrs["label_type"] = LABEL_TYPE
        f.attrs["gaussian_sigma"] = GAUSSIAN_SIGMA

def main():
    angle_combinations = generate_balanced_combinations()
    
    total_samples = len(angle_combinations) * len(SNR_list) * len(snapshot_list)
    print(f"总样本量: {total_samples}")
    print(f"角度组合数: {len(angle_combinations)}")
    
    X_data = np.zeros((total_samples, N * (N-1) // 2, 4), dtype=np.float32)
    labels = np.zeros((total_samples, len(theta_grid)), dtype=np.float32)
    snr_values = np.zeros(total_samples, dtype=np.int32)
    snapshots = np.zeros(total_samples, dtype=np.int32)
    source_counts = np.zeros(total_samples, dtype=np.int32)
    
    process_args = []
    for i, thetas in enumerate(angle_combinations):
        base_seed = 42 + i
        process_args.append((thetas, i, base_seed))
    
    num_processes = min(cpu_count(), 8)
    print(f"使用 {num_processes} 个进程")
    
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_angle_combination, process_args),
            total=len(process_args),
            desc="生成数据"
        ))
    
    print("整理数据...")
    idx = 0
    for angle_results in tqdm(results, desc="保存数据"):
        for result in angle_results:
            X_data[idx] = result['r']
            labels[idx] = result['label']
            snr_values[idx] = result['SNR']
            snapshots[idx] = result['snapshots']
            source_counts[idx] = result['source_count']
            idx += 1
    
    print("保存到HDF5文件...")
    save_to_h5(X_data, labels, snr_values, snapshots, source_counts)
    print(f"数据集已保存到 {OUTPUT_PATH}")
    print(f"接收信号维度: {X_data.shape}")
    print(f"标签维度: {labels.shape}")

if __name__ == "__main__":
    main()
