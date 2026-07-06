import os
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from data import generate_received_signal
import inspect
from models import CMAFormer

# 设置matplotlib字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 配置参数
THETA_GRID = np.arange(-60, 61, 1)  # 角度网格
N = 16                              # 天线数
d = 0.5                             # 天线间距

# 三组对比：
# 1) hard label, no refinement
# 2) soft label, no refinement
# 3) soft label, with refinement
MODEL_CONFIGS = [
    {
        "name": "hard",
        "label": "Hard label",
        "model_path": "checkpoints/depth_4_embed_dim_128/baseline_hard/best_model.pt",
        "color": "#0072BD",
        "marker": "o",
        "use_refinement": False,
    },
    {
        "name": "soft_no_refine",
        "label": "Soft label",
        "model_path": "checkpoints/depth_4_embed_dim_128/baseline_soft/best_model.pt",
        "color": "#D95319",
        "marker": "s",
        "use_refinement": False,
    },
    {
        "name": "soft_refine",
        "label": "Soft label + refinement",
        "model_path": "checkpoints/depth_4_embed_dim_128/baseline_soft/best_model.pt",
        "color": "#77AC30",
        "marker": "^",
        "use_refinement": True,
    },
]

MC_TRIALS = 1000                    # 蒙特卡洛实验次数
SNR_RANGE = np.arange(-20, 1, 2)    # 信噪比测试范围 [-20, -18, ..., 0]
snapshot_num = 1000

# 创建实验目录
EXP_DIR = "exp/hard_soft_ablation"
os.makedirs(EXP_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 固定信源配置
FIXED_SOURCES = [
    {"angles": [-8.25, -1.11], "label": "ThreeTargets"},
]


def load_models():
    """加载hard / soft / soft+refinement三组配置"""
    print("=== 加载模型 ===")
    models = {}

    for config in MODEL_CONFIGS:
        model_name = config["name"]
        model_path = config["model_path"]

        try:
            model, model_config = load_model_compat(model_path=model_path)
            model = model.to(device)
            model.eval()
            models[model_name] = {
                "model": model,
                "config": config,
                "loaded": True
            }
            print(f"✅ {model_name} 加载成功: {model_path}")
        except Exception as e:
            models[model_name] = {
                "model": None,
                "config": config,
                "loaded": False,
                "error": str(e)
            }
            print(f"❌ {model_name} 加载失败: {e}")

    return models

def load_model_compat(model_path):
    """兼容加载 baseline_hard / baseline_soft checkpoint"""
    print(f"正在加载模型: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    model_config = checkpoint.get("model_config", {})
    print(f"从checkpoint加载模型配置: {model_config}")

    model = CMAFormer(**model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    return model, model_config

def generate_test_dataset():
    """
    预先生成所有测试数据，确保所有模型使用相同的数据集
    返回: test_data[config_idx][snr_idx][trial_idx] = X
    """
    print("=== 生成测试数据集 ===")
    test_data = []

    for config_idx, config in enumerate(FIXED_SOURCES):
        angles = config["angles"]
        label = config["label"]
        print(f"生成 {label} 的测试数据...")

        config_data = []
        for snr_idx, snr in enumerate(SNR_RANGE):
            snr_data = []
            for trial in range(MC_TRIALS):
                X = generate_received_signal(angles, SNR=snr, S=snapshot_num)
                snr_data.append(X)
            config_data.append(snr_data)
        test_data.append(config_data)

    print("测试数据集生成完成")
    return test_data


def preprocess_for_proposed(X):
    """预处理函数（与训练一致）"""
    M, N_snap = np.shape(X)
    R = (X @ X.T.conj()) / N_snap

    # 提取上三角非对角元素
    r_complex = []
    for i in range(M):
        for j in range(i + 1, M):
            r_complex.append(R[i, j])

    r_complex = np.array(r_complex)

    # 与原测试脚本保持一致
    r = np.stack([r_complex.real, r_complex.imag, np.sin(np.angle(r_complex)), np.cos(np.angle(r_complex))], axis=-1)
    r_tensor = torch.tensor(r, dtype=torch.float32).unsqueeze(0).to(device)
    return r_tensor


def get_model_prediction(model, X):
    """获取模型预测"""
    X_tensor = preprocess_for_proposed(X)

    with torch.no_grad():
        spectrum = model(X_tensor).cpu().numpy().squeeze()

    return spectrum


def refine_peak_quadratic(spectrum, peak_idx):
    """
    对单个峰做局部二次插值 refinement
    """
    if peak_idx <= 0 or peak_idx >= len(spectrum) - 1:
        return float(THETA_GRID[peak_idx])

    y_l = float(spectrum[peak_idx - 1])
    y_c = float(spectrum[peak_idx])
    y_r = float(spectrum[peak_idx + 1])

    denom = y_l - 2.0 * y_c + y_r
    if abs(denom) < 1e-12:
        delta = 0.0
    else:
        delta = 0.5 * (y_l - y_r) / denom
        delta = np.clip(delta, -1.0, 1.0)

    step = float(THETA_GRID[1] - THETA_GRID[0])
    refined_theta = float(THETA_GRID[peak_idx] + delta * step)
    return refined_theta


def extract_peaks(spectrum, num_sources, threshold=0.0, use_refinement=False):
    """
    从谱图中提取指定数量的峰值角度
    """
    candidate_indices = []
    for i in range(len(spectrum)):
        if spectrum[i] > threshold:
            if (i == 0 or spectrum[i] > spectrum[i - 1]) and \
               (i == len(spectrum) - 1 or spectrum[i] > spectrum[i + 1]):
                candidate_indices.append(i)

    if not candidate_indices:
        return []

    candidate_indices.sort(key=lambda i: spectrum[i], reverse=True)
    peak_indices = candidate_indices[:num_sources]

    if not use_refinement:
        return sorted(float(THETA_GRID[i]) for i in peak_indices)

    refined_angles = [refine_peak_quadratic(spectrum, i) for i in peak_indices]
    return sorted(refined_angles)


def calculate_rmse(true_angles, pred_angles):
    """计算角度估计的均方根误差(RMSE)"""
    if len(pred_angles) != len(true_angles):
        return 180.0

    squared_errors = [(true - pred) ** 2 for true, pred in zip(true_angles, pred_angles)]
    return np.sqrt(np.mean(squared_errors))


def evaluate_models(models, test_data):
    """
    评估 hard / soft / soft+refinement 三组结果
    """
    print("\n=== 评估 hard vs soft 消融实验 ===")

    all_results = {}

    for model_name in models.keys():
        if models[model_name]["loaded"]:
            all_results[model_name] = {}
            for config_idx in range(len(FIXED_SOURCES)):
                all_results[model_name][config_idx] = []

    for config_idx, config in enumerate(FIXED_SOURCES):
        angles = config["angles"]
        num_sources = len(angles)
        label = config["label"]

        print(f"\n评估配置: {label} (角度: {angles})")

        for snr_idx, snr in enumerate(SNR_RANGE):
            print(f"  SNR: {snr}dB", end="")

            model_rmse_results = {}
            for model_name, model_info in models.items():
                if model_info["loaded"]:
                    model_rmse_results[model_name] = []

            for trial in range(MC_TRIALS):
                X = test_data[config_idx][snr_idx][trial]

                for model_name, model_info in models.items():
                    if model_info["loaded"]:
                        model = model_info["model"]
                        use_refinement = model_info["config"]["use_refinement"]

                        spectrum = get_model_prediction(model, X)
                        pred_angles = extract_peaks(
                            spectrum,
                            num_sources,
                            threshold=0.0,
                            use_refinement=use_refinement
                        )
                        rmse = calculate_rmse(sorted(angles), sorted(pred_angles))
                        model_rmse_results[model_name].append(rmse)

            for model_name in model_rmse_results.keys():
                avg_rmse = np.mean(model_rmse_results[model_name])
                all_results[model_name][config_idx].append(avg_rmse)

            rmse_str = " | ".join([
                f"{models[name]['config']['label']}: {np.mean(model_rmse_results[name]):.2f}°"
                for name in model_rmse_results.keys()
            ])
            print(f" | {rmse_str}")

    return all_results


def calculate_crb(angles, snr_db, N=16, d=0.5, S=snapshot_num):
    """计算CRB理论下界"""
    angles_rad = np.array(angles) * np.pi / 180
    K = len(angles)
    snr_linear = 10 ** (snr_db / 10)
    noise_var = 1.0

    n = np.arange(N)
    A = np.exp(1j * 2 * np.pi * d * np.outer(np.sin(angles_rad), n)).T

    D = np.zeros((N, K), dtype=complex)
    for k in range(K):
        D[:, k] = 1j * 2 * np.pi * d * np.cos(angles_rad[k]) * n * A[:, k]

    P_A_perp = np.eye(N) - A @ np.linalg.pinv(A)

    H = D.conj().T @ P_A_perp @ D
    P_est = snr_linear * np.eye(K)
    FIM = 2 * S / noise_var * np.real(H * P_est.T)

    try:
        crb_matrix = np.linalg.inv(FIM)
        crb_rad = np.diag(crb_matrix)
    except np.linalg.LinAlgError:
        crb_matrix = np.linalg.pinv(FIM)
        crb_rad = np.diag(crb_matrix)

    crb_deg = np.sqrt(np.abs(crb_rad)) * 180 / np.pi
    return crb_deg


def calculate_all_crb(fixed_sources, snr_range):
    """计算所有配置下的CRB"""
    all_crb = []

    print("\n=== 计算CRB理论下界 ===")
    for config in fixed_sources:
        angles = config["angles"]
        label = config["label"]
        crb_results = []

        print(f"计算 {label} 的CRB...")
        for snr in snr_range:
            crb_values = calculate_crb(angles, snr)
            avg_crb = np.mean(crb_values)
            crb_results.append(avg_crb)

        all_crb.append(crb_results)

    return all_crb


def plot_hard_soft_comparison(all_results, condition_idx):
    """绘制单个条件下 hard / soft / soft+refinement 性能对比"""
    config = FIXED_SOURCES[condition_idx]
    label = config["label"]

    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.15)

    for model_cfg in MODEL_CONFIGS:
        model_name = model_cfg["name"]
        if model_name not in all_results:
            continue

        rmse_vals = all_results[model_name][condition_idx]

        plt.plot(
            SNR_RANGE,
            rmse_vals,
            model_cfg["marker"] + '-',
            label=model_cfg["label"],
            color=model_cfg["color"],
            markerfacecolor='white',
            markeredgewidth=2,
            markersize=8,
            linewidth=2
        )

    # 如需画CRB，取消下面注释
    # plt.plot(
    #     SNR_RANGE,
    #     crb_results[condition_idx],
    #     label='CRB',
    #     color='black',
    #     linestyle='--',
    #     linewidth=2
    # )

    plt.xlabel('SNR (dB)', fontfamily='Times New Roman', fontsize=20, fontweight='bold', labelpad=10)
    plt.ylabel('RMSE (degree)', fontfamily='Times New Roman', fontsize=20, fontweight='bold', labelpad=10)
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.3, which='both')

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.yaxis.set_tick_params(pad=10)

    plt.legend(
        loc='best',
        frameon=True,
        framealpha=0.95,
        edgecolor='gray',
        fancybox=True,
        shadow=True,
        prop={'family': 'Times New Roman', 'size': 12}
    )

    plt.savefig(f'{EXP_DIR}/hard_soft_ablation_{label}.eps', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    print("=== 开始 hard vs soft 消融实验 ===")
    print(f"测试配置: {[cfg['label'] for cfg in MODEL_CONFIGS]}")
    print(f"测试场景: {[config['label'] for config in FIXED_SOURCES]}")
    print(f"SNR范围: {SNR_RANGE[0]}dB 到 {SNR_RANGE[-1]}dB")
    print(f"蒙特卡洛试验次数: {MC_TRIALS}")

    try:
        # 1. 加载模型
        models = load_models()

        loaded_models = {name: info for name, info in models.items() if info["loaded"]}
        if not loaded_models:
            print("没有成功加载任何模型，请检查模型路径")
            exit(1)

        print(f"\n成功加载 {len(loaded_models)} 个模型")

        # 2. 预先生成所有测试数据
        test_data = generate_test_dataset()

        # 3. 评估所有模型性能
        all_results = evaluate_models(loaded_models, test_data)

        print("\n=== 生成分析图表 ===")

        # 如需CRB，取消下面注释
        # crb_results = calculate_all_crb(FIXED_SOURCES, SNR_RANGE)

        # 4. 绘图
        for condition_idx in range(len(FIXED_SOURCES)):
            plot_hard_soft_comparison(all_results, condition_idx)
            config_name = FIXED_SOURCES[condition_idx]["label"]
            print(f"✅ {config_name} 性能对比图已保存")

    except Exception as e:
        print(f"❌ 运行时出现错误: {e}")
        import traceback
        traceback.print_exc()