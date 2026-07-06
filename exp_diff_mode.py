import os
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from data import generate_received_signal
from models import CMAFormer

# 设置matplotlib字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 配置参数
THETA_GRID = np.arange(-60, 61, 1)  # 角度网格
N = 16                              # 天线数
d = 0.5                             # 天线间距

DEPTH = 4
EMBED = 128
MC_TRIALS = 1000
SNR_RANGE = np.arange(-20, 1, 2)   # [-20, -18, ..., 0]
snapshot_num = 1000
PEAK_THRESHOLD = 0.5

# checkpoint 根目录
CKPT_ROOT = f"checkpoints/depth_{DEPTH}_embed_dim_{EMBED}"

# 创建实验目录
EXP_DIR = "exp/ablation_comparison"
os.makedirs(EXP_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# baseline_hard 作为 baseline，baseline_soft 不展示
ABLATION_MODELS = [
    {"name": "baseline", "label": "Baseline",         "color": "#FF0000", "marker": "*"},
    {"name": "no_layerscale", "label": "No LayerScale",   "color": "#0072BD", "marker": "o"},
    {"name": "no_droppath",   "label": "No DropPath",     "color": "#D95319", "marker": "s"},
    {"name": "fixed_droppath","label": "Fixed DropPath",  "color": "#77AC30", "marker": "^"},
    {"name": "mean_pooling",  "label": "Mean Pooling",    "color": "#7E2F8E", "marker": "v"},
    {"name": "pair_aware_pe", "label": "Pair-aware PE",   "color": "#A2142F", "marker": "D"},
    {"name": "lag_aware_pe",  "label": "Lag-aware PE",    "color": "#EDB120", "marker": "p"},
    {"name": "linear_head",   "label": "Linear Head",     "color": "#4DBEEE", "marker": "h"},
]

# 固定信源配置
FIXED_SOURCES = [
    {"angles": [-13.15, -5.21, 8.33], "label": "ThreeTargets"},
]

def load_ablation_model(model_path):
    """从checkpoint直接加载模型"""
    print(f"正在加载模型: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    model_config = checkpoint["model_config"]
    print(f"从checkpoint加载模型配置: {model_config}")

    model = CMAFormer(**model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, model_config

def load_models():
    """加载所有消融模型"""
    print("=== 加载模型 ===")
    models = {}

    for config in ABLATION_MODELS:
        model_name = config["name"]
        model_path = os.path.join(CKPT_ROOT, model_name, "best_model.pt")

        try:
            model, model_config = load_ablation_model(model_path)
            models[model_name] = {
                "model": model,
                "config": config,
                "model_config": model_config,
                "loaded": True
            }
            print(f"✅ {model_name} 加载成功")
        except Exception as e:
            models[model_name] = {
                "model": None,
                "config": config,
                "loaded": False,
                "error": str(e)
            }
            print(f"❌ {model_name} 加载失败: {e}")

    return models

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

    # 与训练一致：3通道
    r = np.stack([r_complex.real, r_complex.imag, np.sin(np.angle(r_complex)), np.cos(np.angle(r_complex))], axis=-1)
    r_tensor = torch.tensor(r, dtype=torch.float32).unsqueeze(0).to(device)
    return r_tensor

def get_model_prediction(model, X):
    """获取模型预测"""
    X_tensor = preprocess_for_proposed(X)

    with torch.no_grad():
        spectrum = model(X_tensor).cpu().numpy().squeeze()

    return spectrum

def extract_peaks(spectrum, num_sources, threshold=0.0):
    """从谱图中提取指定数量的峰值角度"""
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
    return sorted(float(THETA_GRID[i]) for i in peak_indices)

def calculate_rmse(true_angles, pred_angles):
    """计算角度估计的均方根误差(RMSE)"""
    if len(pred_angles) != len(true_angles):
        return 180.0

    squared_errors = [(true - pred) ** 2 for true, pred in zip(true_angles, pred_angles)]
    return np.sqrt(np.mean(squared_errors))

def evaluate_ablation_models(models, test_data):
    """评估所有消融模型性能"""
    print("\n=== 评估 one-by-one ablation 模型性能 ===")

    all_results = {}

    # 初始化结果存储
    for model_name in models.keys():
        if models[model_name]["loaded"]:
            all_results[model_name] = {}
            for config_idx in range(len(FIXED_SOURCES)):
                all_results[model_name][config_idx] = []

    # 对每个配置进行评估
    for config_idx, config in enumerate(FIXED_SOURCES):
        angles = config["angles"]
        num_sources = len(angles)
        label = config["label"]

        print(f"\n评估配置: {label} (角度: {angles})")

        # 对每个SNR进行评估
        for snr_idx, snr in enumerate(SNR_RANGE):
            print(f"  SNR: {snr}dB", end="")

            model_rmse_results = {}
            for model_name, model_info in models.items():
                if model_info["loaded"]:
                    model_rmse_results[model_name] = []

            # 使用同一批测试数据评估所有模型
            for trial in range(MC_TRIALS):
                X = test_data[config_idx][snr_idx][trial]

                for model_name, model_info in models.items():
                    if model_info["loaded"]:
                        model = model_info["model"]
                        spectrum = get_model_prediction(model, X)
                        pred_angles = extract_peaks(spectrum, num_sources, threshold=PEAK_THRESHOLD)
                        rmse = calculate_rmse(sorted(angles), sorted(pred_angles))
                        model_rmse_results[model_name].append(rmse)

            # 计算当前SNR下平均RMSE
            for model_name in model_rmse_results.keys():
                avg_rmse = np.mean(model_rmse_results[model_name])
                all_results[model_name][config_idx].append(avg_rmse)

            rmse_str = " | ".join([
                f"{models[name]['config']['label']}: {np.mean(model_rmse_results[name]):.2f}°"
                for name in model_rmse_results.keys()
            ])
            print(f" | {rmse_str}")

    return all_results

def plot_ablation_comparison(all_results, condition_idx):
    """绘制单个条件下所有消融模型的性能对比"""
    config = FIXED_SOURCES[condition_idx]
    label = config["label"]

    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.15)

    for model_cfg in ABLATION_MODELS:
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

    plt.xlabel('SNR (dB)', fontfamily='Times New Roman', fontsize=20, fontweight='bold', labelpad=10)
    plt.ylabel('RMSE (degree)', fontfamily='Times New Roman', fontsize=20, fontweight='bold', labelpad=10)
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.3, which='both')
    plt.xlim([SNR_RANGE[0], SNR_RANGE[-1]])
    plt.xticks(SNR_RANGE[::2])

    ax = plt.gca()
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
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
        prop={'family': 'Times New Roman', 'size': 11}
    )

    plt.savefig(f'{EXP_DIR}/ablation_comparison_{label}.eps', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("=== 开始 one-by-one ablation Monte Carlo 实验 ===")
    print(f"展示模型: {[cfg['name'] for cfg in ABLATION_MODELS]}")
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

        # 2. 生成统一测试集
        test_data = generate_test_dataset()

        # 3. 评估模型
        all_results = evaluate_ablation_models(loaded_models, test_data)

        print("\n=== 生成分析图表 ===")

        # 4. 绘图
        for condition_idx in range(len(FIXED_SOURCES)):
            plot_ablation_comparison(all_results, condition_idx)
            config_name = FIXED_SOURCES[condition_idx]["label"]
            print(f"✅ {config_name} 性能对比图已保存")

    except Exception as e:
        print(f"❌ 运行时出现错误: {e}")
        import traceback
        traceback.print_exc()