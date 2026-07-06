import os
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from data import generate_received_signal
from exp_utils.load_model import load_model

# 设置matplotlib字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 配置参数
THETA_GRID = np.arange(-60, 61, 1)  # 角度网格
N = 16                              # 天线数
d = 0.5                             # 天线间距

# 不同num_heads配置的实验设置
HEAD_CONFIGS = [
    {"num_heads": [4, 4, 4], "name": "heads_4x3", "color": "#0072BD", "marker": "o"},
    {"num_heads": [8, 8, 8], "name": "heads_8x3", "color": "#D95319", "marker": "s"}, 
    {"num_heads": [16, 16, 16], "name": "heads_16x3", "color": "#77AC30", "marker": "^"},
    {"num_heads": [4], "name": "heads_4x1", "color": "#7E2F8E", "marker": "v"},
    {"num_heads": [8], "name": "heads_8x1", "color": "#A2142F", "marker": "D"},
    {"num_heads": [16], "name": "heads_16x1", "color": "#EDB120", "marker": "p"},
    {"num_heads": [4, 8, 16], "name": "heads_mix", "color": "#FF0000", "marker": "*"}
]

DEPTH = 4                          # 模型深度
EMBED = 128                        # 嵌入维度
MC_TRIALS = 1000                   # 蒙特卡洛实验次数
SNR_RANGE = np.arange(-20, 1, 2)  # 信噪比测试范围 [-20, -18, ..., 0]
snapshot_num = 1000

# 创建实验目录
EXP_DIR = "exp/heads_comparison"
os.makedirs(EXP_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 固定信源配置
FIXED_SOURCES = [
    {"angles": [-3.05, -1.01, 1.03], "label": "ThreeTargets"},
]

def load_head_models():
    """加载所有不同头数配置的模型"""
    print("=== 加载不同头数配置的模型 ===")
    models = {}
    
    for config in HEAD_CONFIGS:
        model_name = config["name"]
        model_path = f"checkpoints/{model_name}/best_model.pt"
        
        try:
            model, model_config = load_model(model_path=model_path)
            model.eval()
            models[model_name] = {
                "model": model,
                "config": config,
                "loaded": True
            }
            print(f"✅ {model_name} 加载成功 (num_heads: {config['num_heads']})")
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
                # 生成接收信号
                X = generate_received_signal(angles, SNR=snr, S=snapshot_num)
                snr_data.append(X)
            config_data.append(snr_data)
        test_data.append(config_data)
    
    print("测试数据集生成完成")
    return test_data

def preprocess_for_proposed(X):
    """预处理函数（与原始训练一致）"""
    M, N = np.shape(X)
    R = (X @ X.T.conj()) / N
    
    # 提取上三角非对角元素
    r_complex = []
    for i in range(M):
        for j in range(i+1, M):
            r_complex.append(R[i, j])
    
    r_complex = np.array(r_complex)
    
    # 分离实部和虚部
    r = np.stack([r_complex.real, r_complex.imag, np.angle(r_complex)], axis=-1)
    r_tensor = torch.tensor(r, dtype=torch.float32).unsqueeze(0).to(device)
    return r_tensor

def get_model_prediction(model, X):
    """获取模型预测"""
    X_tensor = preprocess_for_proposed(X)
    
    with torch.no_grad():
        spectrum = model(X_tensor).cpu().numpy().squeeze()
    
    return spectrum

def extract_peaks(spectrum, num_sources, threshold=0.):
    """
    从谱图中提取指定数量的峰值角度
    """
    candidate_indices = []
    for i in range(len(spectrum)):
        if spectrum[i] > threshold:
            if (i == 0 or spectrum[i] > spectrum[i-1]) and \
               (i == len(spectrum)-1 or spectrum[i] > spectrum[i+1]):
                candidate_indices.append(i)
    
    if not candidate_indices:
        return []
    
    candidate_indices.sort(key=lambda i: spectrum[i], reverse=True)
    peak_indices = candidate_indices[:num_sources]
    return sorted(THETA_GRID[i] for i in peak_indices)

def calculate_rmse(true_angles, pred_angles):
    """计算角度估计的均方根误差(RMSE)"""
    if len(pred_angles) != len(true_angles):
        return 180.0
    
    squared_errors = [(true - pred)**2 for true, pred in zip(true_angles, pred_angles)]
    return np.sqrt(np.mean(squared_errors))

def evaluate_head_models(models, test_data):
    """
    评估所有头数配置的模型性能
    """
    print("\n=== 评估不同头数配置的模型性能 ===")
    
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
            
            # 为每个模型计算在当前SNR下的性能
            model_rmse_results = {}
            for model_name, model_info in models.items():
                if model_info["loaded"]:
                    model_rmse_results[model_name] = []
            
            # 在所有试验中评估（使用相同的数据）
            for trial in range(MC_TRIALS):
                # 获取当前试验的数据（所有模型使用相同数据）
                X = test_data[config_idx][snr_idx][trial]
                
                # 让所有模型对相同数据进行预测
                for model_name, model_info in models.items():
                    if model_info["loaded"]:
                        model = model_info["model"]
                        spectrum = get_model_prediction(model, X)
                        pred_angles = extract_peaks(spectrum, num_sources)
                        rmse = calculate_rmse(sorted(angles), sorted(pred_angles))
                        model_rmse_results[model_name].append(rmse)
            
            # 计算每个模型在当前SNR下的平均RMSE
            for model_name in model_rmse_results.keys():
                avg_rmse = np.mean(model_rmse_results[model_name])
                all_results[model_name][config_idx].append(avg_rmse)
            
            # 打印当前SNR的结果
            rmse_str = " | ".join([f"{name}: {np.mean(model_rmse_results[name]):.2f}°" 
                                  for name in model_rmse_results.keys()])
            print(f" | {rmse_str}")
    
    return all_results

def calculate_crb(angles, snr_db, N=16, d=0.5, S=snapshot_num):
    """计算CRB理论下界"""
    angles_rad = np.array(angles) * np.pi / 180
    K = len(angles)
    snr_linear = 10**(snr_db / 10)
    noise_var = 1.0
    
    # 构建导向矩阵和导数矩阵
    n = np.arange(N)
    A = np.exp(1j * 2 * np.pi * d * np.outer(np.sin(angles_rad), n)).T
    
    # 计算导数矩阵
    D = np.zeros((N, K), dtype=complex)
    for k in range(K):
        D[:, k] = 1j * 2 * np.pi * d * np.cos(angles_rad[k]) * n * A[:, k]
    
    # 投影矩阵 
    P_A_perp = np.eye(N) - A @ np.linalg.pinv(A)
    
    # Fisher信息矩阵 
    H = D.conj().T @ P_A_perp @ D
    P_est = snr_linear * np.eye(K)
    FIM = 2 * S / noise_var * np.real(H * P_est.T)
    
    # CRB = FIM^(-1) 的对角元素
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

def plot_head_comparison(all_results, condition_idx):
    """绘制单个条件下不同头数配置的性能对比"""
    config = FIXED_SOURCES[condition_idx]
    label = config['label']
    
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.15)
    
    # 绘制各头数配置的性能曲线
    for model_name, results in all_results.items():
        config_info = HEAD_CONFIGS[[c["name"] for c in HEAD_CONFIGS].index(model_name)]
        rmse_vals = results[condition_idx]
        
        plt.plot(SNR_RANGE, rmse_vals, 
                config_info["marker"] + '-', 
                label=f"{model_name} ({config_info['num_heads']})", 
                color=config_info["color"],
                markerfacecolor='white', 
                markeredgewidth=2, 
                markersize=8,
                linewidth=2)
    
    # 绘制CRB曲线
    """plt.plot(SNR_RANGE, crb_results[condition_idx],
            label='CRB', 
            color='black',
            linestyle='--',
            linewidth=2)"""
    
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
    
    legend = plt.legend(
        loc='best',
        frameon=True,
        framealpha=0.95,
        edgecolor='gray',
        fancybox=True,
        shadow=True,
        prop={'family': 'Times New Roman', 'size': 12}
    )
    
    plt.savefig(f'{EXP_DIR}/heads_comparison_{label}.eps', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    print("=== 开始不同头数配置模型性能对比实验 ===")
    print(f"测试的头数配置: {[config['name'] for config in HEAD_CONFIGS]}")
    print(f"测试场景: {[config['label'] for config in FIXED_SOURCES]}")
    print(f"SNR范围: {SNR_RANGE[0]}dB 到 {SNR_RANGE[-1]}dB")
    print(f"蒙特卡洛试验次数: {MC_TRIALS}")
    
    try:
        # 1. 加载所有不同头数配置的模型
        models = load_head_models()
        
        # 检查是否有成功加载的模型
        loaded_models = {name: info for name, info in models.items() if info["loaded"]}
        if not loaded_models:
            print("没有成功加载任何模型，请检查模型路径")
            exit(1)
        
        print(f"\n 成功加载 {len(loaded_models)} 个模型")
        
        # 2. 预先生成所有测试数据
        test_data = generate_test_dataset()
        
        # 4. 评估所有模型性能
        all_results = evaluate_head_models(loaded_models, test_data)
        
        print("\n=== 生成分析图表和报告 ===")
        
        # 5. 为每个测试条件绘制性能对比图
        for condition_idx in range(len(FIXED_SOURCES)):
            plot_head_comparison(all_results, condition_idx)
            config_name = FIXED_SOURCES[condition_idx]['label']
            print(f"✅ {config_name} 性能对比图已保存")
        
    except Exception as e:
        print(f"❌ 运行时出现错误: {e}")
        import traceback
        traceback.print_exc()
