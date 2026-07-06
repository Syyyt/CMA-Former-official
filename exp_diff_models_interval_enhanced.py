import os
import gpu_manager
gpu_manager.auto_select_gpu()
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from data import generate_received_signal
from exp_utils.load_model import load_model, load_cnn_model, load_dct_model, load_dnn_model, load_hmc_model
from exp_utils.music import music_spectrum

# 配置参数
THETA_GRID = np.arange(-60, 61, 1)  # 角度网格
N = 16                              # 天线数
d = 0.5                             # 天线间距

DEPTH = 4                           # 指定要加载的模型depth
EMBED = 128
MODEL_PATH = f"checkpoints/depth_{DEPTH}_embed_dim_{EMBED}/best_model.pt"
MC_TRIALS = 2000                     # 蒙特卡洛实验次数
FIXED_SNR = -10                       # 固定信噪比为0dB
snapshot_num = 1000
SUCCESS_THRESHOLD = 0.5             # 判定阈值(度); 每个配对角度误差都 < 此值才算成功

# Times New Roman 
# 设置全局字体为Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'
#matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建实验子目录
EXP_DIR = "exp/angle_interval_analysis"
os.makedirs(EXP_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义不同角度间隔的信源配置 - 避开离格误差，所有角度都在网格点上
ANGLE_INTERVAL_CONFIGS = [
    #{"angles": [-30, 30], "interval": 60, "label": "TwoSources_60deg", "type": "two_sources"},
    #{"angles": [-25, 25], "interval": 50, "label": "TwoSources_50deg", "type": "two_sources"},
    #{"angles": [-20, 20], "interval": 40, "label": "TwoSources_40deg", "type": "two_sources"},
    #{"angles": [-15, 15], "interval": 30, "label": "TwoSources_30deg", "type": "two_sources"},
    #{"angles": [-12, 12], "interval": 24, "label": "TwoSources_24deg", "type": "two_sources"},
    #{"angles": [-10, 10], "interval": 20, "label": "TwoSources_20deg", "type": "two_sources"},
    #{"angles": [-8, 8], "interval": 16, "label": "TwoSources_16deg", "type": "two_sources"},
    #{"angles": [-6, 6], "interval": 12, "label": "TwoSources_12deg", "type": "two_sources"},
    #{"angles": [-5.1, 4.9], "interval": 10, "label": "TwoSources_10deg", "type": "two_sources"},
    #{"angles": [-4.1, 3.9], "interval": 8, "label": "TwoSources_8deg", "type": "two_sources"},
    #{"angles": [-3.1, 2.9], "interval": 6, "label": "TwoSources_6deg", "type": "two_sources"},
    #{"angles": [-3.1, 1.9], "interval": 5, "label": "TwoSources_6deg", "type": "two_sources"},
    #{"angles": [-2.1, 1.9], "interval": 4, "label": "TwoSources_4deg", "type": "two_sources"},
    #{"angles": [-1.1, 1.9], "interval": 3, "label": "TwoSources_4deg", "type": "two_sources"},
    {"angles": [-1.1, 0.9], "interval": 2, "label": "TwoSources_2deg", "type": "two_sources"},
    # MC_TRIALS = 2000  FIXED_SNR = -16  snapshot_num = 1000

    # 三信源情况 - 避开0.5度离格误差
    #{"angles": [-20, 0, 20], "interval": 20, "label": "ThreeSources_20deg", "type": "three_sources"},
    #{"angles": [-15, 0, 15], "interval": 15, "label": "ThreeSources_15deg", "type": "three_sources"},
    #{"angles": [-12, 0, 12], "interval": 12, "label": "ThreeSources_12deg", "type": "three_sources"},
    {"angles": [-10.1, -0.1, 9.9], "interval": 10, "label": "ThreeSources_10deg", "type": "three_sources"},
    {"angles": [-8.1, -0.1, 7.9], "interval": 8, "label": "ThreeSources_8deg", "type": "three_sources"},
    {"angles": [-6.1, -0.1, 5.9], "interval": 6, "label": "ThreeSources_6deg", "type": "three_sources"},
    {"angles": [-5.1, -0.1, 4.9], "interval": 5, "label": "ThreeSources_5deg", "type": "three_sources"},
    {"angles": [-4.1, -0.1, 3.9], "interval": 4, "label": "ThreeSources_4deg", "type": "three_sources"},
    {"angles": [-3.1, -0.1, 2.9], "interval": 3, "label": "ThreeSources_3deg", "type": "three_sources"},
    {"angles": [-2.1, -0.1, 1.9], "interval": 2, "label": "ThreeSources_2deg", "type": "three_sources"},
    #   MC_TRIALS = 2000    FIXED_SNR = -10     snapshot_num = 1000
    # 四信源情况 - 避开0.5度离格误差
    #{"angles": [-15, -5, 5, 15], "interval": 10, "label": "FourSources_10deg", "type": "four_sources"},
    #{"angles": [-12, -4, 4, 12], "interval": 8, "label": "FourSources_8deg", "type": "four_sources"},
    #{"angles": [-9, -3, 3, 9], "interval": 6, "label": "FourSources_6deg", "type": "four_sources"},
    #{"angles": [-6, -2, 2, 6], "interval": 4, "label": "FourSources_4deg", "type": "four_sources"},
    #{"angles": [-4, -1, 1, 4], "interval": 3, "label": "FourSources_3deg", "type": "four_sources"},  # 最小间隔1度
    #{"angles": [-3, -1, 1, 3], "interval": 2, "label": "FourSources_2deg", "type": "four_sources"},
    
    # 五信源情况 - 测试更高密度
    #{"angles": [-8, -4, 0, 4, 8], "interval": 4, "label": "FiveSources_4deg", "type": "five_sources"},
    #{"angles": [-6, -3, 0, 3, 6], "interval": 3, "label": "FiveSources_3deg", "type": "five_sources"},
    #{"angles": [-4, -2, 0, 2, 4], "interval": 2, "label": "FiveSources_2deg", "type": "five_sources"},
]

def generate_test_dataset():
    """
    预先生成所有测试数据，确保所有模型使用相同的数据集
    返回: test_data[config_idx][trial_idx] = X
    """
    print("=== 生成角度间隔测试数据集 ===")
    test_data = []
    
    for config_idx, config in enumerate(ANGLE_INTERVAL_CONFIGS):
        angles = config["angles"]
        label = config["label"]
        interval = config["interval"]
        print(f"生成 {label} (间隔: {interval}°) 的测试数据...")
        
        config_data = []
        for trial in range(MC_TRIALS):
            # 生成接收信号
            X = generate_received_signal(angles, SNR=FIXED_SNR, S=snapshot_num)
            config_data.append(X)
        test_data.append(config_data)
    
    print("角度间隔测试数据集生成完成")
    return test_data

def preprocess_for_proposed(X):
    M, N = np.shape(X)
    R = (X @ X.T.conj()) / N
    # 提取上三角非对角元素
    r_complex = []
    for i in range(M):
        for j in range(i+1, M):
            r_complex.append(R[i, j])
    
    r_complex = np.array(r_complex)
    
    # 分离实部和虚部
    r = np.stack([r_complex.real, r_complex.imag, np.sin(np.angle(r_complex)), np.cos(np.angle(r_complex))], axis=-1)
    r_tensor = torch.tensor(r, dtype=torch.float32).unsqueeze(0).to(device)
    return r_tensor

def preprocess_for_cnn(X):
    M, N = np.shape(X)
    R = (X @ X.T.conj()) / N
    R_real = np.real(R)
    R_imag = np.imag(R)
    R_phase = np.angle(R)
    R_combined = np.stack([R_real, R_imag, R_phase], axis=0)
    R_tensor = torch.tensor(R_combined, dtype=torch.float32).unsqueeze(0).to(device)
    return R_tensor

def preprocess_for_dct(X):
    M, N = np.shape(X)
    R = (X @ X.T.conj()) / N
    R_real = np.real(R)
    R_imag = np.imag(R)
    R_phase = np.angle(R)
    R_combined = np.stack([R_real, R_imag, R_phase], axis=0)
    R_tensor = torch.tensor(R_combined, dtype=torch.float32).unsqueeze(0).to(device)
    return R_tensor

def preprocess_for_dnn(X):
    M, N = np.shape(X)
    R = (X @ X.T.conj()) / N
    # 提取上三角非对角元素
    r_complex = []
    for i in range(M):
        for j in range(i+1, M):
            r_complex.append(R[i, j])
    
    r_complex = np.array(r_complex)
    
    r = np.concatenate([r_complex.real, r_complex.imag])
    r = r / np.linalg.norm(r)
    r_tensor = torch.tensor(r, dtype=torch.float32).unsqueeze(0).to(device)
    
    return r_tensor 

def preprocess_for_hmc(X):
    """
    HMC-ViT的输入: 协方差矩阵上三角的实部、虚部和相位, 形状 3 x 1 x 120
    """
    M, N = np.shape(X)
    R = (X @ X.T.conj()) / N
    # 提取上三角非对角元素
    r_complex = []
    for i in range(M):
        for j in range(i+1, M):
            r_complex.append(R[i, j])
    r_complex = np.array(r_complex)

    r_real = r_complex.real
    r_imag = r_complex.imag
    r_phase = np.angle(r_complex)

    r_stack = np.stack([r_real, r_imag, r_phase], axis=0)  # 3 x 120
    r_stack = r_stack[:, np.newaxis, :]                    # 3 x 1 x 120
    r_tensor = torch.tensor(r_stack, dtype=torch.float32).unsqueeze(0).to(device)  # 1 x 3 x 1 x 120
    return r_tensor

def get_model_prediction(model, X, model_name):
    """根据模型类型进行相应的预处理和预测"""
    if model_name == "MUSIC":
        # MUSIC算法: 非网络模型, 直接调用
        spectrum = music_spectrum(X, num_sources=current_num_sources, theta_grid=THETA_GRID, N=N, d=d)
        return spectrum

    if model_name == "Proposed":
        X_tensor = preprocess_for_proposed(X)
    elif model_name == "CNN":
        X_tensor = preprocess_for_cnn(X)
    elif model_name == "DCT-ViT":
        X_tensor = preprocess_for_dct(X)
    elif model_name == "DNN":
        X_tensor = preprocess_for_dnn(X)
    elif model_name == "HMC-ViT":
        X_tensor = preprocess_for_hmc(X)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    with torch.no_grad():
        if model_name == "DNN":
            spectrum = model(X_tensor)[0]
            spectrum = spectrum.cpu().numpy().squeeze()
        elif model_name == "HMC-ViT":
            logits = model(X_tensor)
            spectrum = torch.sigmoid(logits).cpu().numpy().squeeze()
        else :
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

def calculate_success(true_angles, pred_angles, threshold=SUCCESS_THRESHOLD):
    """
    判断本次 DOA 估计是否成功。
    条件: 估计出的角度数量与真实一致, 且排序配对后最大绝对误差 < threshold。
    返回: 1 (成功) 或 0 (失败)
    """
    if len(pred_angles) != len(true_angles):
        return 0
    true_sorted = sorted(true_angles)
    pred_sorted = sorted(pred_angles)
    max_err = max(abs(t - p) for t, p in zip(true_sorted, pred_sorted))
    return 1 if max_err < threshold else 0

def evaluate_models_on_angle_intervals(models, test_data):
    """
    在预生成的角度间隔数据集上评估所有模型
    参数:
        models: 模型字典 {model_name: model}
        test_data: 预生成的测试数据 [config_idx][trial_idx]
    返回:
        all_results: {model_name: [success_rate_per_config]}
    """
    global current_num_sources
    print("\n=== 在角度间隔数据集上评估所有模型 ===")
    
    all_results = {}
    
    # 初始化结果存储
    for model_name in models.keys():
        all_results[model_name] = []
    
    # 对每个配置进行评估
    for config_idx, config in enumerate(ANGLE_INTERVAL_CONFIGS):
        angles = config["angles"]
        num_sources = len(angles)
        current_num_sources = num_sources  # 供MUSIC使用
        label = config["label"]
        interval = config["interval"]
        
        print(f"\n评估配置: {label} (角度: {angles}, 间隔: {interval}°)")
        
        # 为每个模型计算在当前配置下的性能
        model_success_results = {}
        for model_name in models.keys():
            model_success_results[model_name] = []
        
        # 在所有试验中评估（使用相同的数据）
        for trial in range(MC_TRIALS):
            # 获取当前试验的数据（所有模型使用相同数据）
            X = test_data[config_idx][trial]
            
            # 让所有模型对相同数据进行预测
            for model_name, model in models.items():
                spectrum = get_model_prediction(model, X, model_name)
                pred_angles = extract_peaks(spectrum, num_sources)
                success = calculate_success(sorted(angles), sorted(pred_angles))
                model_success_results[model_name].append(success)
        
        # 计算每个模型在当前配置下的成功率
        for model_name in models.keys():
            success_rate = np.mean(model_success_results[model_name])
            all_results[model_name].append(success_rate)
        
        # 打印当前配置的结果
        rate_str = " | ".join([f"{name}: {np.mean(model_success_results[name])*100:.1f}%"
                              for name in models.keys()])
        print(f"  成功率: {rate_str}")
    
    return all_results

def plot_angle_interval_comparison(all_models_results, source_type):
    """为特定信源数量绘制角度间隔对比图 - 使用线性坐标"""
    # 筛选特定类型的配置
    filtered_configs = []
    filtered_results = {}
    
    for model_name in all_models_results.keys():
        filtered_results[model_name] = []
    
    for config_idx, config in enumerate(ANGLE_INTERVAL_CONFIGS):
        if config["type"] == source_type:
            filtered_configs.append(config)
            for model_name in all_models_results.keys():
                filtered_results[model_name].append(all_models_results[model_name][config_idx])
    
    if not filtered_configs:
        return
    
    # 提取角度间隔作为x轴
    intervals = [config["interval"] for config in filtered_configs]
    
    width = 8
    height = 6
    plt.figure(figsize=(width, height))
    plt.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.15)
    
    colors = ['#0072BD', '#D95319', '#77AC30', '#7E2F8E', '#EDB120', '#A2142F']
    markers = ['o', 's', '^', 'v', 'D', 'P']
    linestyles = ['-', '--', '-.', ':', '-', '--']
    
    # 绘制各模型的性能曲线 - 使用线性坐标
    for i, (model_name, success_vals) in enumerate(filtered_results.items()):
        plt.plot(intervals, success_vals,
                markers[i] + linestyles[i], 
                label=model_name, 
                color=colors[i],
                markerfacecolor='white', 
                markeredgewidth=2, 
                markersize=8,
                linewidth=2)
    
    plt.xlabel('Angular Separation (degree)', fontfamily='Times New Roman', fontsize=20, fontweight='bold', labelpad=10)
    plt.ylabel('Success Rate', fontfamily='Times New Roman', fontsize=20, fontweight='bold', labelpad=10)
    
    # 使用线性坐标
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.ylim(-0.02, 1.02)
    
    # 设置标题
    source_num = source_type.split('_')[0].capitalize()
    #plt.title(f'{source_num} Angular Resolution Performance (SNR = {FIXED_SNR}dB)', 
    #         fontfamily='Times New Roman', fontsize=16, fontweight='bold', pad=15)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.yaxis.set_tick_params(pad=10)
    
    legend = plt.legend(
        loc='lower right',
        #bbox_to_anchor=(0.05, 0.95),
        frameon=True,
        framealpha=0.95,
        edgecolor='gray',
        fancybox=True,
        shadow=True,
        prop={'family': 'Times New Roman', 'size': 18}
    )
    
    plt.savefig(f'{EXP_DIR}/success_rate_{source_type}_thr{SUCCESS_THRESHOLD}.eps', dpi=300, bbox_inches='tight')
    plt.close()

def plot_combined_angle_interval_comparison(all_models_results):
    """绘制所有信源数量的综合对比图 - 使用线性坐标"""
    # 按信源类型分组
    source_types = ["two_sources", "three_sources", "four_sources", "five_sources"]
    type_labels = ["Two Sources", "Three Sources", "Four Sources", "Five Sources"]
    
    # 检查哪些类型有数据
    available_types = []
    available_labels = []
    for source_type, type_label in zip(source_types, type_labels):
        if any(config["type"] == source_type for config in ANGLE_INTERVAL_CONFIGS):
            available_types.append(source_type)
            available_labels.append(type_label)
    
    n_plots = len(available_types)
    if n_plots == 0:
        return
    
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 6))
    if n_plots == 1:
        axes = [axes]
    
    fig.suptitle(f'Angular Resolution Performance Comparison (SNR = {FIXED_SNR}dB)', 
                fontfamily='Times New Roman', fontsize=16, fontweight='bold')
    
    colors = ['#0072BD', '#D95319', '#77AC30', '#7E2F8E', '#EDB120', '#A2142F']
    markers = ['o', 's', '^', 'v', 'D', 'P']
    linestyles = ['-', '--', '-.', ':', '-', '--']
    
    for subplot_idx, (source_type, type_label) in enumerate(zip(available_types, available_labels)):
        ax = axes[subplot_idx]
        
        # 筛选特定类型的配置
        filtered_configs = []
        filtered_results = {}
        
        for model_name in all_models_results.keys():
            filtered_results[model_name] = []
        
        for config_idx, config in enumerate(ANGLE_INTERVAL_CONFIGS):
            if config["type"] == source_type:
                filtered_configs.append(config)
                for model_name in all_models_results.keys():
                    filtered_results[model_name].append(all_models_results[model_name][config_idx])
        
        if not filtered_configs:
            continue
        
        # 提取角度间隔作为x轴
        intervals = [config["interval"] for config in filtered_configs]
        
        # 绘制各模型的性能曲线
        for i, (model_name, success_vals) in enumerate(filtered_results.items()):
            ax.plot(intervals, success_vals,
                    markers[i] + linestyles[i], 
                    label=model_name if subplot_idx == 0 else "", 
                    color=colors[i],
                    markerfacecolor='white', 
                    markeredgewidth=2, 
                    markersize=8,
                    linewidth=2)
        
        ax.set_xlabel('Angular Separation (°)', fontfamily='Times New Roman', fontsize=20, fontweight='bold')
        if subplot_idx == 0:
            ax.set_ylabel('Success Rate', fontfamily='Times New Roman', fontsize=20, fontweight='bold')
        
        # 使用线性坐标
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_title(type_label, fontfamily='Times New Roman', fontsize=14, fontweight='bold')
        
        # 设置x轴刻度
        major_intervals = [i for i in intervals if i % 5 == 0 or i <= 5]
        if not major_intervals:
            major_intervals = intervals[::2] if len(intervals) > 8 else intervals
        ax.set_xticks(major_intervals)
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        
        # 设置y轴范围为 [0, 1]
        ax.set_ylim(-0.02, 1.02)
        
        # 美化坐标轴
        ax.spines['top'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
    
    # 添加图例
    if len(axes) > 0:
        axes[0].legend(
            loc='lower right',
            frameon=True,
            framealpha=0.95,
            edgecolor='gray',
            fancybox=True,
            shadow=True,
            prop={'family': 'Times New Roman', 'size': 18}
        )
    
    plt.tight_layout()
    plt.savefig(f'{EXP_DIR}/success_rate_combined_thr{SUCCESS_THRESHOLD}.eps', dpi=300, bbox_inches='tight')
    plt.close()

def generate_spectrum_examples(models, test_data):
    """生成不同角度间隔下的谱图示例"""
    global current_num_sources
    # 选择几个代表性的配置进行谱图展示
    example_configs = [
        {"type": "two_sources", "intervals": [20, 8, 2]},
        {"type": "three_sources", "intervals": [15, 6, 2]},
        {"type": "four_sources", "intervals": [8, 4, 2]}
    ]
    
    colors = ['#0072BD', '#D95319', '#77AC30', "#BD00AD"]
    
    for example in example_configs:
        source_type = example["type"]
        intervals = example["intervals"]
        
        for interval in intervals:
            # 找到对应的配置
            config_idx = None
            for idx, config in enumerate(ANGLE_INTERVAL_CONFIGS):
                if config["type"] == source_type and config["interval"] == interval:
                    config_idx = idx
                    break
            
            if config_idx is None:
                continue
            
            config = ANGLE_INTERVAL_CONFIGS[config_idx]
            angles = config["angles"]
            label = config["label"]
            current_num_sources = len(angles)  # 供MUSIC使用
            
            # 使用测试数据集中的第一个样本
            X = test_data[config_idx][0]
            
            for i, (model_name, model) in enumerate(models.items()):
                # 为每个模型创建独立的图片
                plt.figure(figsize=(8, 6))
                
                spectrum = get_model_prediction(model, X, model_name)
                
                if model_name == 'DNN':
                    GRID = np.arange(-60, 60, 1)
                else:
                    GRID = np.arange(-60, 61, 1)
                    
                plt.plot(GRID, spectrum, '-', color=colors[0], linewidth=2.5, label='Predicted Spectrum')
                
                for angle in angles:
                    plt.axvline(x=angle, color='#D95319', linestyle='--', linewidth=2, alpha=0.8, 
                               label='True DOA' if angle == angles[0] else "")
                
                plt.xlabel(r'Angle ($^\circ$)', fontfamily='Times New Roman', fontsize=20, fontweight='bold', labelpad=8)
                plt.ylabel('Normalized Spectrum', fontfamily='Times New Roman', fontsize=20, fontweight='bold', labelpad=8)
                
                #plt.title(f'{model_name} - {label} (Δθ={interval}°)', 
                #         fontfamily='Times New Roman', fontsize=16, fontweight='bold', pad=15)
                plt.grid(True, linestyle='--', alpha=0.3)
                
                ax = plt.gca()
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_linewidth(1.5)
                ax.spines['bottom'].set_linewidth(1.5)
                
                plt.xlim(-60, 60)
                
                legend = plt.legend(
                    loc='upper right',
                    frameon=True,
                    framealpha=0.95,
                    edgecolor='gray',
                    fancybox=True,
                    shadow=True,
                    prop={'family': 'Times New Roman', 'size': 18}
                )
                
                # 保存独立的图片
                plt.savefig(f'{EXP_DIR}/spectrum_{model_name}_{label}.eps', 
                           dpi=300, bbox_inches='tight')
                plt.close()

# 全局变量: 当前配置的信源数量, 供MUSIC使用
current_num_sources = 1

if __name__ == "__main__":
    print("=== 开始角度间隔分析实验（避开离格误差）===")

    try:
        # 加载所有模型
        print("=== 加载模型 ===")
        proposed_model, proposed_model_config = load_model(model_path=MODEL_PATH)
        print(f"Proposed模型加载成功")
        
        cnn_model = load_cnn_model()
        print(f"CNN模型加载成功")

        dct_vit_model = load_dct_model()
        print(f"DCT-ViT模型加载成功")

        dnn_model = load_dnn_model()
        print(f"DNN模型加载成功")

        hmc_vit_model = load_hmc_model()
        print(f"HMC-ViT模型加载成功")

        models = {
            "Proposed": proposed_model,
            "CNN": cnn_model,
            "DCT-ViT": dct_vit_model,
            "HMC-ViT": hmc_vit_model,
            "DNN": dnn_model,
            "MUSIC": None  # MUSIC为传统算法, 无需加载模型
        }
        
        # 验证角度配置（确保没有离格误差）
        print("\n=== 验证角度配置 ===")
        for config in ANGLE_INTERVAL_CONFIGS:
            angles = config["angles"]
            for angle in angles:
                if angle not in THETA_GRID:
                    print(f"警告: 角度 {angle} 不在网格点上！")
                else:
                    print(f"✓ 配置 {config['label']}: 角度 {angles} 都在网格点上")
            break  # 只显示第一个作为示例
        
        # 预先生成所有测试数据
        test_data = generate_test_dataset()
        
        # 在固定数据集上评估所有模型
        all_models_results = evaluate_models_on_angle_intervals(models, test_data)
        
        print("\n=== 生成角度间隔分析图表 ===")
        
        # 保存详细结果
        for model_name, results in all_models_results.items():
            filename = f"{EXP_DIR}/success_rate_{model_name}_angle_intervals.txt"
            with open(filename, "w") as f:
                f.write(f"# Threshold = {SUCCESS_THRESHOLD} deg\n")
                f.write("Config\tInterval(deg)\tAngles\tSuccessRate\n")
                for config_idx, sr in enumerate(results):
                    config = ANGLE_INTERVAL_CONFIGS[config_idx]
                    f.write(f"{config['label']}\t{config['interval']}\t{config['angles']}\t{sr:.4f}\n")
        
        # 为每种信源数量单独绘制角度间隔对比图
        source_types = ["two_sources", "three_sources", "four_sources", "five_sources"]
        for source_type in source_types:
            # 检查是否有该类型的数据
            has_data = any(config["type"] == source_type for config in ANGLE_INTERVAL_CONFIGS)
            if has_data:
                plot_angle_interval_comparison(all_models_results, source_type)
                print(f"{source_type} 角度间隔对比图已保存")
        
        # 绘制综合对比图
        plot_combined_angle_interval_comparison(all_models_results)
        print("综合角度间隔对比图已保存")
        
        # 生成谱图示例
        generate_spectrum_examples(models, test_data)
        print("谱图示例已保存")
        
        print("\n=== 角度间隔分析实验完成 ===")
        print(f"\n所有结果文件已保存到 {EXP_DIR}/ 目录")
        
        # 输出关键发现总结
        print("\n=== 关键发现总结 ===")
        source_types = ["two_sources", "three_sources", "four_sources", "five_sources"]
        for source_type in source_types:
            type_configs = [config for config in ANGLE_INTERVAL_CONFIGS if config["type"] == source_type]
            if type_configs:
                print(f"\n{source_type.replace('_', ' ').title()}:")
                min_interval = min(config["interval"] for config in type_configs)
                max_interval = max(config["interval"] for config in type_configs)
                print(f"  测试角度间隔范围: {min_interval}° - {max_interval}°")
                
                # 找到最小间隔下的性能
                min_interval_idx = None
                for idx, config in enumerate(ANGLE_INTERVAL_CONFIGS):
                    if config["type"] == source_type and config["interval"] == min_interval:
                        min_interval_idx = idx
                        break
                
                if min_interval_idx is not None:
                    print(f"  最小间隔({min_interval}°)下的成功率:")
                    for model_name, results in all_models_results.items():
                        sr = results[min_interval_idx]
                        print(f"    {model_name}: {sr*100:.1f}%")
                    print(f"  对应角度: {ANGLE_INTERVAL_CONFIGS[min_interval_idx]['angles']}")

    except Exception as e:
        print(f"运行时出现错误: {e}")