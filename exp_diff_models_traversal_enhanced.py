# exp_diff_models_traversal_enhanced.py
import os
import gpu_manager
gpu_manager.auto_select_gpu()
import torch
import numpy as np
import matplotlib.pyplot as plt
from data import generate_received_signal
from exp_utils.load_model import load_model, load_cnn_model, load_dct_model, load_dnn_model, load_hmc_model
from exp_utils.music import music_spectrum

# 设置matplotlib字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['legend.fontsize'] = 34
plt.rcParams['xtick.labelsize'] = 32
plt.rcParams['ytick.labelsize'] = 32

# ==================== 可配置参数区域 ====================
# 角度扫描配置
ANGLE_START = -60           # 起始角度
ANGLE_END = 60              # 结束角度  
ANGLE_STEP = 1              # 角度步长（可方便调整）

# 多目标间距配置
TARGET_SPACING = 3         # 多目标间的角度间距（可方便调整）

# 其他实验参数
N = 16                      # 天线数
d = 0.5                     # 天线间距
DEPTH = 4                   # 指定要加载的模型depth
EMBED = 128
MODEL_PATH = f"checkpoints/depth_{DEPTH}_embed_dim_{EMBED}/best_model.pt"
MC_TRIALS = 1              # 每个角度的蒙特卡洛试验次数
SNR = -10                   # 固定信噪比 (dB)
snapshot_num = 1000

# 可视化配置
COLORS = ['red', 'orange', 'green', 'purple', 'brown', 'pink']  # 目标颜色
MARKERS = ['o', 's', '^', 'D', 'v', '<']                       # 目标形状
MARKER_SIZES = [60, 50, 55, 45, 55, 50]                       # 标记大小
# ==================== 配置参数区域结束 ====================

# 基于可配置参数生成角度网格
THETA_GRID = np.arange(ANGLE_START, ANGLE_END + 1, ANGLE_STEP)

# 实验配置
ANGLE_CONFIGS = [
    {
        "name": "SingleTarget",
        "angles_range": np.arange(ANGLE_START, ANGLE_END + 1, ANGLE_STEP),
        "num_targets": 1,
        "label": "Single Target"
    },
    {
        "name": "DualTarget", 
        "angles_range": np.arange(ANGLE_START, ANGLE_END + 1, ANGLE_STEP),
        "num_targets": 2,
        "label": "Dual Target"
    },
    {
        "name": "TripleTarget",
        "angles_range": np.arange(ANGLE_START, ANGLE_END + 1, ANGLE_STEP),
        "num_targets": 3,
        "label": "Triple Target"
    }
]

# 创建实验子目录
EXP_DIR = "exp/angle_sweep"
os.makedirs(EXP_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        else:
            spectrum = model(X_tensor).cpu().numpy().squeeze()
    
    return spectrum

def extract_peaks(spectrum, num_sources, threshold=0.1):
    """
    从谱图中提取指定数量的峰值角度
    """
    # 找到所有局部最大值
    candidate_indices = []
    for i in range(len(spectrum)):
        if spectrum[i] > threshold:
            if (i == 0 or spectrum[i] >= spectrum[i-1]) and \
               (i == len(spectrum)-1 or spectrum[i] >= spectrum[i+1]):
                candidate_indices.append(i)
    
    if not candidate_indices:
        # 如果没有找到峰值，返回谱图最大值位置
        max_idx = np.argmax(spectrum)
        return [THETA_GRID[max_idx] if len(THETA_GRID) > max_idx else 0]
    
    # 按谱图值排序，选择前num_sources个峰值
    candidate_indices.sort(key=lambda i: spectrum[i], reverse=True)
    peak_indices = candidate_indices[:num_sources]
    
    # 转换为角度并排序
    peak_angles = [THETA_GRID[i] for i in peak_indices if i < len(THETA_GRID)]
    return sorted(peak_angles)

def generate_multi_target_angles(center_angle, num_targets, spacing=None):
    """
    为多目标场景生成角度组合
    center_angle: 中心角度
    num_targets: 目标数量
    spacing: 目标间角度间隔（如果为None，使用全局配置TARGET_SPACING）
    """
    if spacing is None:
        spacing = TARGET_SPACING
        
    if num_targets == 1:
        return [center_angle]
    elif num_targets == 2:
        # 双目标：center_angle ± spacing/2
        angle1 = center_angle - spacing // 2
        angle2 = center_angle + spacing // 2
        # 确保角度在有效范围内
        angles = [angle1, angle2]
        angles = [a for a in angles if ANGLE_START <= a <= ANGLE_END]
        return angles if len(angles) == 2 else None
    elif num_targets == 3:
        # 三目标：center_angle, center_angle ± spacing
        angles = [center_angle - spacing, center_angle, center_angle + spacing]
        angles = [a for a in angles if ANGLE_START <= a <= ANGLE_END]
        return angles if len(angles) == 3 else None
    else:
        raise ValueError(f"不支持的目标数量: {num_targets}")

def generate_test_dataset_for_config(config):
    """
    为特定配置预先生成所有测试数据，确保所有模型使用相同的数据集
    """
    print(f"=== 为 {config['label']} 生成测试数据集 ===")
    test_data = []
    valid_angle_configs = []
    
    for center_angle in config['angles_range']:
        # 生成当前角度配置的真实角度
        true_angles = generate_multi_target_angles(
            center_angle, 
            config['num_targets'], 
            spacing=TARGET_SPACING
        )
        
        if true_angles is None:
            continue  # 跳过无效的角度配置
        
        valid_angle_configs.append(true_angles)
        
        # 为当前角度配置生成测试数据
        angle_data = []
        for trial in range(MC_TRIALS):
            # 生成接收信号
            X = generate_received_signal(true_angles, SNR=SNR, S=snapshot_num)
            angle_data.append(X)
        
        test_data.append(angle_data)
    
    print(f"为 {config['label']} 生成了 {len(valid_angle_configs)} 个有效角度配置的测试数据")
    return test_data, valid_angle_configs

def run_angle_sweep_experiment(models, config):
    """
    运行角度扫描实验
    """
    global current_num_sources
    print(f"\n=== 开始 {config['label']} 角度扫描实验 ===")

    # 当前配置下的信源数量, 供MUSIC使用
    current_num_sources = config['num_targets']

    # 预先生成所有测试数据
    test_data, valid_angle_configs = generate_test_dataset_for_config(config)
    
    results = {}
    for model_name in models.keys():
        results[model_name] = {
            'true_angles': [],
            'estimated_angles': [],
            'angle_indices': [],
            'target_indices': []  
        }
    
    # 对每个有效的角度配置进行测试
    for angle_idx, (true_angles, angle_test_data) in enumerate(zip(valid_angle_configs, test_data)):
        print(f"测试角度: {true_angles} (索引: {angle_idx})")
        
        # 对每个模型使用相同的测试数据进行测试
        for model_name, model in models.items():
            trial_estimates = []
            
            # 对所有试验数据进行预测
            for trial_data in angle_test_data:
                # 获取模型预测
                spectrum = get_model_prediction(model, trial_data, model_name)
                
                # 提取峰值角度
                estimated_angles = extract_peaks(spectrum, config['num_targets'])
                trial_estimates.append(estimated_angles)
            
            # 取第一次试验的结果作为代表（所有模型使用相同的随机种子）
            if trial_estimates:
                representative_estimate = list(trial_estimates[0])

                # 以"真值"为主循环，每个真值找最近的估计值配对
                # 即使估计值不够，真值也一定会被记录（估计值用 NaN 占位）
                used = [False] * len(representative_estimate)
                for i, true_ang in enumerate(true_angles):
                    # 从未被占用的估计里挑离 true_ang 最近的
                    diffs = [abs(e - true_ang) if not used[j] else np.inf
                             for j, e in enumerate(representative_estimate)]

                    if len(diffs) == 0 or not np.isfinite(min(diffs)):
                        est_val = np.nan
                    else:
                        j = int(np.argmin(diffs))
                        used[j] = True
                        est_val = representative_estimate[j]

                    results[model_name]['true_angles'].append(true_ang)
                    results[model_name]['estimated_angles'].append(est_val)
                    results[model_name]['angle_indices'].append(angle_idx)
                    results[model_name]['target_indices'].append(i)
    
    return results

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

def plot_angle_estimation_scatter(results, config):
    print(f"绘制 {config['label']} 的散点图...")

    for model_name, model_results in results.items():
        if not model_results['true_angles']:
            print(f"警告: {model_name} 没有有效的估计结果")
            continue

        fig, ax = plt.subplots(figsize=(14, 10))

        # ===== 绘制估计值与真实值 =====
        if config['num_targets'] == 1:
            ax.scatter(model_results['angle_indices'],
                       model_results['estimated_angles'],
                       alpha=0.7, s=60, c='blue', marker='o',
                       label='ET', edgecolors='black', linewidth=0.5)

            sorted_data = sorted(zip(model_results['angle_indices'],
                                     model_results['true_angles']))
            sorted_indices, sorted_true = zip(*sorted_data)
            ax.plot(sorted_indices, sorted_true, 'r-', alpha=0.8, linewidth=3,
                    label='TT')
        else:
            for target_idx in range(config['num_targets']):
                target_angle_indices = []
                target_estimated_angles = []
                target_true_angles = []

                for i, t_idx in enumerate(model_results['target_indices']):
                    if t_idx == target_idx:
                        target_angle_indices.append(model_results['angle_indices'][i])
                        target_estimated_angles.append(model_results['estimated_angles'][i])
                        target_true_angles.append(model_results['true_angles'][i])

                # 该目标完全没有数据才跳过
                if not target_true_angles:
                    continue

                color = COLORS[target_idx % len(COLORS)]
                marker = MARKERS[target_idx % len(MARKERS)]
                size = MARKER_SIZES[target_idx % len(MARKER_SIZES)]

                # 真值线：只依赖真值，必画（修复点）
                sorted_data = sorted(zip(target_angle_indices, target_true_angles))
                if sorted_data:
                    sorted_indices, sorted_true = zip(*sorted_data)
                    ax.plot(sorted_indices, sorted_true,
                            color=color, linestyle='-', alpha=0.8, linewidth=3,
                            label=f'TT{target_idx + 1}')

                # 估计散点：单独判断，过滤 NaN
                valid = [(x, y) for x, y in zip(target_angle_indices, target_estimated_angles)
                         if not (isinstance(y, float) and np.isnan(y))]
                if valid:
                    vx, vy = zip(*valid)
                    ax.scatter(vx, vy,
                               alpha=0.7, s=size, c=color, marker=marker,
                               label=f'ET{target_idx + 1}',
                               edgecolors='black', linewidth=0.5)

        # ===== 坐标轴属性 =====
        ax.set_xlabel('Angle Index', fontweight='bold')
        ax.set_ylabel('Angle (degrees)', fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(ANGLE_START - 5, ANGLE_END + 5)
        ax.set_xlim(np.min(np.array(model_results['angle_indices'])),
                    np.max(np.array(model_results['angle_indices'])))

        # Legend 样式（大号加粗 + 简洁缩写）
        legend = ax.legend(loc='best', framealpha=0.9, fontsize=22, prop={'weight': 'bold'})
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('gray')

        # ===== 放大镜位置坐标范围的计算逻辑 =====
        all_idx = np.array(model_results['angle_indices'])
        all_est = np.array(
            [e for e in model_results['estimated_angles']
             if not (isinstance(e, float) and np.isnan(e))]
        )

        if config['num_targets'] == 3:
            center_x = (np.min(all_idx) + np.max(all_idx)) / 2
            center_y = (np.min(all_est) + np.max(all_est)) / 2
            zoom_x_range = (center_x - 5, center_x + 5)
            zoom_y_range = (center_y - 10, center_y + 10)

        elif config['num_targets'] == 2:
            min_x = np.min(all_idx)
            min_y = np.min(all_est)
            zoom_x_range = (min_x + 3, min_x + 13)
            zoom_y_range = (min_y + 3, min_y + 13)

        elif config['num_targets'] == 1:
            max_x = np.max(all_idx)
            max_y = np.max(all_est)
            zoom_x_range = (max_x - 5, max_x)
            zoom_y_range = (max_y - 5, max_y)

        else:
            center_x = (np.min(all_idx) + np.max(all_idx)) / 2
            center_y = (np.min(all_est) + np.max(all_est)) / 2
            zoom_x_range = (center_x - 5, center_x + 5)
            zoom_y_range = (center_y - 10, center_y + 10)

        # ===== 绘制放大镜（右下角） =====
        axins = inset_axes(ax, width="35%", height="35%", loc='lower right')
        for line in ax.get_lines():
            axins.plot(line.get_xdata(), line.get_ydata(),
                       color=line.get_color(), linestyle=line.get_linestyle(),
                       alpha=line.get_alpha(), linewidth=line.get_linewidth())
        for coll in ax.collections:
            offs = coll.get_offsets()
            axins.scatter(offs[:, 0], offs[:, 1],
                          alpha=coll.get_alpha(), c=coll.get_facecolors(),
                          marker=coll.get_paths()[0], s=coll.get_sizes(),
                          edgecolors=coll.get_edgecolors(), linewidths=coll.get_linewidths())

        axins.set_xlim(zoom_x_range)
        axins.set_ylim(zoom_y_range)
        axins.grid(True, alpha=0.3, linestyle='--')
        axins.set_xticks([])
        axins.set_yticks([])
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

        # ===== 保存文件 =====
        filename = f"{EXP_DIR}/{model_name}_{config['name']}_scatter.eps"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"  {model_name} 散点图已保存: {filename}")


def calculate_estimation_errors(results, config):
    """
    计算并保存估计误差统计
    """
    print(f"计算 {config['label']} 的估计误差...")
    
    error_stats = {}
    
    for model_name, model_results in results.items():
        if not model_results['true_angles']:
            continue
            
        # 计算角度估计误差
        errors = []
        for true_angle, est_angle in zip(model_results['true_angles'], 
                                       model_results['estimated_angles']):
            error = abs(est_angle - true_angle)
            errors.append(error)
        
        if errors:
            error_stats[model_name] = {
                'mean_error': np.mean(errors),
                'std_error': np.std(errors),
                'max_error': np.max(errors),
                'min_error': np.min(errors),
                'rmse': np.sqrt(np.mean([e**2 for e in errors]))
            }
    
    # 保存统计结果
    stats_filename = f"{EXP_DIR}/{config['name']}_error_statistics.txt"
    with open(stats_filename, 'w') as f:
        f.write(f"Angle Estimation Error Statistics - {config['label']}\n")
        f.write("=" * 60 + "\n")
        f.write(f"SNR: {SNR} dB\n")
        f.write(f"Monte Carlo Trials: {MC_TRIALS}\n")
        f.write(f"Target Count: {config['num_targets']}\n")
        f.write(f"Angle Range: {ANGLE_START}° to {ANGLE_END}°\n")
        f.write(f"Angle Step: {ANGLE_STEP}°\n")
        f.write(f"Target Spacing: {TARGET_SPACING}°\n\n")
        
        for model_name, stats in error_stats.items():
            f.write(f"{model_name}:\n")
            f.write(f"  Mean Error: {stats['mean_error']:.3f}°\n")
            f.write(f"  Std Error:  {stats['std_error']:.3f}°\n")
            f.write(f"  RMSE:       {stats['rmse']:.3f}°\n")
            f.write(f"  Max Error:  {stats['max_error']:.3f}°\n")
            f.write(f"  Min Error:  {stats['min_error']:.3f}°\n\n")
    
    print(f"  误差统计已保存: {stats_filename}")
    return error_stats

def save_detailed_results(results, config):
    """
    保存详细的实验结果数据
    """
    for model_name, model_results in results.items():
        if not model_results['true_angles']:
            continue
            
        filename = f"{EXP_DIR}/{model_name}_{config['name']}_detailed_results.txt"
        with open(filename, 'w') as f:
            f.write(f"Detailed Results - {model_name} - {config['label']}\n")
            f.write("=" * 60 + "\n")
            f.write("Angle_Index\tTarget_Index\tTrue_Angle\tEstimated_Angle\tError\n")
            
            for i, (idx, t_idx, true_ang, est_ang) in enumerate(zip(
                model_results['angle_indices'],
                model_results['target_indices'],
                model_results['true_angles'], 
                model_results['estimated_angles'])):
                
                error = abs(est_ang - true_ang)
                f.write(f"{idx}\t{t_idx}\t{true_ang:.1f}\t{est_ang:.1f}\t{error:.3f}\n")
        
        print(f"  {model_name} 详细结果已保存: {filename}")

# 全局变量: 当前配置的信源数量, 供MUSIC使用
current_num_sources = 1

if __name__ == "__main__":
    print("=== 开始角度扫描实验 ===")
    print(f"配置参数:")
    print(f"  角度范围: {ANGLE_START}° 到 {ANGLE_END}°")
    print(f"  角度步长: {ANGLE_STEP}°")
    print(f"  目标间距: {TARGET_SPACING}°")
    print(f"  信噪比: {SNR} dB")
    print(f"  蒙特卡洛试验次数: {MC_TRIALS}")
    
    try:
        # 加载所有模型
        print("\n=== 加载模型 ===")
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
        
        # 对每种角度配置运行实验
        for config in ANGLE_CONFIGS:
            print(f"\n{'='*50}")
            print(f"开始 {config['label']} 实验")
            print(f"角度范围: {config['angles_range'][0]}° 到 {config['angles_range'][-1]}°")
            print(f"角度间隔: {ANGLE_STEP}°")
            print(f"目标数量: {config['num_targets']}")
            if config['num_targets'] > 1:
                print(f"目标间距: {TARGET_SPACING}°")
            
            # 运行角度扫描实验
            results = run_angle_sweep_experiment(models, config)
            
            # 绘制散点图
            plot_angle_estimation_scatter(results, config)
            
            # 计算误差统计
            error_stats = calculate_estimation_errors(results, config)
            
            # 保存详细结果
            save_detailed_results(results, config)
            
            print(f"{config['label']} 实验完成")
        
        print(f"\n=== 所有实验完成 ===")
        print(f"所有结果文件已保存到 {EXP_DIR}/ 目录")
        
        # 生成实验总结
        summary_file = f"{EXP_DIR}/experiment_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("Angle Sweep Experiment Summary\n")
            f.write("=" * 50 + "\n")
            f.write(f"SNR: {SNR} dB\n")
            f.write(f"Monte Carlo Trials per angle: {MC_TRIALS}\n")
            f.write(f"Snapshots per trial: {snapshot_num}\n")
            f.write(f"Angle range: {ANGLE_START}° to {ANGLE_END}°\n")
            f.write(f"Angle step: {ANGLE_STEP}°\n")
            f.write(f"Target spacing (multi-target): {TARGET_SPACING}°\n\n")
            
            for config in ANGLE_CONFIGS:
                f.write(f"{config['label']}:\n")
                f.write(f"  Target count: {config['num_targets']}\n")
                f.write(f"  Angle range: {config['angles_range'][0]}° to {config['angles_range'][-1]}°\n")
                f.write(f"  Angle step: {ANGLE_STEP}°\n")
                f.write(f"  Total test points: {len(config['angles_range'])}\n")
                if config['num_targets'] > 1:
                    f.write(f"  Target spacing: {TARGET_SPACING}°\n")
                f.write("\n")
        
        print(f"实验总结已保存: {summary_file}")

    except Exception as e:
        print(f"运行时出现错误: {e}")
        import traceback
        traceback.print_exc()