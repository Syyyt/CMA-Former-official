import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from data import generate_received_signal
from exp_utils.load_model import load_model, load_cnn_model, load_dct_model, load_dnn_model

# 配置参数
THETA_GRID = np.arange(-60, 61, 1)  # 角度网格
N = 16                              # 天线数
d = 0.5                             # 天线间距

DEPTH =  4                          # 指定要加载的模型depth
EMBED = 128
MODEL_PATH = f"checkpoints/depth_{DEPTH}_embed_dim_{EMBED}/best_model.pt"
MC_TRIALS = 500                     # 蒙特卡洛实验次数
SNAPSHOT_RANGE = [10, 20, 50, 100, 200, 500, 1000]  # 快拍数测试范围
FIXED_SNR = 0                       # 固定信噪比为0dB

# 创建实验子目录
EXP_DIR = "exp/snapshot_comparison"
os.makedirs(EXP_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 固定信源配置
FIXED_SOURCES = [
    {"angles": [-23.15], "label": "OneTarget"},
    {"angles": [-28.02, 1.18], "label": "TwoTargets"},
    {"angles": [-14.97, 0.21, 43.11], "label": "ThreeTargets"},
    {"angles": [-41.79, -10.23, 15.59, 27.18], "label": "FourTargets"}
]

def generate_test_dataset():
    """
    预先生成所有测试数据，确保所有模型使用相同的数据集
    返回: test_data[config_idx][snapshot_idx][trial_idx] = X
    """
    print("=== 生成测试数据集 ===")
    test_data = []
    
    for config_idx, config in enumerate(FIXED_SOURCES):
        angles = config["angles"]
        label = config["label"]
        print(f"生成 {label} 的测试数据...")
        
        config_data = []
        for snapshot_idx, snapshot_num in enumerate(SNAPSHOT_RANGE):
            snapshot_data = []
            print(f"  快拍数: {snapshot_num}")
            for trial in range(MC_TRIALS):
                # 生成接收信号
                X = generate_received_signal(angles, SNR=FIXED_SNR, S=snapshot_num)
                snapshot_data.append(X)
            config_data.append(snapshot_data)
        test_data.append(config_data)
    
    print("测试数据集生成完成")
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
    r = np.stack([r_complex.real, r_complex.imag, np.angle(r_complex)], axis=-1)
    #   归一化
   # r_phase = np.angle(r_complex)
    
    #r_norm = np.linalg.norm(r_complex)
    #r_complex_normalized = r_complex / (r_norm)
    #r_real = r_complex_normalized.real
    #r_imag = r_complex_normalized.imag    
    #real_norm = (r_real - np.mean(r_real)) / (np.std(r_real) + 1e-8)
    #imag_norm = (r_imag - np.mean(r_imag)) / (np.std(r_imag) + 1e-8)
    #phase_norm = r_phase / np.pi
    #r = np.stack([r_real, r_imag, phase_norm], axis=-1)
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


def get_model_prediction(model, X, model_name):
    """根据模型类型进行相应的预处理和预测"""
    if model_name == "Proposed":
        X_tensor = preprocess_for_proposed(X)
    elif model_name == "CNN":
        X_tensor = preprocess_for_cnn(X)
    elif model_name == "DCT-ViT":
        X_tensor = preprocess_for_dct(X)
    elif model_name == "DNN":
        X_tensor = preprocess_for_dnn(X)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    with torch.no_grad():
        if model_name == "DNN":
            spectrum = model(X_tensor)[0]
            spectrum = spectrum.cpu().numpy().squeeze()
        else :
            spectrum = model(X_tensor).cpu().numpy().squeeze()
    
    return spectrum

def extract_peaks(spectrum, num_sources, threshold=0.3):
    """
    从谱图中提取指定数量的峰值角度
    """
    candidate_indices = []
    for i in range(len(spectrum)):
        if spectrum[i] > threshold:
            if (i == 0 or spectrum[i] >= spectrum[i-1]) and \
               (i == len(spectrum)-1 or spectrum[i] >= spectrum[i+1]):
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

def evaluate_models_on_dataset(models, test_data):
    """
    在预生成的数据集上评估所有模型
    参数:
        models: 模型字典 {model_name: model}
        test_data: 预生成的测试数据 [config_idx][snapshot_idx][trial_idx]
    返回:
        all_results: {model_name: {config_idx: [rmse_per_snapshot]}}
    """
    print("\n=== 在固定数据集上评估所有模型 ===")
    
    all_results = {}
    
    # 初始化结果存储
    for model_name in models.keys():
        all_results[model_name] = {}
        for config_idx in range(len(FIXED_SOURCES)):
            all_results[model_name][config_idx] = []
    
    # 对每个配置进行评估
    for config_idx, config in enumerate(FIXED_SOURCES):
        angles = config["angles"]
        num_sources = len(angles)
        label = config["label"]
        
        print(f"\n评估配置: {label} (角度: {angles})")
        
        # 对每个快拍数进行评估
        for snapshot_idx, snapshot_num in enumerate(SNAPSHOT_RANGE):
            print(f"  快拍数: {snapshot_num}", end="")
            
            # 为每个模型计算在当前快拍数下的性能
            model_rmse_results = {}
            for model_name in models.keys():
                model_rmse_results[model_name] = []
            
            # 在所有试验中评估（使用相同的数据）
            for trial in range(MC_TRIALS):
                # 获取当前试验的数据（所有模型使用相同数据）
                X = test_data[config_idx][snapshot_idx][trial]
                
                # 让所有模型对相同数据进行预测
                for model_name, model in models.items():
                    spectrum = get_model_prediction(model, X, model_name)
                    pred_angles = extract_peaks(spectrum, num_sources)
                    rmse = calculate_rmse(sorted(angles), sorted(pred_angles))
                    model_rmse_results[model_name].append(rmse)
            
            # 计算每个模型在当前快拍数下的平均RMSE
            for model_name in models.keys():
                avg_rmse = np.mean(model_rmse_results[model_name])
                all_results[model_name][config_idx].append(avg_rmse)
            
            # 打印当前快拍数的结果
            rmse_str = " | ".join([f"{name}: {np.mean(model_rmse_results[name]):.2f}°" 
                                  for name in models.keys()])
            print(f" | {rmse_str}")
    
    return all_results

def calculate_crb_snapshot(angles, snr_db, snapshot_num, N=16, d=0.5):
    """计算给定快拍数下的DOA估计CRB"""
    angles_rad = np.array(angles) * np.pi / 180
    K = len(angles)
    snr_linear = 10**(snr_db / 10)
    
    n = np.arange(N)
    A = np.exp(1j * 2 * np.pi * d * np.outer(np.sin(angles_rad), n))
    dA_dtheta = 1j * 2 * np.pi * d * np.cos(angles_rad[:, np.newaxis]) * n[np.newaxis, :] * A
    
    P_A_perp = np.eye(N) - A.T @ np.linalg.pinv(A.T)
    
    fim_diag = []
    for k in range(K):
        da_k = dA_dtheta[k, :].reshape(-1, 1)
        fim_kk = 2 * snapshot_num * snr_linear * np.real(da_k.conj().T @ P_A_perp @ da_k)
        fim_diag.append(fim_kk[0, 0])
    
    crb_rad = np.array([1.0 / max(fim, 1e-10) for fim in fim_diag])
    crb_deg = np.sqrt(crb_rad) * 180 / np.pi
    
    return crb_deg

def calculate_all_crb(fixed_sources, snapshot_range):
    """计算所有配置下的CRB"""
    all_crb = []
    
    print("\n=== 计算CRB理论下界 ===")
    for config in fixed_sources:
        angles = config["angles"]
        label = config["label"]
        crb_results = []
        
        print(f"计算 {label} 的CRB...")
        for snapshot_num in snapshot_range:
            crb_values = calculate_crb_snapshot(angles, FIXED_SNR, snapshot_num)
            avg_crb = np.mean(crb_values)
            crb_results.append(avg_crb)
        
        all_crb.append(crb_results)
    
    return all_crb

def plot_single_condition_comparison(all_models_results, condition_idx, crb_results):
    """为单个测试条件绘制所有模型的对比图"""
    config = FIXED_SOURCES[condition_idx]
    label = config['label']
    
    width = 10
    height = 6
    plt.figure(figsize=(width, height))
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    
    colors = ['#0072BD', '#D95319', '#77AC30', '#7E2F8E', '#000000']
    markers = ['o', 's', '^', 'v', 'D']
    linestyles = ['-', '--', '-.', ':', '-']
    
    # 绘制各模型的性能曲线
    model_names = list(all_models_results.keys())
    for i, (model_name, results) in enumerate(all_models_results.items()):
        rmse_vals = results[condition_idx]
        plt.loglog(SNAPSHOT_RANGE, rmse_vals, 
                  markers[i] + linestyles[i], 
                  label=model_name, 
                  color=colors[i],
                  markerfacecolor='white', 
                  markeredgewidth=2, 
                  markersize=8,
                  linewidth=2)
    
    # 绘制CRB曲线
    plt.loglog(SNAPSHOT_RANGE, crb_results[condition_idx],
              label='CRB', 
              color=colors[len(model_names)],
              markerfacecolor='white', 
              markeredgewidth=2, 
              markersize=8,
              linewidth=2)
    
    plt.xlabel('Number of Snapshots', fontfamily='Times New Roman', fontsize=14, fontweight='bold', labelpad=10)
    plt.ylabel('RMSE (°)', fontfamily='Times New Roman', fontsize=14, fontweight='bold', labelpad=10)
    plt.grid(True, linestyle='--', alpha=0.3, which='both')
    plt.xlim([SNAPSHOT_RANGE[0], SNAPSHOT_RANGE[-1]])
    
    # 设置x轴刻度
    plt.xticks(SNAPSHOT_RANGE, [str(s) for s in SNAPSHOT_RANGE])
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.yaxis.set_tick_params(pad=10)
    
    legend = plt.legend(
        loc='upper right',
        bbox_to_anchor=(0.95, 0.95),
        frameon=True,
        framealpha=0.95,
        edgecolor='gray',
        fancybox=True,
        shadow=True,
        prop={'family': 'Times New Roman', 'size': 12}
    )
    
    plt.title(f'{label} - SNR = {FIXED_SNR} dB', 
              fontfamily='Times New Roman', fontsize=16, fontweight='bold', pad=20)
    
    plt.savefig(f'{EXP_DIR}/snapshot_comparison_{label}.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_spectrum_comparison(models, test_data, condition_idx, snapshot_idx=3):  # 默认使用100快拍
    """生成特定条件下的谱图对比（每个模型生成独立的图片）"""
    config = FIXED_SOURCES[condition_idx]
    angles = config["angles"]
    label = config["label"]
    snapshot_num = SNAPSHOT_RANGE[snapshot_idx]
    
    # 使用测试数据集中的第一个样本
    X = test_data[condition_idx][snapshot_idx][0]
    
    colors = ['#0072BD', '#D95319', '#77AC30', "#BD00AD"]
    
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
        
        plt.xlabel('Angle (°)', fontfamily='Times New Roman', fontsize=14, fontweight='bold', labelpad=8)
        plt.ylabel('Probability', fontfamily='Times New Roman', fontsize=14, fontweight='bold', labelpad=8)
        
        plt.title(f'{model_name} - {label} ({snapshot_num} Snapshots, SNR = {FIXED_SNR} dB)', 
                 fontfamily='Times New Roman', fontsize=16, fontweight='bold', pad=15)
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
            prop={'family': 'Times New Roman', 'size': 12}
        )
        
        # 保存独立的图片
        plt.savefig(f'{EXP_DIR}/spectrum_{model_name}_{label}_{snapshot_num}snapshots.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()


def generate_performance_table(all_models_results, crb_results):
    """生成性能统计表"""
    snapshot_indices = [0, len(SNAPSHOT_RANGE)//2, -1]
    selected_snapshots = [SNAPSHOT_RANGE[i] for i in snapshot_indices]
    
    with open(f'{EXP_DIR}/performance_summary.txt', 'w') as f:
        f.write(f"Model Performance Summary (SNR = {FIXED_SNR} dB)\n")
        f.write("=" * 80 + "\n\n")
        
        for snapshot_idx, snapshot_num in zip(snapshot_indices, selected_snapshots):
            f.write(f"Snapshots = {snapshot_num}\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Model':<12} {'1-Target':<10} {'2-Target':<10} {'3-Target':<10} {'4-Target':<10} {'Average':<10}\n")
            f.write("-" * 80 + "\n")
            
            for model_name, results in all_models_results.items():
                rmse_values = []
                for config_idx in range(len(FIXED_SOURCES)):
                    rmse = results[config_idx][snapshot_idx]
                    rmse_values.append(rmse)
                    
                avg_rmse = np.mean(rmse_values)
                f.write(f"{model_name:<12} {rmse_values[0]:<10.2f} {rmse_values[1]:<10.2f} {rmse_values[2]:<10.2f} {rmse_values[3]:<10.2f} {avg_rmse:<10.2f}\n")
            
            crb_values = [crb_results[i][snapshot_idx] for i in range(len(FIXED_SOURCES))]
            avg_crb = np.mean(crb_values)
            f.write(f"{'CRB':<12} {crb_values[0]:<10.2f} {crb_values[1]:<10.2f} {crb_values[2]:<10.2f} {crb_values[3]:<10.2f} {avg_crb:<10.2f}\n")
            f.write("\n")

def plot_overall_comparison():
    """绘制所有条件下的综合对比图"""
    width = 12
    height = 10
    fig, axes = plt.subplots(2, 2, figsize=(width, height))
    plt.subplots_adjust(left=0.1, right=0.95, top=0.93, bottom=0.08, wspace=0.3, hspace=0.4)
    
    colors = ['#0072BD', '#D95319', '#77AC30', '#7E2F8E', '#000000']
    markers = ['o', 's', '^', 'v', 'D']
    linestyles = ['-', '--', '-.', ':', '-']
    
    for idx, (config, ax) in enumerate(zip(FIXED_SOURCES, axes.flat)):
        label = config['label']
        
        # 绘制各模型的性能曲线
        model_names = list(all_models_results.keys())
        for i, (model_name, results) in enumerate(all_models_results.items()):
            rmse_vals = results[idx]
            ax.loglog(SNAPSHOT_RANGE, rmse_vals, 
                     markers[i] + linestyles[i], 
                     label=model_name, 
                     color=colors[i],
                     markerfacecolor='white', 
                     markeredgewidth=1.5, 
                     markersize=6,
                     linewidth=2)
        
        # 绘制CRB曲线
        ax.loglog(SNAPSHOT_RANGE, crb_results[idx],
                 label='CRB', 
                 color=colors[len(model_names)],
                 markerfacecolor='white', 
                 markeredgewidth=1.5, 
                 markersize=6,
                 linewidth=2)
        
        ax.set_xlabel('Number of Snapshots', fontfamily='Times New Roman', fontsize=12, fontweight='bold')
        ax.set_ylabel('RMSE (°)', fontfamily='Times New Roman', fontsize=12, fontweight='bold')
        ax.set_title(label, fontfamily='Times New Roman', fontsize=14, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3, which='both')
        ax.set_xlim([SNAPSHOT_RANGE[0], SNAPSHOT_RANGE[-1]])
        ax.set_xticks(SNAPSHOT_RANGE)
        ax.set_xticklabels([str(s) for s in SNAPSHOT_RANGE])
        
        # 只在第一个子图显示图例
        if idx == 0:
            ax.legend(loc='upper right', prop={'family': 'Times New Roman', 'size': 10})
    
    plt.suptitle(f'RMSE vs Number of Snapshots (SNR = {FIXED_SNR} dB)', 
                fontfamily='Times New Roman', fontsize=16, fontweight='bold')
    
    plt.savefig(f'{EXP_DIR}/overall_snapshot_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("=== 开始快拍数变化的模型对比实验 ===")

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

        models = {
            "Proposed": proposed_model,
            "CNN": cnn_model,
            "DCT-ViT": dct_vit_model,
            "DNN": dnn_model
        }
        
        # 预先生成所有测试数据
        test_data = generate_test_dataset()
        
        # 计算CRB理论下界
        crb_results = calculate_all_crb(FIXED_SOURCES, SNAPSHOT_RANGE)
        
        # 在固定数据集上评估所有模型
        all_models_results = evaluate_models_on_dataset(models, test_data)
        
        print("\n=== 生成对比图表 ===")
        
        # 保存详细结果
        for model_name, results in all_models_results.items():
            for config_idx, config in enumerate(FIXED_SOURCES):
                filename = f"{EXP_DIR}/rmse_{model_name}_{config['label']}.txt"
                with open(filename, "w") as f:
                    f.write("Snapshots\tRMSE(degree)\n")
                    for snapshot_num, rmse in zip(SNAPSHOT_RANGE, results[config_idx]):
                        f.write(f"{snapshot_num}\t{rmse:.4f}\n")
        
        # 为每个测试条件单独绘制对比图
        for condition_idx in range(len(FIXED_SOURCES)):
            plot_single_condition_comparison(all_models_results, condition_idx, crb_results)
            config_name = FIXED_SOURCES[condition_idx]['label']
            print(f"{config_name} 条件下的性能对比图已保存")
        
        # 绘制综合对比图
        plot_overall_comparison()
        print("综合对比图已保存")
        
        # 生成性能统计表
        generate_performance_table(all_models_results, crb_results)
        print("性能统计表已保存")
        
        # 生成特定快拍数下的谱图对比（使用相同的测试样本）
        snapshot_100_idx = SNAPSHOT_RANGE.index(100)  # 找到100快拍的索引
        for condition_idx in range(len(FIXED_SOURCES)):
            generate_spectrum_comparison(models, test_data, condition_idx, snapshot_100_idx)
            config_name = FIXED_SOURCES[condition_idx]['label']
            print(f"{config_name} 谱图对比已保存")
        
        # 保存CRB结果
        with open(f'{EXP_DIR}/crb_results.txt', 'w') as f:
            f.write(f"CRB (Cramér-Rao Bound) Results (SNR = {FIXED_SNR} dB)\n")
            f.write("=" * 60 + "\n")
            f.write("Snapshots\t1-Target\t2-Target\t3-Target\t4-Target\n")
            for snapshot_idx, snapshot_num in enumerate(SNAPSHOT_RANGE):
                f.write(f"{snapshot_num}")
                for config_idx in range(len(FIXED_SOURCES)):
                    f.write(f"\t{crb_results[config_idx][snapshot_idx]:.4f}")
                f.write("\n")
        print("CRB结果已保存")
        
        print("\n=== 实验完成 ===")
        print(f"\n所有结果文件已保存到 {EXP_DIR}/ 目录")
        print(f"实验设置: SNR = {FIXED_SNR} dB, 快拍数范围: {SNAPSHOT_RANGE}")

    except Exception as e:
        print(f"运行时出现错误: {e}")
        import traceback
        traceback.print_exc()
