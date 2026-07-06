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


# 设置全局字体为Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 配置参数
THETA_GRID = np.arange(-60, 61, 1)  # 角度网格
N = 16                              # 天线数
d = 0.5                             # 天线间距

DEPTH =  4                          # 指定要加载的模型depth
EMBED = 128
MODEL_PATH = f"checkpoints/depth_{DEPTH}_embed_dim_{EMBED}/best_model.pt"
MC_TRIALS = 2000                     # 蒙特卡洛实验次数
SNR_RANGE = np.arange(-20, 1, 2)    # 信噪比测试范围 [-20, -18, ..., 0]
snapshot_num = 1000

# 创建实验子目录
EXP_DIR = "exp/model_comparison"
os.makedirs(EXP_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 固定信源配置
FIXED_SOURCES = [
    #{"angles": [-3.15], "label": "OneTarget"},
    #{"angles": [-18.2, 1.18], "label": "TwoTargets"}, 
    #{"angles": [-6.77, -1.21, 5.11], "label": "ThreeTargets"},  # -20db 2 0db
    {"angles": [-12.89, -9.13, 8.09, 10.12], "label": "FourTargets"} # -10db 2 10db
]

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
    #   归一化
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
    对于 M=16 的ULA, 上三角非对角元素数目为 M*(M-1)/2 = 120
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

    # 组装成 3 x 1 x 120
    r_stack = np.stack([r_real, r_imag, r_phase], axis=0)  # 3 x 120
    r_stack = r_stack[:, np.newaxis, :]                    # 3 x 1 x 120
    r_tensor = torch.tensor(r_stack, dtype=torch.float32).unsqueeze(0).to(device)  # 1 x 3 x 1 x 120
    return r_tensor


def get_model_prediction(model, X, model_name):
    """根据模型类型进行相应的预处理和预测"""
    if model_name == "MUSIC":
        # MUSIC算法: 非网络模型, 直接调用
        # num_sources通过闭包外部传入, 这里用current_num_sources全局变量
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
            # HMC-ViT 训练时使用 BCEWithLogitsLoss，模型输出为 logits
            # 推理时需要套 sigmoid 转成概率，才能与其他模型输出语义一致
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

def calculate_squared_errors(true_angles, pred_angles):
    """
    计算每个源的 (pred - true)^2, 用于全局 Monte Carlo RMSE 汇总.
    - 按角度排序后一一配对
    - 若预测数目与真实数目不匹配, 用 180° 惩罚 (保留原有思想)
    返回: 长度 = len(true_angles) 的平方误差数组
    """
    true_sorted = np.sort(np.asarray(true_angles, dtype=float))
    K = len(true_sorted)
    if len(pred_angles) != K:
        return np.full(K, 180.0 ** 2)
    pred_sorted = np.sort(np.asarray(pred_angles, dtype=float))
    return (pred_sorted - true_sorted) ** 2

def evaluate_models_on_dataset(models, test_data):
    """
    在预生成的数据集上评估所有模型
    参数:
        models: 模型字典 {model_name: model}
        test_data: 预生成的测试数据 [config_idx][snr_idx][trial_idx]
    返回:
        all_results: {model_name: {config_idx: [rmse_per_snr]}}

    [修正] RMSE 采用严格意义上的 Monte Carlo 定义:
           RMSE = sqrt( (1/(T*K)) * sum_t sum_k (pred_{t,k} - true_k)^2 )
           即: 先收集所有 trial 的所有源的平方误差, 最后统一开根号.
           原先的 mean(per-trial RMSE) 由于 sqrt 的凹性会系统性低估真实 RMSE.
    """
    global current_num_sources
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
        current_num_sources = num_sources  # 供MUSIC使用
        label = config["label"]
        
        print(f"\n评估配置: {label} (角度: {angles})")
        
        # 对每个SNR进行评估
        for snr_idx, snr in enumerate(SNR_RANGE):
            print(f"  SNR: {snr}dB", end="")
            
            # [修正] 收集每个模型在当前 SNR 下的所有平方误差 (跨 trial 跨源)
            model_sqerr_results = {}
            for model_name in models.keys():
                model_sqerr_results[model_name] = []
            
            # 在所有试验中评估（使用相同的数据）
            for trial in range(MC_TRIALS):
                # 获取当前试验的数据（所有模型使用相同数据）
                X = test_data[config_idx][snr_idx][trial]
                
                # 让所有模型对相同数据进行预测
                for model_name, model in models.items():
                    spectrum = get_model_prediction(model, X, model_name)
                    pred_angles = extract_peaks(spectrum, num_sources)
                    sq_errors = calculate_squared_errors(sorted(angles), sorted(pred_angles))
                    model_sqerr_results[model_name].extend(sq_errors.tolist())
            
            # [修正] 当前 SNR 的 RMSE = sqrt(所有平方误差的均值)
            for model_name in models.keys():
                avg_rmse = float(np.sqrt(np.mean(model_sqerr_results[model_name])))
                all_results[model_name][config_idx].append(avg_rmse)
            
            # 打印当前SNR的结果
            rmse_str = " | ".join([f"{name}: {np.sqrt(np.mean(model_sqerr_results[name])):.2f}°" 
                                  for name in models.keys()])
            print(f" | {rmse_str}")
    
    return all_results

def calculate_crb(angles, snr_db, N=16, d=0.5, S=snapshot_num):
    """
    修正的CRB计算.

    [修正] 返回的 crb_deg 不再是 sqrt(CRB_k) (源级别的标准差下界),
           而是 CRB_k 本身 (方差下界, 单位: deg^2),
           以便外层按 RMSE 指标对应的公式 sqrt(mean(CRB_k)) 来聚合.
    """
    angles_rad = np.array(angles) * np.pi / 180
    K = len(angles)
    snr_linear = 10**(snr_db / 10)
    noise_var = 1.0  # 归一化噪声功率
    
    # 构建导向矩阵和导数矩阵
    n = np.arange(N)
    A = np.exp(1j * 2 * np.pi * d * np.outer(np.sin(angles_rad), n)).T  # N x K
    
    # 计算导数矩阵
    D = np.zeros((N, K), dtype=complex)
    for k in range(K):
        D[:, k] = 1j * 2 * np.pi * d * np.cos(angles_rad[k]) * n * A[:, k]
    
    # 投影矩阵 
    P_A_perp = np.eye(N) - A @ np.linalg.pinv(A)
    
    # Fisher信息矩阵 
    H = D.conj().T @ P_A_perp @ D
    
    # 确定性模型的CRB
    P_est = snr_linear * np.eye(K)  # 假设源功率相等且不相关
    FIM = 2 * S / noise_var * np.real(H * P_est.T)
    
    # CRB = FIM^(-1) 的对角元素
    try:
        crb_matrix = np.linalg.inv(FIM)
        crb_rad = np.diag(crb_matrix)
    except np.linalg.LinAlgError:
        # 如果矩阵奇异，使用伪逆
        crb_matrix = np.linalg.pinv(FIM)
        crb_rad = np.diag(crb_matrix)
    
    # [修正] 返回 deg^2 单位的方差下界, 而不是开根后的 deg
    crb_var_deg2 = np.abs(np.real(crb_rad)) * (180 / np.pi) ** 2
    return crb_var_deg2


def calculate_all_crb(fixed_sources, snr_range):
    """
    计算所有配置下的 CRB 曲线 (RMSE 形式).

    [修正] 聚合方式改为:
           CRB_RMSE(snr) = sqrt( mean_k( CRB_k(snr) ) )     (单位: deg)
           与 RMSE = sqrt( mean_k( E[(theta_hat_k - theta_k)^2] ) ) 对应.
           原先 mean_k( sqrt(CRB_k) ) 的聚合方式对 RMSE 指标不是紧下界.
    """
    all_crb = []
    
    print("\n=== 计算CRB理论下界 ===")
    for config in fixed_sources:
        angles = config["angles"]
        label = config["label"]
        crb_results = []
        
        print(f"计算 {label} 的CRB...")
        for snr in snr_range:
            crb_var_deg2 = calculate_crb(angles, snr)           # 每源的 CRB_k (deg^2)
            rmse_bound = float(np.sqrt(np.mean(crb_var_deg2)))  # sqrt(mean(CRB_k))  (deg)
            crb_results.append(rmse_bound)
        
        all_crb.append(crb_results)
    
    return all_crb

def plot_single_condition_comparison(all_models_results,  condition_idx, crb_results):
    """为单个测试条件绘制所有模型的对比图"""
    config = FIXED_SOURCES[condition_idx]
    label = config['label']

    width = 8
    height = 6
    plt.figure(figsize=(width, height))
    plt.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.15)

    colors = ['#0072BD', '#D95319', '#77AC30', '#7E2F8E', '#EDB120', '#A2142F', '#000000']
    markers = ['o', 's', '^', 'v', 'D', 'P', 'X']
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.']

    # 绘制各模型的性能曲线
    model_names = list(all_models_results.keys())
    for i, (model_name, results) in enumerate(all_models_results.items()):
        rmse_vals = results[condition_idx]  # 修改这里的数据访问方式
        plt.plot(SNR_RANGE, rmse_vals, 
                markers[i] + linestyles[i], 
                label=model_name, 
                color=colors[i],
                markerfacecolor='white', 
                markeredgewidth=2, 
                markersize=8,
                linewidth=2) 

    # 绘制CRB曲线
    plt.plot(SNR_RANGE, crb_results[condition_idx],
            label='CRB', 
            color=colors[len(model_names)],
            markerfacecolor='white', 
            markeredgewidth=2, 
            markersize=8,
            linewidth=2)  

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
    
    legend = plt.legend(
        loc='best',
        #bbox_to_anchor=(0.95, 0.95),
        frameon=True,
        framealpha=0.95,
        edgecolor='gray',
        fancybox=True,
        shadow=True,
        prop={'family': 'Times New Roman', 'size': 16}
    )

    plt.savefig(f'{EXP_DIR}/performance_comparison_{label}.eps', dpi=300, bbox_inches='tight')
    plt.close()


def generate_spectrum_comparison(models, test_data, condition_idx, snr_idx=10):  # SNR=0dB对应索引10
    """生成特定条件下的谱图对比（每个模型生成独立的图片）"""
    global current_num_sources
    config = FIXED_SOURCES[condition_idx]
    angles = config["angles"]
    label = config["label"]
    snr = SNR_RANGE[snr_idx]
    current_num_sources = len(angles)  # 供MUSIC使用
    
    # 使用测试数据集中的第一个样本
    X = test_data[condition_idx][snr_idx][0]
    
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
        
        plt.title(f'{model_name} - {label} (SNR={snr}dB)', 
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
        plt.savefig(f'{EXP_DIR}/spectrum_{model_name}_{label}_SNR{snr}dB.eps', 
                   dpi=300, bbox_inches='tight')
        plt.close()


def generate_performance_table(all_models_results, crb_results):
    """生成性能统计表"""
    snr_indices = [0, len(SNR_RANGE)//2, -1]
    selected_snrs = [SNR_RANGE[i] for i in snr_indices]
    
    with open(f'{EXP_DIR}/performance_summary.txt', 'w') as f:
        f.write("Model Performance Summary\n")
        f.write("=" * 80 + "\n\n")
        
        for snr_idx, snr in zip(snr_indices, selected_snrs):
            f.write(f"SNR = {snr} dB\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Model':<12} {'1-Target':<10} {'2-Target':<10} {'3-Target':<10} {'4-Target':<10} {'Average':<10}\n")
            f.write("-" * 80 + "\n")
            
            for model_name, results in all_models_results.items():
                rmse_values = []
                for config_idx in range(len(FIXED_SOURCES)):
                    rmse = results[config_idx][snr_idx]
                    rmse_values.append(rmse)
                    
                avg_rmse = np.mean(rmse_values)
                f.write(f"{model_name:<12} {rmse_values[0]:<10.2f} {rmse_values[1]:<10.2f} {rmse_values[2]:<10.2f} {rmse_values[3]:<10.2f} {avg_rmse:<10.2f}\n")
            
            crb_values = [crb_results[i][snr_idx] for i in range(len(FIXED_SOURCES))]
            avg_crb = np.mean(crb_values)
            f.write(f"{'CRB':<12} {crb_values[0]:<10.2f} {crb_values[1]:<10.2f} {crb_values[2]:<10.2f} {crb_values[3]:<10.2f} {avg_crb:<10.2f}\n")
            f.write("\n")

# 全局变量: 当前配置的信源数量, 供MUSIC使用
current_num_sources = 1

if __name__ == "__main__":
    print("=== 开始严谨的模型对比实验 ===")

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
        
        # 预先生成所有测试数据
        test_data = generate_test_dataset()
        
        # 计算CRB理论下界
        crb_results = calculate_all_crb(FIXED_SOURCES, SNR_RANGE)
        
        # 在固定数据集上评估所有模型
        all_models_results = evaluate_models_on_dataset(models, test_data)
        
        print("\n=== 生成对比图表 ===")
        
        # 保存详细结果
        for model_name, results in all_models_results.items():
            for config_idx, config in enumerate(FIXED_SOURCES):
                filename = f"{EXP_DIR}/rmse_{model_name}_{config['label']}.txt"
                with open(filename, "w") as f:
                    f.write("SNR(dB)\tRMSE(degree)\n")
                    for snr, rmse in zip(SNR_RANGE, results[config_idx]):
                        f.write(f"{snr}\t{rmse:.4f}\n")
        
        # 为每个测试条件单独绘制对比图
        for condition_idx in range(len(FIXED_SOURCES)):
            plot_single_condition_comparison(all_models_results, condition_idx, crb_results)
            config_name = FIXED_SOURCES[condition_idx]['label']
            print(f"{config_name} 条件下的性能对比图已保存")
        
        # 生成性能统计表
        #generate_performance_table(all_models_results, crb_results)
        #print("性能统计表已保存")
        
        # 生成特定SNR下的谱图对比（使用相同的测试样本）
        snr_0db_idx = list(SNR_RANGE).index(0)  # 找到SNR=0dB的索引
        for condition_idx in range(len(FIXED_SOURCES)):
            generate_spectrum_comparison(models, test_data, condition_idx, snr_0db_idx)
            config_name = FIXED_SOURCES[condition_idx]['label']
            print(f"{config_name} 谱图对比已保存")
        
        # 保存CRB结果
        with open(f'{EXP_DIR}/crb_results.txt', 'w') as f:
            f.write("CRB (Cramér-Rao Bound) Results\n")
            f.write("=" * 50 + "\n")
            f.write("SNR(dB)\t1-Target\t2-Target\t3-Target\t4-Target\n")
            for snr_idx, snr in enumerate(SNR_RANGE):
                f.write(f"{snr}")
                for config_idx in range(len(FIXED_SOURCES)):
                    f.write(f"\t{crb_results[config_idx][snr_idx]:.4f}")
                f.write("\n")
        print("CRB结果已保存")
        
        print("\n=== 实验完成 ===")
        print(f"\n所有结果文件已保存到 {EXP_DIR}/ 目录")

    except Exception as e:
        print(f"运行时出现错误: {e}")
        import traceback
        traceback.print_exc()