import os
import gpu_manager
gpu_manager.auto_select_gpu()
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from exp_utils.load_model import load_model, load_cnn_model, load_dct_model, load_dnn_model, load_hmc_model
from exp_utils.music import music_spectrum


# 设置全局字体为Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ============ 基本参数 (与理想场景保持一致) ============
THETA_GRID = np.arange(-60, 61, 1)  # 角度网格
N = 16                              # 天线数
d = 0.5                             # 天线间距

DEPTH = 4                           # 指定要加载的模型depth
EMBED = 128
MODEL_PATH = f"checkpoints/depth_{DEPTH}_embed_dim_{EMBED}/best_model.pt"
MC_TRIALS = 2000                    # 蒙特卡洛实验次数
SNR_RANGE = np.arange(-20, 1, 2)   # 信噪比测试范围
snapshot_num = 200

# ============ 非理想场景参数 ============
# 1) 互耦: 带状对称 Toeplitz 矩阵, 阶数 P=3, 系数为固定值
#    C[i,j] = c_{|i-j|},   |i-j| <= P;   0,   |i-j| > P
#    c_0 = 1 (对角), 其余按幅度衰减 + 固定相位设置
MC_COUPLING_COEFFS = np.array([
    1.0 + 0.0j,                          # c_0
    0.40 * np.exp(1j * np.pi / 3),       # c_1,  |c_1|=0.30, phase= 60°
    0.10 * np.exp(1j * 2 * np.pi / 3),   # c_2,  |c_2|=0.10, phase=120°
    0.05 * np.exp(1j * np.pi),           # c_3,  |c_3|=0.05, phase=180°
], dtype=complex)

# 2) 校准误差: 整个实验固定一组 (通过固定随机种子生成一次)
#    Γ = diag(g_n * exp(j φ_n))
CALIB_SEED = 42
CALIB_AMP_RANGE = (0.7, 1.3)                 # 幅度误差 ~ U(0.8, 1.2)
CALIB_PHASE_RANGE = (-np.pi / 6, np.pi / 6)  # 相位误差 ~ U(-15°, 15°)

# ============ 实验目录 ============
EXP_DIR = "exp/model_comparison_nonideal"
os.makedirs(EXP_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 固定信源配置 (1/2/3/4 目标全部打开)
FIXED_SOURCES = [
    #{"angles": [-3.15], "label": "OneTarget"},
    #{"angles": [-18.2, 1.18], "label": "TwoTargets"},
    {"angles": [-6.77, -1.21, 5.11], "label": "ThreeTargets"},
    #{"angles": [-6.77, -1.21, 5.11, 11.03], "label": "FourTargets"}
]


# ============ 非理想场景数据生成 ============
def build_coupling_matrix(N, coeffs):
    """
    构造带状对称 Toeplitz 互耦矩阵 C (N x N).
    coeffs = [c_0, c_1, ..., c_P]
    """
    P = len(coeffs) - 1
    C = np.zeros((N, N), dtype=complex)
    for i in range(N):
        for j in range(N):
            if abs(i - j) <= P:
                C[i, j] = coeffs[abs(i - j)]
    return C


def build_calibration_matrix(N, seed, amp_range, phase_range):
    """
    构造整个实验固定的对角校准误差矩阵 Γ = diag(g_n * exp(j φ_n)).
    通过固定 seed 只生成一次, 保证蒙特卡洛各次 trial 共用同一组 Γ.
    """
    rng = np.random.RandomState(seed)
    g = rng.uniform(amp_range[0], amp_range[1], size=N)
    phi = rng.uniform(phase_range[0], phase_range[1], size=N)
    return np.diag(g * np.exp(1j * phi))


# 全局固定: 互耦矩阵 C, 校准误差矩阵 Γ (脚本启动即确定, 全程不变)
C_MATRIX = build_coupling_matrix(N, MC_COUPLING_COEFFS)
GAMMA_MATRIX = build_calibration_matrix(N, CALIB_SEED, CALIB_AMP_RANGE, CALIB_PHASE_RANGE)


def generate_steering_vector(theta):
    """生成理想导向矢量 (互耦/校准误差作用在其上)"""
    return np.exp(1j * 2 * np.pi * d * np.arange(N) * np.sin(np.deg2rad(theta)))


def generate_received_signal_nonideal(thetas, SNR, S):
    """
    非理想场景接收信号:
        X = Γ · C · A · s + n
    其中:
        A : 理想导向矩阵 (N x K)
        C : 互耦矩阵 (N x N), 固定
        Γ : 校准误差矩阵 (N x N, 对角), 固定
    """
    A = np.column_stack([generate_steering_vector(theta) for theta in thetas])
    K = len(thetas)

    # 生成信号 (单位功率)
    signal = (np.random.randn(K, S) + 1j * np.random.randn(K, S)) / np.sqrt(2)

    # 生成噪声
    noise_power = 10 ** (-SNR / 10)
    noise = np.sqrt(noise_power / 2) * (np.random.randn(N, S) + 1j * np.random.randn(N, S))

    # 非理想接收模型
    return GAMMA_MATRIX @ C_MATRIX @ A @ signal + noise


def generate_test_dataset():
    """
    预先生成所有测试数据, 确保所有模型使用相同的数据集
    返回: test_data[config_idx][snr_idx][trial_idx] = X
    """
    print("=== 生成非理想场景测试数据集 ===")
    test_data = []

    for config_idx, config in enumerate(FIXED_SOURCES):
        angles = config["angles"]
        label = config["label"]
        print(f"生成 {label} 的测试数据...")

        config_data = []
        for snr_idx, snr in enumerate(SNR_RANGE):
            snr_data = []
            for trial in range(MC_TRIALS):
                X = generate_received_signal_nonideal(angles, SNR=snr, S=snapshot_num)
                snr_data.append(X)
            config_data.append(snr_data)
        test_data.append(config_data)

    print("测试数据集生成完成")
    return test_data


# ============ 各模型预处理 (与理想场景完全一致) ============
def preprocess_for_proposed(X):
    M, N_ = np.shape(X)
    R = (X @ X.T.conj()) / N_
    r_complex = []
    for i in range(M):
        for j in range(i + 1, M):
            r_complex.append(R[i, j])
    r_complex = np.array(r_complex)
    r = np.stack([r_complex.real, r_complex.imag,
                  np.sin(np.angle(r_complex)), np.cos(np.angle(r_complex))], axis=-1)
    r_tensor = torch.tensor(r, dtype=torch.float32).unsqueeze(0).to(device)
    return r_tensor


def preprocess_for_cnn(X):
    M, N_ = np.shape(X)
    R = (X @ X.T.conj()) / N_
    R_combined = np.stack([np.real(R), np.imag(R), np.angle(R)], axis=0)
    R_tensor = torch.tensor(R_combined, dtype=torch.float32).unsqueeze(0).to(device)
    return R_tensor


def preprocess_for_dct(X):
    M, N_ = np.shape(X)
    R = (X @ X.T.conj()) / N_
    R_combined = np.stack([np.real(R), np.imag(R), np.angle(R)], axis=0)
    R_tensor = torch.tensor(R_combined, dtype=torch.float32).unsqueeze(0).to(device)
    return R_tensor


def preprocess_for_dnn(X):
    M, N_ = np.shape(X)
    R = (X @ X.T.conj()) / N_
    r_complex = []
    for i in range(M):
        for j in range(i + 1, M):
            r_complex.append(R[i, j])
    r_complex = np.array(r_complex)
    r = np.concatenate([r_complex.real, r_complex.imag])
    r = r / np.linalg.norm(r)
    r_tensor = torch.tensor(r, dtype=torch.float32).unsqueeze(0).to(device)
    return r_tensor


def preprocess_for_hmc(X):
    M, N_ = np.shape(X)
    R = (X @ X.T.conj()) / N_
    r_complex = []
    for i in range(M):
        for j in range(i + 1, M):
            r_complex.append(R[i, j])
    r_complex = np.array(r_complex)
    r_stack = np.stack([r_complex.real, r_complex.imag, np.angle(r_complex)], axis=0)  # 3 x 120
    r_stack = r_stack[:, np.newaxis, :]                                                # 3 x 1 x 120
    r_tensor = torch.tensor(r_stack, dtype=torch.float32).unsqueeze(0).to(device)
    return r_tensor


def get_model_prediction(model, X, model_name):
    """根据模型类型进行相应的预处理和预测"""
    if model_name == "MUSIC":
        spectrum = music_spectrum(X, num_sources=current_num_sources,
                                  theta_grid=THETA_GRID, N=N, d=d)
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
        else:
            spectrum = model(X_tensor).cpu().numpy().squeeze()

    return spectrum


def extract_peaks(spectrum, num_sources, threshold=0.):
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
    return sorted(THETA_GRID[i] for i in peak_indices)


def calculate_squared_errors(true_angles, pred_angles):
    """
    计算每个源的 (pred - true)^2, 用于全局 Monte Carlo RMSE 汇总.
    - 按角度排序后一一配对
    - 若预测数目与真实数目不匹配, 用 180° 惩罚
    """
    true_sorted = np.sort(np.asarray(true_angles, dtype=float))
    K = len(true_sorted)
    if len(pred_angles) != K:
        return np.full(K, 180.0 ** 2)
    pred_sorted = np.sort(np.asarray(pred_angles, dtype=float))
    return (pred_sorted - true_sorted) ** 2


def evaluate_models_on_dataset(models, test_data):
    """
    在预生成的数据集上评估所有模型.
    RMSE = sqrt( (1/(T*K)) * sum_t sum_k (pred_{t,k} - true_k)^2 )
    """
    global current_num_sources
    print("\n=== 在固定数据集上评估所有模型 (非理想场景) ===")

    all_results = {}
    for model_name in models.keys():
        all_results[model_name] = {}
        for config_idx in range(len(FIXED_SOURCES)):
            all_results[model_name][config_idx] = []

    for config_idx, config in enumerate(FIXED_SOURCES):
        angles = config["angles"]
        num_sources = len(angles)
        current_num_sources = num_sources
        label = config["label"]

        print(f"\n评估配置: {label} (角度: {angles})")

        for snr_idx, snr in enumerate(SNR_RANGE):
            print(f"  SNR: {snr}dB", end="")

            model_sqerr_results = {name: [] for name in models.keys()}

            for trial in range(MC_TRIALS):
                X = test_data[config_idx][snr_idx][trial]
                for model_name, model in models.items():
                    spectrum = get_model_prediction(model, X, model_name)
                    pred_angles = extract_peaks(spectrum, num_sources)
                    sq_errors = calculate_squared_errors(sorted(angles), sorted(pred_angles))
                    model_sqerr_results[model_name].extend(sq_errors.tolist())

            for model_name in models.keys():
                avg_rmse = float(np.sqrt(np.mean(model_sqerr_results[model_name])))
                all_results[model_name][config_idx].append(avg_rmse)

            rmse_str = " | ".join([f"{name}: {np.sqrt(np.mean(model_sqerr_results[name])):.2f}°"
                                   for name in models.keys()])
            print(f" | {rmse_str}")

    return all_results


def plot_single_condition_comparison(all_models_results, condition_idx):
    """为单个测试条件绘制所有模型的对比图 (非理想场景, 不绘制 CRB)"""
    config = FIXED_SOURCES[condition_idx]
    label = config['label']

    width = 8
    height = 6
    plt.figure(figsize=(width, height))
    plt.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.15)

    colors = ['#0072BD', '#D95319', '#77AC30', '#7E2F8E', '#EDB120', '#A2142F']
    markers = ['o', 's', '^', 'v', 'D', 'P']
    linestyles = ['-', '--', '-.', ':', '-', '--']

    for i, (model_name, results) in enumerate(all_models_results.items()):
        rmse_vals = results[condition_idx]
        plt.plot(SNR_RANGE, rmse_vals,
                 markers[i] + linestyles[i],
                 label=model_name,
                 color=colors[i],
                 markerfacecolor='white',
                 markeredgewidth=2,
                 markersize=8,
                 linewidth=2)

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
        loc='upper right',
        frameon=True,
        framealpha=0.95,
        edgecolor='gray',
        fancybox=True,
        shadow=True,
        prop={'family': 'Times New Roman', 'size': 16}
    )

    plt.savefig(f'{EXP_DIR}/performance_comparison_{label}.eps', dpi=300, bbox_inches='tight')
    plt.close()


def generate_spectrum_comparison(models, test_data, condition_idx, snr_idx):
    """生成特定条件下的谱图对比 (每个模型一张独立图)"""
    global current_num_sources
    config = FIXED_SOURCES[condition_idx]
    angles = config["angles"]
    label = config["label"]
    snr = SNR_RANGE[snr_idx]
    current_num_sources = len(angles)

    X = test_data[condition_idx][snr_idx][0]

    colors = ['#0072BD', '#D95319', '#77AC30', "#BD00AD"]

    for i, (model_name, model) in enumerate(models.items()):
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

        plt.legend(
            loc='upper right',
            frameon=True,
            framealpha=0.95,
            edgecolor='gray',
            fancybox=True,
            shadow=True,
            prop={'family': 'Times New Roman', 'size': 12}
        )

        plt.savefig(f'{EXP_DIR}/spectrum_{model_name}_{label}_SNR{snr}dB.eps',
                    dpi=300, bbox_inches='tight')
        plt.close()


# 全局变量: 当前配置的信源数量, 供MUSIC使用
current_num_sources = 1


if __name__ == "__main__":
    print("=== 开始非理想场景(互耦 + 校准误差)模型对比实验 ===")

    try:
        # 加载所有模型
        print("=== 加载模型 ===")
        proposed_model, proposed_model_config = load_model(model_path=MODEL_PATH)
        print("Proposed模型加载成功")

        cnn_model = load_cnn_model()
        print("CNN模型加载成功")

        dct_vit_model = load_dct_model()
        print("DCT-ViT模型加载成功")

        dnn_model = load_dnn_model()
        print("DNN模型加载成功")

        hmc_vit_model = load_hmc_model()
        print("HMC-ViT模型加载成功")

        models = {
            "Proposed": proposed_model,
            "CNN": cnn_model,
            "DCT-ViT": dct_vit_model,
            "HMC-ViT": hmc_vit_model,
            "DNN": dnn_model,
            "MUSIC": None
        }

        # 保存非理想场景配置信息, 便于复现
        with open(f'{EXP_DIR}/nonideal_config.txt', 'w') as f:
            f.write("Non-ideal scenario configuration\n")
            f.write("=" * 60 + "\n")
            f.write("[Mutual Coupling] banded symmetric Toeplitz, fixed coeffs, P=3\n")
            for k, c in enumerate(MC_COUPLING_COEFFS):
                f.write(f"  c_{k}: |c|={np.abs(c):.4f}, phase={np.rad2deg(np.angle(c)):.2f} deg\n")
            f.write("\n[Calibration Error] fixed for entire experiment\n")
            f.write(f"  seed = {CALIB_SEED}\n")
            f.write(f"  amplitude ~ U({CALIB_AMP_RANGE[0]}, {CALIB_AMP_RANGE[1]})\n")
            f.write(f"  phase (deg) ~ U({np.rad2deg(CALIB_PHASE_RANGE[0]):.2f}, "
                    f"{np.rad2deg(CALIB_PHASE_RANGE[1]):.2f})\n")
            f.write("  realized gains : " +
                    ", ".join(f"{v:.4f}" for v in np.abs(np.diag(GAMMA_MATRIX))) + "\n")
            f.write("  realized phases(deg): " +
                    ", ".join(f"{v:.2f}" for v in np.rad2deg(np.angle(np.diag(GAMMA_MATRIX)))) + "\n")

        # 预先生成所有测试数据 (非理想场景)
        test_data = generate_test_dataset()

        # 在固定数据集上评估所有模型
        all_models_results = evaluate_models_on_dataset(models, test_data)

        print("\n=== 生成对比图表 ===")

        # 保存详细 RMSE 结果
        for model_name, results in all_models_results.items():
            for config_idx, config in enumerate(FIXED_SOURCES):
                filename = f"{EXP_DIR}/rmse_{model_name}_{config['label']}.txt"
                with open(filename, "w") as f:
                    f.write("SNR(dB)\tRMSE(degree)\n")
                    for snr, rmse in zip(SNR_RANGE, results[config_idx]):
                        f.write(f"{snr}\t{rmse:.4f}\n")

        # 为每个测试条件绘制对比图 (无CRB)
        for condition_idx in range(len(FIXED_SOURCES)):
            plot_single_condition_comparison(all_models_results, condition_idx)
            config_name = FIXED_SOURCES[condition_idx]['label']
            print(f"{config_name} 条件下的性能对比图已保存")

        # 生成 SNR=0dB 下的谱图对比
        snr_0db_idx = list(SNR_RANGE).index(0)
        for condition_idx in range(len(FIXED_SOURCES)):
            generate_spectrum_comparison(models, test_data, condition_idx, snr_0db_idx)
            config_name = FIXED_SOURCES[condition_idx]['label']
            print(f"{config_name} 谱图对比已保存")

        print("\n=== 实验完成 ===")
        print(f"\n所有结果文件已保存到 {EXP_DIR}/ 目录")

    except Exception as e:
        print(f"运行时出现错误: {e}")
        import traceback
        traceback.print_exc()