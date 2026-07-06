# exp_diff_models_real.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from data import generate_received_signal
from exp_utils.load_model import load_model, load_cnn_model, load_dct_model, load_dnn_model, load_hmc_model
from exp_utils.music import music_spectrum
import h5py

# 脚本所在目录,用于把相对路径锚定到脚本自身(不依赖运行时工作目录)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 设置matplotlib字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# ==================== 可配置参数区域 ====================
# 角度扫描配置
ANGLE_START = -60           # 起始角度
ANGLE_END = 60              # 结束角度  
ANGLE_STEP = 1              # 角度步长（可方便调整）

# 其他实验参数
DEPTH = 4  
EMBED = 128
MODEL_PATH = f"checkpoints/depth_{DEPTH}_embed_dim_{EMBED}/best_model.pt"

# 阵列参数（MUSIC算法使用）
N_ANTENNA = 16              # 天线数
d = 0.5                     # 天线间距

# 可视化配置
COLORS = ['red', 'orange', 'green', 'purple', 'brown', 'pink']  # 目标颜色
MARKERS = ['o', 's', '^', 'D', 'v', '<']                       # 目标形状
MARKER_SIZES = [60, 50, 55, 45, 55, 50]                       # 标记大小

# 基于可配置参数生成角度网格
THETA_GRID = np.arange(ANGLE_START, ANGLE_END + 1, ANGLE_STEP)

# 创建实验子目录
EXP_DIR = "exp/real_data"
os.makedirs(EXP_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Real-data input variant written by validate_real_data_music_16ULA.m:
#   raw              -> /DATA_real, /DATA_imag
#   doa_normalized   -> /DATA_doa_norm_real, /DATA_doa_norm_imag
#   range_equalized  -> /DATA_range_eq_real, /DATA_range_eq_imag
REAL_DATA_VARIANT = os.environ.get("REAL_DATA_VARIANT", "range_equalized")
REAL_DATA_FILE = os.environ.get(
    "REAL_DATA_FILE",
    os.path.join(SCRIPT_DIR, "DATA", "260521darkroom_0-375_120-290-4_music_16ULA.h5")
)
COV_NORM_MODE = os.environ.get("COV_NORM_MODE", "trace")
COV_NORM_EPS = 1e-8
USE_RANGE_BIN_ENSEMBLE = os.environ.get("USE_RANGE_BIN_ENSEMBLE", "1") not in ("0", "false", "False")
RANGE_BIN_ENSEMBLE_MODE = os.environ.get("RANGE_BIN_ENSEMBLE_MODE", "max")
RANGE_BIN_MIN_SNAPSHOTS = int(os.environ.get("RANGE_BIN_MIN_SNAPSHOTS", "16"))

# 全局变量: 当前配置的信源数量, 供MUSIC使用
current_num_sources = 2

def normalize_covariance(R, mode=None):
    """Remove only the covariance scale, keeping phase and relative structure."""
    if mode is None:
        mode = COV_NORM_MODE
    mode = mode.lower()
    if mode in ("none", "off", "false", "0"):
        return R
    if mode == "trace":
        scale = np.real(np.trace(R)) / max(R.shape[0], 1)
    elif mode == "fro":
        scale = np.linalg.norm(R, ord="fro")
    elif mode == "maxabs":
        scale = np.max(np.abs(R))
    else:
        raise ValueError(f"Unknown covariance normalization mode: {mode}")
    return R / (scale + COV_NORM_EPS)

def sample_covariance(X):
    M, snapshot_count = np.shape(X)
    R = (X @ X.T.conj()) / max(snapshot_count, 1)
    return normalize_covariance(R)

# 所提出算法和对比算法处理
def preprocess_for_proposed(X):
    R = sample_covariance(X)
    M = R.shape[0]
    # 提取上三角非对角元素
    r_complex = []
    for i in range(M):
        for j in range(i+1, M):
            r_complex.append(R[i, j])
    
    r_complex = np.array(r_complex)
    
    # 分离实部、虚部、相位的sin和cos —— 共4个特征
    r = np.stack([
        r_complex.real,
        r_complex.imag,
        np.sin(np.angle(r_complex)),
        np.cos(np.angle(r_complex))
    ], axis=-1)
    r_tensor = torch.tensor(r, dtype=torch.float32).unsqueeze(0).to(device)
    return r_tensor

def preprocess_for_cnn(X):
    R = sample_covariance(X)
    R_real = np.real(R)
    R_imag = np.imag(R)
    R_phase = np.angle(R)
    R_combined = np.stack([R_real, R_imag, R_phase], axis=0)
    R_tensor = torch.tensor(R_combined, dtype=torch.float32).unsqueeze(0).to(device)
    return R_tensor

def preprocess_for_dct(X):
    R = sample_covariance(X)
    R_real = np.real(R)
    R_imag = np.imag(R)
    R_phase = np.angle(R)
    R_combined = np.stack([R_real, R_imag, R_phase], axis=0)
    R_tensor = torch.tensor(R_combined, dtype=torch.float32).unsqueeze(0).to(device)
    return R_tensor

def preprocess_for_dnn(X):
    R = sample_covariance(X)
    M = R.shape[0]
    # 提取上三角非对角元素
    r_complex = []
    for i in range(M):
        for j in range(i+1, M):
            r_complex.append(R[i, j])
    
    r_complex = np.array(r_complex)
    
    r = np.concatenate([r_complex.real, r_complex.imag])
    r = r / (np.linalg.norm(r) + COV_NORM_EPS)
    r_tensor = torch.tensor(r, dtype=torch.float32).unsqueeze(0).to(device)
    
    return r_tensor 

def preprocess_for_hmc(X):
    """
    HMC-ViT的输入: 协方差矩阵上三角的实部、虚部和相位, 形状 3 x 1 x 120
    对于 M=16 的ULA, 上三角非对角元素数目为 M*(M-1)/2 = 120
    """
    R = sample_covariance(X)
    M = R.shape[0]
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
        # num_sources通过全局变量current_num_sources传入
        spectrum = music_spectrum(X, num_sources=current_num_sources,
                                  theta_grid=THETA_GRID, N=N_ANTENNA, d=d)
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
            spectrum = torch.sigmoid(model(X_tensor)[0])
            spectrum = spectrum.cpu().numpy().squeeze()
        elif model_name == "HMC-ViT":
            # HMC-ViT 训练时使用 BCEWithLogitsLoss，模型输出为 logits
            # 推理时需要套 sigmoid 转成概率，才能与其他模型输出语义一致
            logits = model(X_tensor)
            spectrum = torch.sigmoid(logits).cpu().numpy().squeeze()
        else:
            spectrum = model(X_tensor).cpu().numpy().squeeze()
    
    return spectrum

def get_range_bin_ensemble_prediction(model, X, model_name, snapshot_range_bins):
    """Run a network on each selected range gate separately, then merge spectra."""
    if snapshot_range_bins is None:
        return get_model_prediction(model, X, model_name)

    snapshot_range_bins = np.asarray(snapshot_range_bins).reshape(-1)
    if snapshot_range_bins.size != X.shape[1]:
        raise ValueError(
            f"snapshot_range_bins length {snapshot_range_bins.size} does not match snapshots {X.shape[1]}"
        )

    spectra = []
    used_bins = []
    for range_bin in np.unique(snapshot_range_bins):
        idx = snapshot_range_bins == range_bin
        if np.count_nonzero(idx) < RANGE_BIN_MIN_SNAPSHOTS:
            continue
        spectra.append(get_model_prediction(model, X[:, idx], model_name))
        used_bins.append(int(range_bin))

    if not spectra:
        return get_model_prediction(model, X, model_name)

    spectra = np.stack(spectra, axis=0)
    mode = RANGE_BIN_ENSEMBLE_MODE.lower()
    if mode == "max":
        merged = np.max(spectra, axis=0)
    elif mode == "mean":
        merged = np.mean(spectra, axis=0)
    else:
        raise ValueError(f"Unknown RANGE_BIN_ENSEMBLE_MODE: {RANGE_BIN_ENSEMBLE_MODE}")

    print(f"{model_name}: range-bin ensemble {mode}, bins={used_bins}")
    return merged

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

def load_real_data(filename=None, N=None, data_variant=None, return_bins=False):
    """
    读取真实数据。
    h5 文件由 MATLAB 脚本 process_real_data_16ULA.m 生成,
    /DATA_real 和 /DATA_imag 已经是截取好的 16 元均匀子阵接收信号 X。
    无需再做阵元切片,直接返回即可。

    参数:
        filename: h5 文件路径, 默认相对脚本目录的 DATA/real_data.h5
        N: 可选, 限制使用的快拍数 (None 表示全部使用)
    """
    if filename is None:
        filename = REAL_DATA_FILE

    if data_variant is None:
        data_variant = REAL_DATA_VARIANT

    dataset_map = {
        "raw": ("/DATA_real", "/DATA_imag"),
        "doa_normalized": ("/DATA_doa_norm_real", "/DATA_doa_norm_imag"),
        "range_equalized": ("/DATA_range_eq_real", "/DATA_range_eq_imag"),
    }
    if data_variant not in dataset_map:
        raise ValueError(f"Unknown real-data variant: {data_variant}")

    with h5py.File(filename, 'r') as f:
            real_key, imag_key = dataset_map[data_variant]
            if real_key not in f or imag_key not in f:
                print(f"Variant {data_variant!r} is not in {filename}; falling back to raw /DATA_*")
                real_key, imag_key = dataset_map["raw"]
            data_real = np.array(f[real_key])  # 实部
            data_imag = np.array(f[imag_key])  # 虚部
            DATA = data_real + 1j * data_imag
            snapshot_range_bins = np.array(f["/snapshot_range_bins"]).reshape(-1) \
                if "/snapshot_range_bins" in f else None

    # MATLAB 列优先 vs h5py 行优先, 读出来维度是转置的, 转回 (阵元 x 快拍)
    if DATA.shape[0] > DATA.shape[1]:
        DATA = DATA.T

    print("DATA shape:", DATA.shape)
    print("DATA dtype:", DATA.dtype)
    print("DATA variant:", data_variant)
    print("DATA complex RMS:", np.sqrt(np.mean(np.abs(DATA) ** 2)))

    # DATA 已是 16 x 快拍 的 16 元 ULA 接收信号
    if N is not None and N < DATA.shape[1]:
        DATA = DATA[:, :N]
        if snapshot_range_bins is not None:
            snapshot_range_bins = snapshot_range_bins[:N]

    if return_bins:
        return DATA, snapshot_range_bins
    return DATA

def load_real_metadata(filename=None):
    if filename is None:
        filename = REAL_DATA_FILE
    metadata = {}
    with h5py.File(filename, "r") as f:
        for key, value in f.attrs.items():
            metadata[key] = np.array(value) if isinstance(value, np.ndarray) else value
    return metadata


def generate_spectrum_examples(models):
    """真实数据的谱图示例"""
    global current_num_sources
    
    colors = ['#0072BD', '#D95319', '#77AC30', "#BD00AD"]    
    label = "real"
    
    # 使用真实数据。学习模型默认按距离门分别推理后合成谱，避免跨距离门拼接造成漏检。
    X, snapshot_range_bins = load_real_data(return_bins=True)
    angles = [0, 22.4794]
    current_num_sources = len(angles)  # 供MUSIC使用
    
    for i, (model_name, model) in enumerate(models.items()):
        # 为每个模型创建独立的图片
        plt.figure(figsize=(8, 6))
        
        if USE_RANGE_BIN_ENSEMBLE and model_name != "MUSIC":
            spectrum = get_range_bin_ensemble_prediction(model, X, model_name, snapshot_range_bins)
        else:
            spectrum = get_model_prediction(model, X, model_name)
        
        if model_name == 'DNN':
            GRID = np.arange(-60, 60, 1)
        else:
            GRID = np.arange(-60, 61, 1)
            
        plt.plot(GRID, spectrum, '-', color=colors[0], linewidth=2.5, label='Predicted Spectrum')
        
        for angle in angles:
            plt.axvline(x=angle, color='#D95319', linestyle='--', linewidth=2, alpha=0.8, 
                        label='True DOA' if angle == angles[0] else "")
        
        plt.xlabel(r'Angle ($^\circ$)', fontfamily='Times New Roman', fontsize=14, fontweight='bold', labelpad=8)
        plt.ylabel('Normalized Spectrum', fontfamily='Times New Roman', fontsize=14, fontweight='bold', labelpad=8)
        
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
            loc='upper left',
            frameon=True,
            framealpha=0.95,
            edgecolor='gray',
            fancybox=True,
            shadow=True,
            prop={'family': 'Times New Roman', 'size': 12}
        )
        
        # 保存独立的图片
        plt.savefig(f'{EXP_DIR}/spectrum_{model_name}_{label}.eps', 
                    dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    print("=== 开始真实数据测试 ===")

    #try:
        # 加载所有模型
    print("=== 加载模型 ===")
    proposed_model, proposed_model_config = load_model(model_path=MODEL_PATH)
    print(f"Proposed模型加载成功")
    
    cnn_model = load_cnn_model()
    print(f"CNN模型加载成功")

    dct_vit_model = load_dct_model()
    print(f"DCT-ViT模型加载成功")

    hmc_vit_model = load_hmc_model()
    print(f"HMC-ViT模型加载成功")

    dnn_model = load_dnn_model()
    print(f"DNN模型加载成功")

    models = {
        "Proposed": proposed_model,
        "CNN": cnn_model,
        "DCT-ViT": dct_vit_model,
        "HMC-ViT": hmc_vit_model,
        "DNN": dnn_model,
        "MUSIC": None  # MUSIC为传统算法, 无需加载模型
    }        

    generate_spectrum_examples(models)

"""    except Exception as e:
        print(f"运行时出现错误: {e}")
        import traceback
        traceback.print_exc()"""
