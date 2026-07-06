import numpy as np


def music_spectrum(X, num_sources, theta_grid, N=16, d=0.5):
    """
    MUSIC算法实现
    参数:
        X: 接收信号矩阵 (N x snapshots)
        num_sources: 信源数量
        theta_grid: 角度搜索网格 (度)
        N: 天线数
        d: 天线间距 (以波长为单位)
    返回:
        spectrum: MUSIC空间谱 (与theta_grid同长度)
    """
    M, S = np.shape(X)
    # 协方差矩阵
    R = (X @ X.conj().T) / S

    # 特征分解
    eigvals, eigvecs = np.linalg.eigh(R)
    # eigh返回升序排列的特征值，噪声子空间对应较小的特征值
    idx = np.argsort(eigvals)[::-1]  # 降序
    eigvecs = eigvecs[:, idx]

    # 噪声子空间: 后 M - num_sources 个特征向量
    En = eigvecs[:, num_sources:]

    # 构建导向矢量并计算空间谱
    theta_rad = theta_grid * np.pi / 180
    n = np.arange(M)
    spectrum = np.zeros(len(theta_grid))
    for i, th in enumerate(theta_rad):
        a = np.exp(1j * 2 * np.pi * d * np.sin(th) * n).reshape(-1, 1)
        denom = np.abs(a.conj().T @ En @ En.conj().T @ a).squeeze()
        spectrum[i] = 1.0 / (denom + 1e-12)

    # 归一化
    spectrum = spectrum / np.max(spectrum)
    return spectrum