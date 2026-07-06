import os
import torch
from .DCT_ViT import DCT_ViT
from .cnn_low import CNN_lowSNR
from .dnn import DNNDOAEstimator
from models import CMAFormer
from pathlib import Path
from .HMC_ViT import cnn_trans as HMC_ViT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def torch_load_compat(path, map_location=None, weights_only=True):
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception:
        if weights_only:
            return torch.load(path, map_location=map_location, weights_only=False)
        raise


def load_model(model_path=None, depth=4, embed_dim=128):
    """
    加载预训练模型
    参数:
        model_path: 模型文件路径，如果为None则使用默认路径
        depth: 模型深度，如果为None则从checkpoint中读取
    """
    if model_path is None:
        model_path = f"checkpoints/depth_{depth}_embed_dim_{embed_dim}/best_model.pt"
    
    print(f"正在加载模型: {model_path}")
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 修改2: 加载checkpoint并提取模型配置
    checkpoint = torch_load_compat(model_path, map_location='cpu', weights_only=True)
    
    # 从checkpoint中获取模型配置
    if "model_config" in checkpoint:
        model_config = checkpoint["model_config"]
        #print(f"从checkpoint加载模型配置: {model_config}")
    else:
        # 向后兼容：如果没有保存配置，使用默认配置
        print("警告: checkpoint中未找到模型配置")
        model_config = {
            "in_M": 16,
            "depth": depth,
            "embed_dim": embed_dim,
            "out_class": 121,
            "num_heads": [4, 8, 16],
        }
    
    # 创建模型实例
    model = CMAFormer(**model_config)
    
    # 加载模型权重
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
        print(f"成功加载模型权重 (epoch: {checkpoint.get('epoch', 'unknown')})")
        if "best_metric" in checkpoint:
            print(f"最佳验证指标: {checkpoint['best_metric']:.4f}")
    else:
        # 向后兼容：直接加载state_dict
        model.load_state_dict(checkpoint)
        print("加载旧格式模型权重")
    
    model.eval().to(device)
    return model, model_config

model_path_cnn = Path("exp_utils") / "weights" / "cnn_low_best.pt"
def load_cnn_model(model_path=model_path_cnn):
    print(f"正在加载模型: {model_path}")
    # 检查文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 加载
    model = CNN_lowSNR(G=121).to(device)
    checkpoint = torch_load_compat(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"]) 
    model.eval()

    return model

model_path_dct = Path("exp_utils") / "weights" / "dct_vit_best.pt"
def load_dct_model(model_path=model_path_dct):
    print(f"正在加载模型: {model_path}")
    # 检查文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 加载
    model = DCT_ViT(
        num_classes=121,
        embed_dim=243,
        depth=4,
        num_heads=9,
        drop_path_rate=0.1
    ).to(device)
    checkpoint = torch_load_compat(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"]) 
    model.eval()

    return model    

model_path_dnn = Path("exp_utils") / "weights" / "dnn_best.pth"
def load_dnn_model(model_path=model_path_dnn):
    print(f"正在加载模型: {model_path}")
    # 检查文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 加载
    input_dim = 240
    hidden_dim = input_dim // 2
    num_subregions = 6
    total_grids = 121
    grids_per_subregion = total_grids // num_subregions
    classifier_hidden_dims = [
        int(2/3 * input_dim),
        int(4/9 * input_dim)
    ]
    model = DNNDOAEstimator(
        input_dim=input_dim, 
        hidden_dim= hidden_dim, 
        num_subregions= num_subregions, 
        grids_per_subregion=grids_per_subregion, 
        classifier_hidden_dims=classifier_hidden_dims
        ).to(device)
    checkpoint = torch_load_compat(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint) 
    model.eval()

    return model 

model_path_hmc = Path("exp_utils") / "weights" / "hmc_vit.pth"
HMC_VIT_PARAMS = {
    "in_channels": 3,
    "patch_size_x": 1,
    "patch_size_y": 6,
    "emb_size": 128,
    "img_size": 120,
    "depth": 4,
    "n_classes": 121,
    "num_cls_tokens": 8,
}
def load_hmc_model(model_path=model_path_hmc):
    print(f"正在加载模型: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    model = HMC_ViT(**HMC_VIT_PARAMS).to(device)
    state_dict = torch_load_compat(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)      # ← 直接加载，不要 ["model"]
    model.eval()
    return model
