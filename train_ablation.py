import os
import gpu_manager
gpu_manager.auto_select_gpu()
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.nn import BCEWithLogitsLoss, BCELoss
from models import CMAFormer
from tqdm import tqdm
import json
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
import math
from torch.optim.lr_scheduler import LambdaLR

# 配置参数
CONFIG = {
    "data_path": "DATA/doa_signal_dataset_lite_hard.h5",
    "save_dir": "checkpoints",
    "batch_size": 64,
    "lr": 1e-3,
    "epochs": 100,
    "weight_decay": 0.01,
    "warmup_epochs": 0,
    "snapshots": {
        "best_metric": "val_loss",
        "save_top_k": 3,
        "mode": "min"
    },
    "seed": 42,

    # 改这里选择 ablation
    "ablation_name": "baseline",

    "model_params": {
        "in_M": 16,
        "depth": 4,
        "embed_dim": 128,
        "out_class": 121,
        "num_heads": [4, 8, 16],
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "drop_rate": 0.1,
        "drop_path_rate": 0.1,

        # baseline
        "use_layerscale": True,
        "drop_path_type": "scheduled",
        "pool_type": "cls",
        "pe_type": "learnable",
        "head_type": "mlp",
        "use_diag_tokens": False,
    }
}

# one-by-one ablation
ABLATION_CONFIGS = {
    "baseline": {
        "model_params": {}
    },
    "baseline_hard": {
        "model_params": {},
        "data_path": "DATA/doa_signal_dataset_lite_hard.h5"
    },
    "baseline_soft": {
        "model_params": {},
        "data_path": "DATA/doa_signal_dataset_lite_soft.h5"
    },
    "no_layerscale": {
        "model_params": {
            "use_layerscale": False
        }
    },
    "no_droppath": {
        "model_params": {
            "drop_path_type": "none"
        }
    },
    "fixed_droppath": {
        "model_params": {
            "drop_path_type": "fixed"
        }
    },
    "mean_pooling": {
        "model_params": {
            "pool_type": "mean"
        }
    },
    "pair_aware_pe": {
        "model_params": {
            "pe_type": "pair_aware"
        }
    },
    "lag_aware_pe": {
        "model_params": {
            "pe_type": "lag_aware"
        }
    },
    "linear_head": {
        "model_params": {
            "head_type": "linear"
        }
    },
    "with_diag_tokens": {
        "model_params": {
            "use_diag_tokens": True
        },
        # 改成你的“上三角含对角项”数据文件
        "data_path": "DATA/doa_signal_dataset_with_diag.h5"
    },
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DOADataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path

        with h5py.File(h5_path, "r") as f:
            X_data = f["X"][:]
            labels_data = f["labels"][:]

        self.X_tensor = torch.tensor(X_data)
        self.labels_tensor = torch.tensor(labels_data).float()

        self.length = len(self.X_tensor)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.X_tensor[idx], self.labels_tensor[idx]


def setup_seed(seed):
    """设置随机种子确保可复现性"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    """
    创建带warmup的余弦退火学习率调度器
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine_decay)

    return LambdaLR(optimizer, lr_lambda)


def train():
    setup_seed(CONFIG["seed"])

    ablation_name = CONFIG["ablation_name"]
    if ablation_name not in ABLATION_CONFIGS:
        raise ValueError(f"Unknown ablation_name: {ablation_name}")

    ablation_cfg = ABLATION_CONFIGS[ablation_name]
    model_params = dict(CONFIG["model_params"])
    model_params.update(ablation_cfg.get("model_params", {}))
    data_path = ablation_cfg.get("data_path", CONFIG["data_path"])

    depth = model_params["depth"]
    embed_dim = model_params["embed_dim"]

    save_dir = os.path.join(
        CONFIG["save_dir"],
        f"depth_{depth}_embed_dim_{embed_dim}",
        ablation_name
    )
    os.makedirs(save_dir, exist_ok=True)

    model = CMAFormer(**model_params).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)
    criterion = BCELoss()

    full_dataset = DOADataset(data_path)
    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_set,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=CONFIG["batch_size"],
        num_workers=4,
        pin_memory=True
    )

    total_steps = len(train_loader) * CONFIG["epochs"]
    warmup_steps = len(train_loader) * CONFIG["warmup_epochs"]

    """scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        min_lr_ratio=0.1
    )"""

    start_epoch = 0
    best_metric = np.inf if CONFIG["snapshots"]["mode"] == "min" else -np.inf
    checkpoint_path = os.path.join(save_dir, "last_checkpoint.pt")

    best_models = []

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_metric = checkpoint["best_metric"]
        best_models = checkpoint.get("best_models", [])
        print(f"恢复训练从 epoch {start_epoch}, 最佳指标: {best_metric:.4f}")

    log_file = os.path.join(save_dir, "training_log.json")
    log_history = []
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            log_history = json.load(f)

    for epoch in range(start_epoch, CONFIG["epochs"]):
        model.train()
        train_loss = 0.0
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]") as pbar:
            for x, y in pbar:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * x.size(0)
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_set)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Val]") as pbar:
                for x, y in pbar:
                    x, y = x.to(device), y.to(device)
                    out = model(x)
                    loss = criterion(out, y)
                    val_loss += loss.item() * x.size(0)
                    pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})

        avg_val_loss = val_loss / len(val_set)

        scheduler.step(avg_val_loss)

        epoch_log = {
            "epoch": epoch + 1,
            "ablation_name": ablation_name,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        log_history.append(epoch_log)
        with open(log_file, "w") as f:
            json.dump(log_history, f, indent=2)

        current_metric = avg_val_loss

        model_save_data = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_metric": best_metric,
            "best_models": best_models,
            "model_config": model_params,
            "ablation_name": ablation_name,
            "data_path": data_path,
            "depth": depth,
            "training_config": CONFIG
        }

        epoch_filename = f"epoch_{epoch+1}_val_loss_{avg_val_loss:.4f}.pt"
        epoch_path = os.path.join(save_dir, epoch_filename)

        torch.save(model_save_data, epoch_path)

        best_models.append((avg_val_loss, epoch + 1, epoch_filename))

        if CONFIG["snapshots"]["mode"] == "min":
            best_models.sort(key=lambda x: x[0])
        else:
            best_models.sort(key=lambda x: x[0], reverse=True)

        if len(best_models) > CONFIG["snapshots"]["save_top_k"]:
            for _, _, filename in best_models[CONFIG["snapshots"]["save_top_k"]:]:
                old_path = os.path.join(save_dir, filename)
                if os.path.exists(old_path):
                    os.remove(old_path)

            best_models = best_models[:CONFIG["snapshots"]["save_top_k"]]

        if CONFIG["snapshots"]["mode"] == "min":
            if current_metric < best_metric:
                best_metric = current_metric
        else:
            if current_metric > best_metric:
                best_metric = current_metric

        model_save_data["best_metric"] = best_metric
        model_save_data["best_models"] = best_models
        torch.save(model_save_data, checkpoint_path)

        if best_models and best_models[0][1] == epoch + 1:
            best_model_path = os.path.join(save_dir, "best_model.pt")
            torch.save(model_save_data, best_model_path)

        print(
            f"[{ablation_name}] Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}"
        )
        print(f"当前保存的最佳模型: {[f'Epoch {ep} (val_loss: {metric:.4f})' for metric, ep, _ in best_models]}")


if __name__ == "__main__":
    train()
