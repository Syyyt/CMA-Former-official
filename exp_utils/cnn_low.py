import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNN_lowSNR(nn.Module):
    """CNN DOA 
    Deep Networks for Direction-of-Arrival Estimation in Low SNR

    Args:
        G (int): Grid of spectrum (121 or 181)
    """
    def __init__(self, G=121):
        super().__init__()
        self.Conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=256, kernel_size=(3,3), padding=0, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(2,2), padding=0, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(2,2), padding=0, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(2,2), padding=0, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.linear = nn.Sequential(
            nn.Linear(in_features=4*4*256, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=1024, out_features=G)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        x = self.Conv(x)
        x = x.view(batch_size, -1)
        x = self.linear(x)

        return F.sigmoid(x)

if __name__ == '__main__':
    """测试脚本 (模拟论文实验设置)"""
    # 1. 模拟输入数据 (论文中N=16, 3通道SCM)
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 16, 16)  # [B, 3, 16, 16]
    
    # 2. 初始化模型 (参数与论文一致)
    model = CNN_lowSNR(G = 121)
    
    # 3. 前向传播测试
    output = model(dummy_input)
    print(f"输入尺寸: {dummy_input.shape}")
    print(f"输出尺寸: {output.shape} (batch_size × num_classes)")
    print(f"输出范围: [{output.min().item():.4f}, {output.max().item():.4f}] (sigmoid激活)")
    
    # 4. 参数统计 (论文中5,876,347参数)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量: {total_params:,} (论文: 5,876,347)")
    
"""    # 5. 计算FLOPs (论文中407.77M)
    from thop import profile
    flops, _ = profile(model, inputs=(dummy_input,))
    print(f"FLOPs: {flops / 1e6:.2f}M (论文: 407.77M)")"""
