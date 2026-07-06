import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DropPath(nn.Module):
    """DropPath (Stochastic Depth) with enhanced stability
    
    Args:
        drop_prob (float): Drop probability (0 = no drop)
    """
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
            
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (B, 1, 1, ...)
        
        # 生成二值掩码
        mask = torch.rand(shape, dtype=x.dtype, device=x.device) < keep_prob
        
        # 数值稳定处理
        if keep_prob > 0:
            mask = mask.to(x.dtype) / keep_prob
        else:
            mask = torch.zeros_like(mask, dtype=x.dtype)
            
        return x * mask


class MSA(nn.Module):
    """多头自注意力机制 (Multi-head Self Attention)
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 论文中的QKV投影矩阵
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        # 生成QKV [B, N, 3*C] -> [3, B, num_heads, N, C//num_heads]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 每个形状 [B, num_heads, N, head_dim]

        # 缩放点积注意力 (公式7)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # 合并多头输出 (公式8)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class TransformerEncoder(nn.Module):
    """Transformer编码块 (对应论文图1(b))
    包含:
    - LayerNorm -> MSA -> DropPath
    - LayerNorm -> MLP -> DropPath
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop_path_rate=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MSA(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.drop_path1 = DropPath(drop_path_rate) #if drop_path_rate > 0. else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # 论文中的MLP结构 (图1(c))
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),  # 论文使用GELU激活
            nn.Linear(mlp_hidden_dim, dim)
        )
        self.drop_path2 = DropPath(drop_path_rate) #if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        # 第一个残差连接 (公式9)
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        # 第二个残差连接 (公式10)
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x

class DCT_ViT(nn.Module):
    """Dual Class Token Vision Transformer 完整实现
    论文架构对应图1(a)
    """
    def __init__(self, 
                 num_classes=121,  # 默认-60°~60°间隔1°共121类
                 in_chans=3,       # 输入通道数 (实部/虚部/相位)
                 embed_dim=243,    # 论文中C=243
                 depth=4,         # Transformer编码器层数L
                 num_heads=9,      # 论文中H_m=9
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.25,
                 drop_path_rate=0.1):
        super().__init__()
        
        # 1. Conv2D + BN 预处理层 (公式5)
        self.conv = nn.Conv2d(in_chans, embed_dim, kernel_size=9, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(embed_dim)
        
        # 计算展平后的序列长度 (论文中P=8 when N=16, H_c=9)
        self.num_patches = (16 - 9 + 1) ** 2  # P^2=64
        
        # 2. 双类别令牌 (公式6)
        self.cls_token_head = nn.Parameter(torch.randn(1, 1, embed_dim))  # 头部class token
        self.cls_token_tail = nn.Parameter(torch.randn(1, 1, embed_dim))  # 尾部class token
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 2, embed_dim))  # +2 for cls tokens
        
        # 3. Transformer编码器 (论文中L层)
        def get_drop_path_rates(depth):
            return [min(0.1 * (l - 1), 1.0) for l in range(1, depth + 1)]
        
        dpr = get_drop_path_rates(depth=depth)  # 递增的drop path率
        self.dr = nn.Dropout(0.2)
        self.blocks = nn.ModuleList([
            TransformerEncoder(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path_rate=dpr[i])
            for i in range(depth)])
        
        # 4. 分类头 (论文中3层FC)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(2 * embed_dim, 8 * embed_dim),  # 拼接双token
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(8 * embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(4 * embed_dim, num_classes)
        )
        
        # 初始化权重
        #nn.init.trunc_normal_(self.cls_token_head, std=0.02)
        #nn.init.trunc_normal_(self.cls_token_tail, std=0.02)
        #nn.init.trunc_normal_(self.pos_embed, std=0.02)
        #self.apply(self._init_weights)  # 论文中没有强调用什么初始化方式

    """def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)"""

    def forward(self, x):
        B = x.shape[0]
        
        # 1. Conv2D + BN (公式5)
        x = self.conv(x)  # [B, 3, 16, 16] -> [B, 243, 8, 8]
        x = self.bn(x)
        x = x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, x.shape[1])  # 展平 [B, 64, 243]
        
        # 2. 添加双class token和位置编码 (公式6)
        cls_token_head = self.cls_token_head.expand(B, -1, -1)  # [1,1,243] -> [B,1,243]
        cls_token_tail = self.cls_token_tail.expand(B, -1, -1)
        x = torch.cat((cls_token_head, x, cls_token_tail), dim=1)  # [B,66,243]
        x = x + self.pos_embed
        
        # 3. Transformer编码器 (公式9-10)
        x = self.dr(x)
        for blk in self.blocks:
            x = blk(x)
        
        # 4. 提取双token特征 (公式11)
        x = self.norm(x)
        cls_head = x[:, 0]   # 第一个class token
        cls_tail = x[:, -1]  # 最后一个class token
        x = torch.cat((cls_head, cls_tail), dim=1)  # [B, 2*243]
        
        # 5. 分类头
        x = self.head(x)
        return F.sigmoid(x)  # 多标签分类用sigmoid

if __name__ == '__main__':
    """测试脚本 (模拟论文实验设置)"""
    # 1. 模拟输入数据 (论文中N=16, 3通道SCM)
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 16, 16)  # [B, 3, 16, 16]
    
    # 2. 初始化模型 (参数与论文一致)
    model = DCT_ViT(
        num_classes=121,
        embed_dim=243,
        depth=4,
        num_heads=9,
        drop_path_rate=0.1
    )
    
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
