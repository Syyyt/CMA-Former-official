import torch
import torch.nn as nn


class DropPath(nn.Module):
    """DropPath (Stochastic Depth)"""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)

        mask = torch.rand(shape, dtype=x.dtype, device=x.device) < keep_prob

        if keep_prob > 0:
            mask = mask.to(x.dtype) / keep_prob
        else:
            mask = torch.zeros_like(mask, dtype=x.dtype)

        return x * mask


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-4):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return self.gamma * x


class MSA(nn.Module):
    """多头自注意力机制"""

    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(0.1)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(
            B, N, 3, self.num_heads, C // self.num_heads
        ).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class AttentionEncoder(nn.Module):
    """串行多尺度注意力Transformer编码块"""

    def __init__(
        self,
        dim,
        num_heads=[4, 8, 16],
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.1,
        use_layerscale=True,
    ):
        super().__init__()
        self.num_scales = len(num_heads)

        self.norms1 = nn.ModuleList([
            nn.LayerNorm(dim) for _ in range(self.num_scales)
        ])

        self.attns = nn.ModuleList([
            MSA(dim, num_heads=num_heads[i], qkv_bias=qkv_bias)
            for i in range(self.num_scales)
        ])

        self.lss = nn.ModuleList([
            LayerScale(dim) if use_layerscale else nn.Identity()
            for _ in range(self.num_scales)
        ])

        self.drop_paths1 = nn.ModuleList([
            DropPath(drop_path_rate) for _ in range(self.num_scales)
        ])

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )
        self.drop_path2 = DropPath(drop_path_rate)
        self.ls2 = LayerScale(dim) if use_layerscale else nn.Identity()

    def forward(self, x):
        for norm, attn, ls, drop_path in zip(
            self.norms1, self.attns, self.lss, self.drop_paths1
        ):
            x = x + drop_path(ls(attn(norm(x))))

        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class CMAFormer(nn.Module):
    def __init__(
        self,
        in_M=16,
        embed_dim=64,
        out_class=121,
        depth=4,
        num_heads=[4, 8, 16],
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.1,
        drop_path_rate=0.1,
        use_layerscale=True,
        drop_path_type="scheduled",   # scheduled / none / fixed
        pool_type="cls",              # cls / mean
        pe_type="learnable",          # learnable / pair_aware / lag_aware
        head_type="mlp",              # mlp / linear
        use_diag_tokens=False,        # False: 上三角非对角; True: 上三角含对角
    ):
        super().__init__()

        self.in_M = in_M
        self.embed_dim = embed_dim
        self.pool_type = pool_type
        self.pe_type = pe_type
        self.head_type = head_type
        self.use_diag_tokens = use_diag_tokens

        self.spatial_embedding = nn.Linear(4, embed_dim)

        pair_indices = self._build_pair_indices(in_M, use_diag_tokens)
        self.register_buffer("pair_indices", pair_indices, persistent=False)
        self.register_buffer("lag_indices", pair_indices[:, 1] - pair_indices[:, 0], persistent=False)
        self.num_pair_tokens = pair_indices.shape[0]

        if self.pool_type == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
            total_tokens = self.num_pair_tokens + 1
        else:
            total_tokens = self.num_pair_tokens

        if self.pe_type == "learnable":
            self.pos_embed = nn.Parameter(torch.randn(1, total_tokens, embed_dim))
        elif self.pe_type == "pair_aware":
            self.pos_i_embed = nn.Embedding(in_M, embed_dim)
            self.pos_j_embed = nn.Embedding(in_M, embed_dim)
            if self.pool_type == "cls":
                self.cls_pos_embed = nn.Parameter(torch.randn(1, 1, embed_dim))
        elif self.pe_type == "lag_aware":
            self.lag_embed = nn.Embedding(in_M, embed_dim)
            if self.pool_type == "cls":
                self.cls_pos_embed = nn.Parameter(torch.randn(1, 1, embed_dim))
        else:
            raise ValueError(f"Unsupported pe_type: {self.pe_type}")

        dpr = self._get_drop_path_rates(depth, drop_path_rate, drop_path_type)

        self.dr = nn.Dropout(drop_rate)
        self.blocks = nn.ModuleList([
            AttentionEncoder(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path_rate=dpr[i],
                use_layerscale=use_layerscale,
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        if self.head_type == "mlp":
            self.head = nn.Sequential(
                nn.Linear(embed_dim, 4 * embed_dim),
                nn.GELU(),
                nn.Dropout(drop_rate),
                nn.Linear(4 * embed_dim, 2 * embed_dim),
                nn.GELU(),
                nn.Dropout(drop_rate),
                nn.Linear(2 * embed_dim, out_class)
            )
        elif self.head_type == "linear":
            self.head = nn.Linear(embed_dim, out_class)
        else:
            raise ValueError(f"Unsupported head_type: {self.head_type}")

        self.apply(self._init_weights)

    def _build_pair_indices(self, in_M, use_diag_tokens):
        pair_list = []
        for i in range(in_M):
            start_j = i if use_diag_tokens else i + 1
            for j in range(start_j, in_M):
                pair_list.append([i, j])
        return torch.tensor(pair_list, dtype=torch.long)

    def _get_drop_path_rates(self, depth, drop_path_rate, drop_path_type):
        if drop_path_type == "none":
            return [0.0 for _ in range(depth)]
        elif drop_path_type == "fixed":
            return [drop_path_rate for _ in range(depth)]
        elif drop_path_type == "scheduled":
            if depth == 1:
                return [drop_path_rate]
            return [drop_path_rate * i / (depth - 1) for i in range(depth)]
        else:
            raise ValueError(f"Unsupported drop_path_type: {drop_path_type}")

    def _get_position_embedding(self, batch_size):
        if self.pe_type == "learnable":
            return self.pos_embed

        elif self.pe_type == "pair_aware":
            pair_embed = (
                self.pos_i_embed(self.pair_indices[:, 0]) +
                self.pos_j_embed(self.pair_indices[:, 1])
            )
            pair_embed = pair_embed.unsqueeze(0).expand(batch_size, -1, -1)

            if self.pool_type == "cls":
                cls_pos = self.cls_pos_embed.expand(batch_size, -1, -1)
                return torch.cat([cls_pos, pair_embed], dim=1)
            return pair_embed

        elif self.pe_type == "lag_aware":
            pair_embed = self.lag_embed(self.lag_indices)
            pair_embed = pair_embed.unsqueeze(0).expand(batch_size, -1, -1)

            if self.pool_type == "cls":
                cls_pos = self.cls_pos_embed.expand(batch_size, -1, -1)
                return torch.cat([cls_pos, pair_embed], dim=1)
            return pair_embed

        else:
            raise ValueError(f"Unsupported pe_type: {self.pe_type}")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)

    def forward(self, x):
        B = x.shape[0]

        if x.shape[1] != self.num_pair_tokens:
            raise ValueError(
                f"Input token number mismatch: expected {self.num_pair_tokens}, got {x.shape[1]}"
            )

        x = self.spatial_embedding(x)

        if self.pool_type == "cls":
            cls_token = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_token, x], dim=1)

        x = self.dr(x)
        x = x + self._get_position_embedding(B)

        x = self.dr(x)
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        if self.pool_type == "cls":
            pooled = x[:, 0]
        else:
            pooled = x.mean(dim=1)

        logits = self.head(pooled)
        return torch.sigmoid(logits)


# Backward-compatible alias used by the original experiment scripts and checkpoints.
GD_ViT = CMAFormer


if __name__ == '__main__':
    batch_size = 4
    dummy_input = torch.randn(batch_size, 120, 4)

    model = CMAFormer(
        in_M=16,
        embed_dim=128,
        out_class=121,
        depth=4,
        num_heads=[4, 8, 16],
    )

    output = model(dummy_input)
    print(f"输入尺寸: {dummy_input.shape}")
    print(f"输出尺寸: {output.shape} (batch_size × num_classes)")
    print(f"输出范围: [{output.min().item():.4f}, {output.max().item():.4f}] (sigmoid激活)")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量: {total_params:,}")
