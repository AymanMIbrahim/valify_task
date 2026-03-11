import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from train.helpers import config as cfg


# =========================================================
# Config helpers
# =========================================================
IMAGE_SIZE = getattr(cfg, "IMAGE_SIZE", 224)
NUM_CHANNELS = getattr(cfg, "NUM_CHANNELS", 3)
NUM_CLASSES = getattr(cfg, "NUM_CLASSES", 2)

# Optional model hyperparameters from config.py
STEM_DIM = getattr(cfg, "STEM_DIM", 64)
BRANCH_DIMS = getattr(cfg, "BRANCH_DIMS", [64, 128, 256, 512])
STAGE_DEPTHS = getattr(cfg, "STAGE_DEPTHS", [2, 2, 2, 2])
NUM_HEADS = getattr(cfg, "NUM_HEADS", [4, 8, 8, 16])
WINDOW_SIZES = getattr(cfg, "WINDOW_SIZES", [7, 7, 7, 7])
SPARSE_STRIDES = getattr(cfg, "SPARSE_STRIDES", [4, 4, 2, 2])
MLP_RATIO = getattr(cfg, "MLP_RATIO", 4.0)
DROPOUT = getattr(cfg, "DROPOUT", 0.1)
ATTN_DROPOUT = getattr(cfg, "ATTN_DROPOUT", 0.1)
USE_DEPTH_STREAM = getattr(cfg, "USE_DEPTH_STREAM", True)
DEPTH_IN_CHANS = getattr(cfg, "DEPTH_IN_CHANS", 3)
MULTISCALE_KERNELS = getattr(cfg, "MULTISCALE_KERNELS", [3, 5, 7, 9])


# =========================================================
# Basic layers
# =========================================================
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        # x: [B, C, H, W]
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=None, groups=1):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class ConvStem(nn.Module):
    """
    Low-level feature extractor before transformer stages.
    Preserves spatial structure better than direct ViT patchify.
    """
    def __init__(self, in_chans, out_chans=STEM_DIM):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(in_chans, out_chans // 2, kernel_size=3, stride=1),
            ConvBNAct(out_chans // 2, out_chans, kernel_size=3, stride=1),
        )

    def forward(self, x):
        return self.stem(x)


# =========================================================
# Multi-scale token embedding
# =========================================================
class MultiScaleTokenEmbedding(nn.Module):
    """
    Paper-inspired multi-scale token embedding:
    multiple convolutional patch extractors at different scales,
    then concatenate and project.
    """
    def __init__(self, in_chans, embed_dim, kernels=(3, 5, 7, 9), stride=2):
        super().__init__()

        num_scales = len(kernels)
        assert embed_dim % num_scales == 0, "embed_dim must be divisible by number of kernels"

        per_scale_dim = embed_dim // num_scales
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_chans,
                    per_scale_dim,
                    kernel_size=k,
                    stride=stride,
                    padding=k // 2,
                    bias=False,
                ),
                nn.BatchNorm2d(per_scale_dim),
                nn.GELU(),
            )
            for k in kernels
        ])

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

    def forward(self, x):
        feats = [branch(x) for branch in self.branches]
        x = torch.cat(feats, dim=1)
        x = self.proj(x)
        return x


# =========================================================
# Weighted Multi-Head Self Attention
# =========================================================
class WeightedMultiHeadSelfAttention(nn.Module):
    """
    Paper-inspired weighted multi-head self-attention.
    Each head has learnable importance weight.
    Input shape: [B, N, C]
    """
    def __init__(self, dim, num_heads=8, attn_dropout=0.1, proj_dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.head_weights = nn.Parameter(torch.ones(num_heads))
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x)  # [B, N, 3C]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # [B, heads, N, head_dim]

        # Learnable head weighting
        head_w = torch.sigmoid(self.head_weights).view(1, self.num_heads, 1, 1)
        out = out * head_w

        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


# =========================================================
# Window utilities
# =========================================================
def window_partition(x, window_size):
    """
    x: [B, C, H, W]
    return windows: [B*num_windows, window_size*window_size, C]
    """
    B, C, H, W = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size

    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h))

    Hp, Wp = x.shape[2], x.shape[3]

    x = x.view(
        B,
        C,
        Hp // window_size,
        window_size,
        Wp // window_size,
        window_size,
    )
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    windows = x.view(-1, window_size * window_size, C)

    return windows, Hp, Wp, pad_h, pad_w


def window_reverse(windows, window_size, B, C, Hp, Wp, pad_h, pad_w):
    """
    windows: [B*num_windows, window_size*window_size, C]
    return x: [B, C, H, W]
    """
    x = windows.view(
        B,
        Hp // window_size,
        Wp // window_size,
        window_size,
        window_size,
        C,
    )
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    x = x.view(B, C, Hp, Wp)

    if pad_h > 0 or pad_w > 0:
        x = x[:, :, :Hp - pad_h, :Wp - pad_w]

    return x


# =========================================================
# Attention blocks on feature maps
# =========================================================
class WindowLocalAttention(nn.Module):
    """
    Local attention inside non-overlapping windows.
    """
    def __init__(self, dim, num_heads, window_size=7, dropout=0.1, attn_dropout=0.1):
        super().__init__()
        self.window_size = window_size
        self.attn = WeightedMultiHeadSelfAttention(
            dim=dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        windows, Hp, Wp, pad_h, pad_w = window_partition(x, self.window_size)
        windows = self.attn(windows)
        x = window_reverse(windows, self.window_size, B, C, Hp, Wp, pad_h, pad_w)
        return x


class SparseGlobalAttention(nn.Module):
    """
    Sparse global attention:
    sample sparse tokens from whole feature map,
    apply global attention on them, then upsample back.
    """
    def __init__(self, dim, num_heads, sparse_stride=4, dropout=0.1, attn_dropout=0.1):
        super().__init__()
        self.sparse_stride = sparse_stride
        self.attn = WeightedMultiHeadSelfAttention(
            dim=dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
        )

    def forward(self, x):
        B, C, H, W = x.shape

        sparse = x[:, :, ::self.sparse_stride, ::self.sparse_stride]  # [B, C, Hs, Ws]
        Hs, Ws = sparse.shape[2], sparse.shape[3]

        tokens = sparse.flatten(2).transpose(1, 2)  # [B, N, C]
        tokens = self.attn(tokens)
        sparse_out = tokens.transpose(1, 2).reshape(B, C, Hs, Ws)

        out = F.interpolate(
            sparse_out,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )
        return out


class ConvFFN(nn.Module):
    """
    Paper-inspired FFN with pointwise + depthwise + pointwise convs.
    """
    def __init__(self, dim, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)

        self.fc1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.dwconv = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_dim,
        )
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class HybridTransformerBlock(nn.Module):
    """
    Window-local attention + sparse-global attention + conv FFN
    """
    def __init__(
        self,
        dim,
        num_heads,
        window_size=7,
        sparse_stride=4,
        mlp_ratio=4.0,
        dropout=0.1,
        attn_dropout=0.1,
    ):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.local_attn = WindowLocalAttention(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            dropout=dropout,
            attn_dropout=attn_dropout,
        )

        self.norm2 = LayerNorm2d(dim)
        self.global_attn = SparseGlobalAttention(
            dim=dim,
            num_heads=num_heads,
            sparse_stride=sparse_stride,
            dropout=dropout,
            attn_dropout=attn_dropout,
        )

        self.norm3 = LayerNorm2d(dim)
        self.ffn = ConvFFN(dim=dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x):
        x = x + self.local_attn(self.norm1(x))
        x = x + self.global_attn(self.norm2(x))
        x = x + self.ffn(self.norm3(x))
        return x


# =========================================================
# HRNet-style multi-branch machinery
# =========================================================
class TransitionLayer(nn.Module):
    """
    Build target branches from previous stage outputs.
    """
    def __init__(self, prev_dims, curr_dims):
        super().__init__()
        self.prev_dims = prev_dims
        self.curr_dims = curr_dims

        self.transforms = nn.ModuleList()

        for i, curr_dim in enumerate(curr_dims):
            if i < len(prev_dims):
                if prev_dims[i] == curr_dim:
                    self.transforms.append(nn.Identity())
                else:
                    self.transforms.append(
                        nn.Sequential(
                            nn.Conv2d(prev_dims[i], curr_dim, kernel_size=3, stride=1, padding=1, bias=False),
                            nn.BatchNorm2d(curr_dim),
                            nn.GELU(),
                        )
                    )
            else:
                ops = []
                in_dim = prev_dims[-1]
                for k in range(i + 1 - len(prev_dims)):
                    out_dim = curr_dim if k == (i - len(prev_dims)) else in_dim * 2
                    ops.extend([
                        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(out_dim),
                        nn.GELU(),
                    ])
                    in_dim = out_dim
                self.transforms.append(nn.Sequential(*ops))

    def forward(self, x_list):
        out = []
        for i, transform in enumerate(self.transforms):
            if i < len(x_list):
                out.append(transform(x_list[i]))
            else:
                out.append(transform(x_list[-1]))
        return out


class FuseLayer(nn.Module):
    """
    Cross-branch fusion:
    resize every branch to target branch resolution,
    align channels, then sum.
    """
    def __init__(self, branch_dims):
        super().__init__()
        self.branch_dims = branch_dims
        self.num_branches = len(branch_dims)

        self.fuse_layers = nn.ModuleList()
        for target in range(self.num_branches):
            row = nn.ModuleList()
            for source in range(self.num_branches):
                if source == target:
                    row.append(nn.Identity())
                else:
                    row.append(
                        nn.Sequential(
                            nn.Conv2d(branch_dims[source], branch_dims[target], kernel_size=1, bias=False),
                            nn.BatchNorm2d(branch_dims[target]),
                        )
                    )
            self.fuse_layers.append(row)

        self.act = nn.GELU()

    def forward(self, x_list):
        fused = []
        for target_idx in range(self.num_branches):
            target_h, target_w = x_list[target_idx].shape[2:]
            y = None
            for source_idx in range(self.num_branches):
                feat = self.fuse_layers[target_idx][source_idx](x_list[source_idx])

                if feat.shape[2:] != (target_h, target_w):
                    feat = F.interpolate(
                        feat,
                        size=(target_h, target_w),
                        mode="bilinear",
                        align_corners=False,
                    )

                y = feat if y is None else y + feat

            fused.append(self.act(y))
        return fused


class HRStage(nn.Module):
    def __init__(
        self,
        branch_dims,
        stage_depth,
        num_heads,
        window_size,
        sparse_stride,
        mlp_ratio=4.0,
        dropout=0.1,
        attn_dropout=0.1,
    ):
        super().__init__()
        self.num_branches = len(branch_dims)

        self.branches = nn.ModuleList([
            nn.Sequential(*[
                HybridTransformerBlock(
                    dim=branch_dims[i],
                    num_heads=num_heads[i],
                    window_size=window_size[i],
                    sparse_stride=sparse_stride[i],
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                )
                for _ in range(stage_depth)
            ])
            for i in range(self.num_branches)
        ])

        self.fuse = FuseLayer(branch_dims)

    def forward(self, x_list):
        x_list = [branch(x) for branch, x in zip(self.branches, x_list)]
        x_list = self.fuse(x_list)
        return x_list


class StreamHead(nn.Module):
    """
    Aggregate all branch features from one stream.
    """
    def __init__(self, branch_dims, out_dim=512):
        super().__init__()
        total_dim = sum(branch_dims)
        self.proj = nn.Sequential(
            nn.Conv2d(total_dim, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x_list):
        target_size = x_list[0].shape[2:]
        feats = []
        for x in x_list:
            if x.shape[2:] != target_size:
                x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
            feats.append(x)

        x = torch.cat(feats, dim=1)
        x = self.proj(x)
        x = self.pool(x).flatten(1)
        return x


# =========================================================
# Single HR-ViT stream
# =========================================================
class HRViTStream(nn.Module):
    def __init__(
        self,
        in_chans=3,
        stem_dim=STEM_DIM,
        branch_dims=BRANCH_DIMS,
        stage_depths=STAGE_DEPTHS,
        num_heads=NUM_HEADS,
        window_sizes=WINDOW_SIZES,
        sparse_strides=SPARSE_STRIDES,
        mlp_ratio=MLP_RATIO,
        dropout=DROPOUT,
        attn_dropout=ATTN_DROPOUT,
        multiscale_kernels=MULTISCALE_KERNELS,
    ):
        super().__init__()

        assert len(branch_dims) == 4, "This implementation expects 4 HR branches"
        assert len(stage_depths) == 4
        assert len(num_heads) == 4
        assert len(window_sizes) == 4
        assert len(sparse_strides) == 4

        self.stem = ConvStem(in_chans=in_chans, out_chans=stem_dim)
        self.token_embed = MultiScaleTokenEmbedding(
            in_chans=stem_dim,
            embed_dim=branch_dims[0],
            kernels=multiscale_kernels,
            stride=2,
        )

        # Stage 1 -> 1 branch
        self.stage1 = HRStage(
            branch_dims=[branch_dims[0]],
            stage_depth=stage_depths[0],
            num_heads=[num_heads[0]],
            window_size=[window_sizes[0]],
            sparse_stride=[sparse_strides[0]],
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attn_dropout=attn_dropout,
        )

        # Transition to 2 branches
        self.transition1 = TransitionLayer(
            prev_dims=[branch_dims[0]],
            curr_dims=branch_dims[:2],
        )
        self.stage2 = HRStage(
            branch_dims=branch_dims[:2],
            stage_depth=stage_depths[1],
            num_heads=num_heads[:2],
            window_size=window_sizes[:2],
            sparse_stride=sparse_strides[:2],
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attn_dropout=attn_dropout,
        )

        # Transition to 3 branches
        self.transition2 = TransitionLayer(
            prev_dims=branch_dims[:2],
            curr_dims=branch_dims[:3],
        )
        self.stage3 = HRStage(
            branch_dims=branch_dims[:3],
            stage_depth=stage_depths[2],
            num_heads=num_heads[:3],
            window_size=window_sizes[:3],
            sparse_stride=sparse_strides[:3],
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attn_dropout=attn_dropout,
        )

        # Transition to 4 branches
        self.transition3 = TransitionLayer(
            prev_dims=branch_dims[:3],
            curr_dims=branch_dims[:4],
        )
        self.stage4 = HRStage(
            branch_dims=branch_dims[:4],
            stage_depth=stage_depths[3],
            num_heads=num_heads[:4],
            window_size=window_sizes[:4],
            sparse_stride=sparse_strides[:4],
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attn_dropout=attn_dropout,
        )

        self.head = StreamHead(branch_dims=branch_dims, out_dim=512)

    def forward(self, x):
        x = self.stem(x)
        x = self.token_embed(x)

        x_list = [x]
        x_list = self.stage1(x_list)

        x_list = self.transition1(x_list)
        x_list = self.stage2(x_list)

        x_list = self.transition2(x_list)
        x_list = self.stage3(x_list)

        x_list = self.transition3(x_list)
        x_list = self.stage4(x_list)

        out = self.head(x_list)
        return out


# =========================================================
# Final SpoofFormerNet
# =========================================================
class SpoofFormer(nn.Module):
    """
    Paper-inspired Spoof-formerNet:
    - RGB stream
    - Depth stream
    - Feature fusion
    - Classification head
    """

    def __init__(
        self,
        img_size=IMAGE_SIZE,
        rgb_in_chans=NUM_CHANNELS,
        depth_in_chans=DEPTH_IN_CHANS,
        num_classes=NUM_CLASSES,
    ):
        super().__init__()
        self.img_size = img_size
        self.use_depth_stream = USE_DEPTH_STREAM

        self.rgb_stream = HRViTStream(in_chans=rgb_in_chans)

        if self.use_depth_stream:
            self.depth_stream = HRViTStream(in_chans=depth_in_chans)
            fusion_in_dim = 512 + 512
        else:
            self.depth_stream = None
            fusion_in_dim = 512

        self.classifier = nn.Sequential(
            nn.Linear(fusion_in_dim, 512),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm2d, LayerNorm2d)):
                if hasattr(module, "weight") and module.weight is not None:
                    nn.init.constant_(module.weight, 1.0)
                if hasattr(module, "bias") and module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, rgb, depth=None):
        """
        Supported usage:
        1) model(rgb, depth)
        2) model(x) where x has 6 channels => first 3 RGB, next 3 depth
        """

        if self.use_depth_stream:
            if depth is None:
                if rgb.dim() == 4 and rgb.size(1) == 6:
                    rgb, depth = rgb[:, :3], rgb[:, 3:6]
                else:
                    raise ValueError(
                        "This model expects RGB and Depth inputs. "
                        "Pass model(rgb, depth) or a 6-channel tensor."
                    )

            rgb_feat = self.rgb_stream(rgb)
            depth_feat = self.depth_stream(depth)
            fused = torch.cat([rgb_feat, depth_feat], dim=1)
            logits = self.classifier(fused)
            return logits

        rgb_feat = self.rgb_stream(rgb)
        logits = self.classifier(rgb_feat)
        return logits