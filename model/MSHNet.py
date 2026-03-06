import torch
import torch.nn as nn
import torch.nn.functional as F

from model.edge_sma import LocalDiffModule, SMAModule
from model.gp_attention import GaussianPinwheelSpatialAttention

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class LDRBlock(nn.Module):
    """
    Lightweight Dilated Residual Block (LDRB, EATUNet-inspired).
    替换 ResNet block 里的 conv1：用 3 条并行扩张卷积（dilation=1/2/3）处理不同感受野，
    concat 后恢复 out_channels，再经 conv2 + CA + SA + residual，其余逻辑与 ResNet 完全一致。
    接口与 ResNet(in_channels, out_channels, stride) 相同，可直接作为 block= 参数传入。

    感受野对比（3×3 卷积，dilation d）：
        d=1  -> 3×3  (有效感受野 3)
        d=2  -> 5×5  (有效感受野 7)
        d=3  -> 7×7  (有效感受野 13)
    三条分支 concat 后覆盖从细粒度到更大局部背景的多尺度信息。
    """
    DILATIONS = (1, 2, 3)

    def __init__(self, in_channels, out_channels, stride=1, **sa_kwargs):
        super().__init__()
        n = len(self.DILATIONS)
        # 均匀分配通道：余数补到前面的分支
        base = out_channels // n
        rem  = out_channels - base * n
        branch_chs = [base + (1 if i < rem else 0) for i in range(n)]

        # 三条并行扩张卷积（各自 BN）
        self.branches = nn.ModuleList([
            nn.Conv2d(in_channels, ch, kernel_size=3, stride=stride,
                      padding=d, dilation=d, bias=False)
            for d, ch in zip(self.DILATIONS, branch_chs)
        ])
        self.branch_bns = nn.ModuleList([nn.BatchNorm2d(ch) for ch in branch_chs])

        self.relu = nn.ReLU(inplace=True)

        # conv2 与原 ResNet 相同
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = _build_spatial_attention(sa_kwargs)

    def forward(self, x):
        residual = x if self.shortcut is None else self.shortcut(x)

        # 三条并行扩张分支 -> ReLU -> concat
        branch_outs = [self.relu(bn(conv(x)))
                       for conv, bn in zip(self.branches, self.branch_bns)]
        out = torch.cat(branch_outs, dim=1)   # [B, out_channels, H, W]

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out += residual
        out = self.relu(out)
        return out


def _build_spatial_attention(sa_kwargs: dict) -> nn.Module:
    """根据 sa_kwargs 构造空间注意力模块（SpatialAttention 或 GP 变体）。"""
    if not sa_kwargs.get('use_gaussian_pinwheel', False):
        return SpatialAttention()
    return GaussianPinwheelSpatialAttention(
        kernel_size=7,
        use_rotated_pinwheel=sa_kwargs.get('use_rotated_pinwheel', False),
        n_orientations=sa_kwargs.get('n_orientations', 4),
        use_learnable_rotated_pinwheel=sa_kwargs.get('use_learnable_rotated_pinwheel', False),
        use_rot_weight=sa_kwargs.get('use_rot_weight', False),
        use_rot_weight_two=sa_kwargs.get('use_rot_weight_two', False),
    )


class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, **sa_kwargs):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = _build_spatial_attention(sa_kwargs)

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out += residual
        out = self.relu(out)
        return out

class MSHNet(nn.Module):
    def __init__(
        self,
        input_channels,
        block=ResNet,
        use_edge: bool = False,
        use_sma: bool = False,
        edge_kernel_size: int = 15,
        use_ldrb: bool = False,
        # GP（高斯-风车空间注意力）参数
        use_gaussian_pinwheel: bool = False,
        use_rotated_pinwheel: bool = False,
        n_orientations: int = 4,
        use_learnable_rotated_pinwheel: bool = False,
        use_rot_weight: bool = False,
        use_rot_weight_two: bool = False,
    ):
        """
        模块开关（支持任意组合消融）：
          use_ldrb   : 并行扩张卷积(d=1/2/3)替换普通 ResNet block → 多尺度感受野
          use_edge   : TFD 边缘感知分支 → decoder_0 输出 WSM 调制
          use_sma    : SMA 轻量多注意力(CA+SA) → decoder_0 输出残差增强
          use_gaussian_pinwheel : 7×7 高斯×风车掩码替换每个 block 的 SpatialAttention → 方向感知
            use_rotated_pinwheel          : 多角度旋转风车 argmax 聚合 (-Rot)
            use_learnable_rotated_pinwheel: 可学习旋转角度 + 可微掩码 (-LearnRot)
            use_rot_weight                : 三路固定掩码加权 (-RotWeight)
            use_rot_weight_two            : 两路掩码加权不含风车 (-RotWeight2)
        """
        super().__init__()
        self.use_edge = use_edge
        self.use_sma = use_sma
        if use_ldrb:
            block = LDRBlock
        # 传给每个 block 的空间注意力参数
        self._sa_kwargs = dict(
            use_gaussian_pinwheel=use_gaussian_pinwheel,
            use_rotated_pinwheel=use_rotated_pinwheel,
            n_orientations=n_orientations,
            use_learnable_rotated_pinwheel=use_learnable_rotated_pinwheel,
            use_rot_weight=use_rot_weight,
            use_rot_weight_two=use_rot_weight_two,
        )
        param_channels = [16, 32, 64, 128, 256]
        param_blocks = [2, 2, 2, 2]
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up_16 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.conv_init = nn.Conv2d(input_channels, param_channels[0], 1, 1)

        self.encoder_0 = self._make_layer(param_channels[0], param_channels[0], block)
        self.encoder_1 = self._make_layer(param_channels[0], param_channels[1], block, param_blocks[0])
        self.encoder_2 = self._make_layer(param_channels[1], param_channels[2], block, param_blocks[1])
        self.encoder_3 = self._make_layer(param_channels[2], param_channels[3], block, param_blocks[2])
     
        self.middle_layer = self._make_layer(param_channels[3], param_channels[4], block, param_blocks[3])
        
        self.decoder_3 = self._make_layer(param_channels[3]+param_channels[4], param_channels[3], block, param_blocks[2])
        self.decoder_2 = self._make_layer(param_channels[2]+param_channels[3], param_channels[2], block, param_blocks[1])
        self.decoder_1 = self._make_layer(param_channels[1]+param_channels[2], param_channels[1], block, param_blocks[0])
        self.decoder_0 = self._make_layer(param_channels[0]+param_channels[1], param_channels[0], block)

        self.output_0 = nn.Conv2d(param_channels[0], 1, 1)
        self.output_1 = nn.Conv2d(param_channels[1], 1, 1)
        self.output_2 = nn.Conv2d(param_channels[2], 1, 1)
        self.output_3 = nn.Conv2d(param_channels[3], 1, 1)

        self.final = nn.Conv2d(4, 1, 3, 1, 1)

        # 增量模块（消融实验可独立开关）
        if use_edge:
            # TFD 边缘感知：从原图算局部差分图，对 decoder_0 输出做 WSM 式调制
            # x_d0 = x_d0 * (1 + scale * edge_map)
            self.edge_branch = LocalDiffModule(
                kernel_size=edge_kernel_size,
                use_dilation=False,
                learnable_scale=True,
            )
        if use_sma:
            # SMA 轻量多注意力：decoder_0 输出后做 CA+SA 残差增强
            self.sma = SMAModule(channels=param_channels[0], reduction=4)


    def _make_layer(self, in_channels, out_channels, block, block_num=1):
        layer = []
        layer.append(block(in_channels, out_channels, **self._sa_kwargs))
        for _ in range(block_num - 1):
            layer.append(block(out_channels, out_channels, **self._sa_kwargs))
        return nn.Sequential(*layer)

    def forward(self, x, warm_flag):
        # 边缘分支：从原图算局部差分图，后面在 decoder_0 出口调制
        if self.use_edge:
            edge_map = self.edge_branch(x)  # [B, 1, H, W]

        x_e0 = self.encoder_0(self.conv_init(x))
        x_e1 = self.encoder_1(self.pool(x_e0))
        x_e2 = self.encoder_2(self.pool(x_e1))
        x_e3 = self.encoder_3(self.pool(x_e2))

        x_m = self.middle_layer(self.pool(x_e3))

        x_d3 = self.decoder_3(torch.cat([x_e3, self.up(x_m)], 1))
        x_d2 = self.decoder_2(torch.cat([x_e2, self.up(x_d3)], 1))
        x_d1 = self.decoder_1(torch.cat([x_e1, self.up(x_d2)], 1))
        x_d0 = self.decoder_0(torch.cat([x_e0, self.up(x_d1)], 1))

        # 边缘调制：WSM 式 x_d0 = x_d0 * (1 + scale * edge_map)
        if self.use_edge:
            x_d0 = x_d0 * (1.0 + self.edge_branch.scale * edge_map)

        # SMA 多注意力增强（CA+SA 残差）
        if self.use_sma:
            x_d0 = self.sma(x_d0)

        if warm_flag:
            mask0 = self.output_0(x_d0)
            mask1 = self.output_1(x_d1)
            mask2 = self.output_2(x_d2)
            mask3 = self.output_3(x_d3)
            output = self.final(torch.cat([mask0, self.up(mask1), self.up_4(mask2), self.up_8(mask3)], dim=1))
            return [mask0, mask1, mask2, mask3], output
    
        else:
            output = self.output_0(x_d0)
            return [], output

       
    