import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 固定 7×7 风车掩码（Pinwheel-shaped mask），与图示一致
PINWHEEL_MASK_7X7 = torch.tensor([
    [1, 0, 0, 1, 0, 0, 1],
    [0, 1, 0, 1, 0, 1, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 1, 0, 1, 0, 1, 0],
    [1, 0, 0, 1, 0, 0, 1],
], dtype=torch.float32)  # (7, 7)

# 五路固定 7×7 二值旋转掩码（图示对应），用于 RotWeight 可学习加权
# 1) [0, π/6] 30°  n=6 十字加粗
ROT_MASK_PI6 = torch.tensor([
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
], dtype=torch.float32)
# 2) [0, π/4] 45°  n=4 十字+对角
ROT_MASK_PI4 = torch.tensor([
    [1, 0, 0, 1, 0, 0, 1],
    [0, 1, 0, 1, 0, 1, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 1, 0, 1, 0, 1, 0],
    [1, 0, 0, 1, 0, 0, 1],
], dtype=torch.float32)
# 3) [0, π/3] 60°  n=3
ROT_MASK_PI3 = torch.tensor([
    [0, 1, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 1, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 1, 0],
], dtype=torch.float32)
# 4) [0, π/2] 90°  n=2 十字
ROT_MASK_PI2 = torch.tensor([
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
], dtype=torch.float32)
# 5) [π/2, π/3]
ROT_MASK_PI2_PI3 = torch.tensor([
    [0, 0, 0, 1, 0, 0, 0],
    [1, 0, 0, 1, 0, 0, 1],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [1, 0, 0, 1, 0, 0, 1],
    [0, 0, 0, 1, 0, 0, 0],
], dtype=torch.float32)
# RotWeight 仅用 3 路：PI4(45°)、PI3(60°)、[π/2,π/3]
ROT_WEIGHT_MASKS = [ROT_MASK_PI4, ROT_MASK_PI3, ROT_MASK_PI2_PI3]
NUM_ROT_WEIGHT = 3
# RotWeight2 仅用 2 路（不含风车）：PI3(60°)、[π/2,π/3]
ROT_WEIGHT_MASKS_TWO = [ROT_MASK_PI6, ROT_MASK_PI2]
NUM_ROT_WEIGHT_TWO = 2
# 第 0 路 ROT_MASK_PI4 与固定风车 PINWHEEL_MASK_7X7 完全相同，用于“从风车出发”的初始化
PINWHEEL_ROT_WEIGHT_INDEX = 0


def _gaussian_kernel_7x7(sigma):
    """固定 7×7 网格上的高斯权重: exp(-(i^2 + j^2) / (2*σ^2))，中心为 (3,3)。"""
    # 相对中心的坐标 i, j in [-3, 3]
    grid = torch.arange(7, dtype=torch.float32, device=sigma.device) - 3  # [-3,-2,...,3]
    i = grid.view(7, 1).expand(7, 7)
    j = grid.view(1, 7).expand(7, 7)
    g = torch.exp(-(i * i + j * j) / (2.0 * sigma * sigma + 1e-8))
    return g


def _make_uv_grid_7x7(device, dtype=torch.float32):
    """7×7 网格坐标 u,v，中心 (3,3) 对应 (0,0)。返回 u(7,7), v(7,7)。"""
    m = 3
    coords = torch.arange(-m, m + 1, dtype=dtype, device=device)
    u, v = torch.meshgrid(coords, coords, indexing='ij')
    return u, v


def _generate_7x7_learnable_rotated_pinwheel(
    init_angle_rad,
    rotate_angle_rad,
    u,
    v,
    m=3,
    tol=0.5,
    tau=None,
):
    """
    可微的 7×7 旋转风车掩码（线带+圆盘，sigmoid 软约束）。
    init_angle_rad, rotate_angle_rad, tau 为可学习张量时梯度可回传。
    返回 (7,7)，值∈[0,1]。
    """
    target_rad = (init_angle_rad + rotate_angle_rad).reshape(-1)[0]
    if tau is None:
        tau_val = 0.1
    else:
        tau_val = tau.clamp(1e-3, 1.0).reshape(-1)[0]
    r = torch.sqrt(u * u + v * v + 1e-8)
    disk_mask = torch.sigmoid((float(m) - r) / tau_val)
    dist = torch.abs(-torch.sin(target_rad) * u + torch.cos(target_rad) * v)
    line_mask = torch.sigmoid((tol - dist) / tau_val)
    pinwheel_mask = disk_mask * line_mask
    pinwheel_mask = pinwheel_mask.clone()
    pinwheel_mask[m, m] = 1.0
    return pinwheel_mask


def _rotate_pinwheel_7x7(pinwheel_1177, theta, align_corners=False):
    """
    对 (1,1,7,7) 的风车掩码绕中心旋转 theta 弧度，参考多角度掩码思想。
    返回 (7,7)，与 pinwheel_1177 同 device。
    """
    # 7x7 中心 (3,3)。输出 (i,j) 取自源 (i',j') = R(-θ)*(i-3,j-3) + (3,3)
    device = pinwheel_1177.device
    dtype = pinwheel_1177.dtype
    c = math.cos(theta)
    s = math.sin(theta)
    ii = torch.arange(7, device=device, dtype=dtype) - 3  # [-3..3]
    i = ii.view(7, 1).expand(7, 7)
    j = ii.view(1, 7).expand(7, 7)
    # 源坐标 i' = 3 + c*(i-3) - s*(j-3), j' = 3 + s*(i-3) + c*(j-3)
    ip = 3 + c * (i - 3) - s * (j - 3)
    jp = 3 + s * (i - 3) + c * (j - 3)
    if align_corners:
        # 像素中心在 0..6，归一化到 [-1,1]
        norm_x = (2 * jp / 6) - 1
        norm_y = (2 * ip / 6) - 1
    else:
        norm_x = (2 * (jp + 0.5) / 7) - 1
        norm_y = (2 * (ip + 0.5) / 7) - 1
    grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)  # (1,7,7,2)
    out = F.grid_sample(
        pinwheel_1177,
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=align_corners,
    )
    return out.squeeze(0).squeeze(0)  # (7,7)


class GaussianPinwheelSpatialAttention(nn.Module):
    """
    使用 σ 自适应的 7×7 高斯卷积 + 风车掩码的空间注意力。
    - 默认：固定风车掩码 + 可学习 σ。
    - use_rotated_pinwheel=True (Pipeline A)：多角度旋转固定风车并 argmax 聚合。
    - gp_pipeline_b=True (Pipeline B)：线积分能量 -> k -> gather，二值 mask。
    - use_learnable_rotated_pinwheel=True：可学习旋转掩码。
    - use_rot_weight / use_rot_weight_two：可学习权重。
    """
    def __init__(
        self,
        in_channels=2,
        out_channels=1,
        kernel_size=7,
        sigma_init=1.5,
        use_rotated_pinwheel=False,
        n_orientations=4,
        use_learnable_rotated_pinwheel=False,
        use_rot_weight=False,
        use_rot_weight_two=False,
        gp_pipeline_b=False,
        tol=0.5,
    ):
        super(GaussianPinwheelSpatialAttention, self).__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2  # 3 for 7x7
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_rotated_pinwheel = use_rotated_pinwheel
        self.n_orientations = max(1, int(n_orientations))
        self.use_learnable_rotated_pinwheel = use_learnable_rotated_pinwheel
        self.use_rot_weight = use_rot_weight
        self.use_rot_weight_two = use_rot_weight_two
        self.gp_pipeline_b = gp_pipeline_b
        self.tol = tol
        # 可学习 log(σ)，每层一个标量；σ = exp(log_sigma) 保证为正
        self.log_sigma = nn.Parameter(torch.tensor(math.log(max(sigma_init, 0.1))))
        # 注册风车掩码为 buffer（不参与梯度，但随模型移动设备）
        self.register_buffer('pinwheel', PINWHEEL_MASK_7X7.unsqueeze(0).unsqueeze(0))  # (1,1,7,7)
        # 五路固定旋转掩码 + 可学习权重（仅 use_rot_weight 时注册，避免旧 checkpoint 多 key）
        if use_rot_weight:
            for i, m in enumerate(ROT_WEIGHT_MASKS):
                self.register_buffer(f'_rot_weight_mask_{i}', m.unsqueeze(0).unsqueeze(0))  # (1,1,7,7)
            # 从“近似固定风车”出发：第 PINWHEEL_ROT_WEIGHT_INDEX 路即风车掩码，初始 logit 偏大，
            # 使 softmax 接近 (0,1,0,0,0)，至少可匹配固定 GP，再学习是否混合更优
            init_logits = torch.zeros(NUM_ROT_WEIGHT)
            init_logits[PINWHEEL_ROT_WEIGHT_INDEX] = 5.0
            self.rot_weight_logits = nn.Parameter(init_logits)
        # 仅两路（PI3、PI2_PI3）不含风车，2 维权重均等初始化
        if use_rot_weight_two:
            for i, m in enumerate(ROT_WEIGHT_MASKS_TWO):
                self.register_buffer(f'_rot_weight_two_mask_{i}', m.unsqueeze(0).unsqueeze(0))
            self.rot_weight_two_logits = nn.Parameter(torch.zeros(NUM_ROT_WEIGHT_TWO))
        # 仅当使用可学习旋转掩码时注册，避免旧 GP 权重加载时多出 missing keys
        if use_learnable_rotated_pinwheel:
            u, v = _make_uv_grid_7x7(torch.device('cpu'), torch.float32)
            self.register_buffer('_u', u)
            self.register_buffer('_v', v)
            self.init_angle_rad = nn.Parameter(torch.tensor(0.0))
            self.rotate_angle_rad = nn.Parameter(torch.tensor(math.pi / 6.0))
            self.log_tau = nn.Parameter(torch.tensor(math.log(0.1)))
        if gp_pipeline_b:
            from model.gpconv_7x7 import GPConv7x7PipelineB
            self.pipeline_b = GPConv7x7PipelineB(n_orientations=self.n_orientations, tol=tol)

    def _get_kernel(self, mask_77=None):
        """mask_77: (7,7) 或 None 表示使用默认风车。返回 (7,7)。"""
        sigma = torch.exp(self.log_sigma).clamp(min=1e-2, max=10.0)
        ref_device = self.pinwheel.device
        g = _gaussian_kernel_7x7(sigma).to(ref_device)
        if mask_77 is None:
            mask_77 = self.pinwheel.squeeze(0).squeeze(0)
        k = (g * mask_77)
        k = k / (k.sum() + 1e-8)
        return k

    def _get_learnable_mask_7x7(self):
        """可学习旋转掩码 (7,7)。前向使用二值 0/1 以保持高斯核特性，反向用 straight-through 使角度可学。"""
        u = self._u.to(dtype=self.init_angle_rad.dtype)
        v = self._v.to(dtype=self.init_angle_rad.dtype)
        tau = torch.exp(self.log_tau).clamp(1e-3, 1.0)
        soft = _generate_7x7_learnable_rotated_pinwheel(
            self.init_angle_rad,
            self.rotate_angle_rad,
            u,
            v,
            m=3,
            tol=self.tol,
            tau=tau,
        )
        binary = (soft > 0.5).float()
        return binary + (soft - soft.detach())

    def forward(self, x):
        # x: (B, 2, H, W) 来自 avg/max
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)

        if self.gp_pipeline_b:
            return torch.sigmoid(self.pipeline_b(x))

        if self.use_rot_weight_two:
            # 仅两路（PI3、PI2_PI3）不含风车
            sigma = torch.exp(self.log_sigma).clamp(min=1e-2, max=10.0)
            g = _gaussian_kernel_7x7(sigma).to(x.device)
            responses = []
            for i in range(NUM_ROT_WEIGHT_TWO):
                mask_77 = getattr(self, f'_rot_weight_two_mask_{i}').squeeze(0).squeeze(0)
                k = (g * mask_77) / (g * mask_77).sum().clamp(min=1e-8)
                w = k.unsqueeze(0).unsqueeze(0).expand(1, 2, 7, 7).clone()
                y = F.conv2d(x, w, padding=self.padding)
                responses.append(y)
            stack = torch.cat(responses, dim=1)
            weights = F.softmax(self.rot_weight_two_logits, dim=0).view(1, -1, 1, 1)
            out = (stack * weights).sum(dim=1, keepdim=True)
            return torch.sigmoid(out)

        if self.use_rot_weight:
            # 三路固定掩码：每路 kernel = Gaussian(σ) * mask，归一化后卷积，再可学习权重加权
            sigma = torch.exp(self.log_sigma).clamp(min=1e-2, max=10.0)
            g = _gaussian_kernel_7x7(sigma).to(x.device)
            responses = []
            for i in range(NUM_ROT_WEIGHT):
                mask_77 = getattr(self, f'_rot_weight_mask_{i}').squeeze(0).squeeze(0)
                k = (g * mask_77) / (g * mask_77).sum().clamp(min=1e-8)
                w = k.unsqueeze(0).unsqueeze(0).expand(1, 2, 7, 7).clone()
                y = F.conv2d(x, w, padding=self.padding)  # (B,1,H,W)
                responses.append(y)
            stack = torch.cat(responses, dim=1)
            weights = F.softmax(self.rot_weight_logits, dim=0)
            weights = weights.view(1, -1, 1, 1)
            out = (stack * weights).sum(dim=1, keepdim=True)
            return torch.sigmoid(out)

        if self.use_learnable_rotated_pinwheel:
            # 可学习旋转掩码：高斯（不变）× 可微旋转线带掩码，再卷积
            mask_77 = self._get_learnable_mask_7x7()
            k = self._get_kernel(mask_77)
            w = k.unsqueeze(0).unsqueeze(0).expand(1, 2, 7, 7).clone()
            x = F.conv2d(x, w, padding=self.padding)
            return torch.sigmoid(x)

        if not self.use_rotated_pinwheel:
            # 原有逻辑：单核 高斯×固定风车
            k = self._get_kernel()
            w = k.unsqueeze(0).unsqueeze(0).expand(1, 2, 7, 7).clone()
            x = F.conv2d(x, w, padding=self.padding)
            return torch.sigmoid(x)

        # 旋转掩码分支：n 个角度 θ_k = k*π/n，每个核 = 高斯 × rotate(风车, θ_k)，再聚合
        n = self.n_orientations
        pi = math.pi
        responses = []
        for k in range(n):
            theta = k * pi / float(n)
            rot_mask = _rotate_pinwheel_7x7(self.pinwheel, theta)  # (7,7)
            kernel_k = self._get_kernel(rot_mask)  # (7,7)
            w_k = kernel_k.unsqueeze(0).unsqueeze(0).expand(1, 2, 7, 7).clone()
            y_k = F.conv2d(x, w_k, padding=self.padding)  # (B,1,H,W)
            responses.append(y_k)
        stack = torch.cat(responses, dim=1)  # (B, n, H, W)
        idx = stack.abs().argmax(dim=1, keepdim=True)
        x = torch.gather(stack, 1, idx)  # (B, 1, H, W)
        return torch.sigmoid(x)


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

class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_gaussian_pinwheel=False, use_rotated_pinwheel=False, n_orientations=4, use_learnable_rotated_pinwheel=False, use_rot_weight=False, use_rot_weight_two=False, gp_pipeline_b=False):
        super(ResNet, self).__init__()
        self._use_rotated_pinwheel = use_rotated_pinwheel
        self._n_orientations = n_orientations
        self._use_learnable_rotated_pinwheel = use_learnable_rotated_pinwheel
        self._use_rot_weight = use_rot_weight
        self._use_rot_weight_two = use_rot_weight_two
        self._gp_pipeline_b = gp_pipeline_b
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
        if use_gaussian_pinwheel:
            self.sa = GaussianPinwheelSpatialAttention(
                in_channels=2, out_channels=1, kernel_size=7,
                use_rotated_pinwheel=getattr(self, '_use_rotated_pinwheel', False),
                n_orientations=getattr(self, '_n_orientations', 4),
                use_learnable_rotated_pinwheel=getattr(self, '_use_learnable_rotated_pinwheel', False),
                use_rot_weight=getattr(self, '_use_rot_weight', False),
                use_rot_weight_two=getattr(self, '_use_rot_weight_two', False),
                gp_pipeline_b=getattr(self, '_gp_pipeline_b', False),
            )
        else:
            self.sa = SpatialAttention(kernel_size=7)

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
    def __init__(self, input_channels, block=ResNet, use_gaussian_pinwheel=False, use_rotated_pinwheel=False, n_orientations=4, use_learnable_rotated_pinwheel=False, use_rot_weight=False, use_rot_weight_two=False, gp_pipeline_b=False):
        super().__init__()
        self.use_gaussian_pinwheel = use_gaussian_pinwheel
        self.use_rotated_pinwheel = use_rotated_pinwheel
        self.n_orientations = n_orientations
        self.use_learnable_rotated_pinwheel = use_learnable_rotated_pinwheel
        self.use_rot_weight = use_rot_weight
        self.use_rot_weight_two = use_rot_weight_two
        self.gp_pipeline_b = gp_pipeline_b
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


    def _make_layer(self, in_channels, out_channels, block, block_num=1):
        layer = []
        kw = dict(use_gaussian_pinwheel=self.use_gaussian_pinwheel, use_rotated_pinwheel=self.use_rotated_pinwheel, n_orientations=self.n_orientations, use_learnable_rotated_pinwheel=self.use_learnable_rotated_pinwheel, use_rot_weight=self.use_rot_weight, use_rot_weight_two=self.use_rot_weight_two, gp_pipeline_b=self.gp_pipeline_b)
        layer.append(block(in_channels, out_channels, **kw))
        for _ in range(block_num - 1):
            layer.append(block(out_channels, out_channels, **kw))
        return nn.Sequential(*layer)

    def forward(self, x, warm_flag):
        x_e0 = self.encoder_0(self.conv_init(x))
        x_e1 = self.encoder_1(self.pool(x_e0))
        x_e2 = self.encoder_2(self.pool(x_e1))
        x_e3 = self.encoder_3(self.pool(x_e2))

        x_m = self.middle_layer(self.pool(x_e3))

        x_d3 = self.decoder_3(torch.cat([x_e3, self.up(x_m)], 1))
        x_d2 = self.decoder_2(torch.cat([x_e2, self.up(x_d3)], 1))
        x_d1 = self.decoder_1(torch.cat([x_e1, self.up(x_d2)], 1))
        x_d0 = self.decoder_0(torch.cat([x_e0, self.up(x_d1)], 1))

        
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

       
    