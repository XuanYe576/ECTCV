"""
Pipeline B for mmmpaperablation: line-energy -> hat_theta -> detach+quantize k -> gather.
- Kernel = mask only (L1-normalized), no Gaussian prior on the mask.
- Learnable scalar gain on the mask output.
- Scale 3->5->7: intended design is to compare 3x3 with/without Gaussian to see if it helps,
  then choose to step to 5x5, then 7x7 (not implemented here; current module is 7x7 only).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_uv_grid_7(device, dtype):
    m = 3
    coords = torch.arange(-m, m + 1, device=device, dtype=dtype)
    u = coords.view(1, 1, 7).expand(1, 7, 7)
    v = coords.view(1, 7, 1).expand(1, 7, 7)
    return u, v, m


def build_bank_7x7(n: int, tol: float = 0.5, eps: float = 1e-6, device=None, dtype=torch.float32):
    """Returns W: (n, 1, 7, 7), L1-normalized. Kernel = mask only (no Gaussian prior)."""
    if device is None:
        device = torch.device("cpu")
    u, v, m = _make_uv_grid_7(device, dtype)
    Delta = math.pi / float(n)
    theta0 = 0.5 * Delta
    thetas = (theta0 + torch.arange(n, device=device, dtype=dtype) * Delta) % math.pi
    thetas = thetas.view(n, 1, 1)
    d = torch.abs(-torch.sin(thetas) * u + torch.cos(thetas) * v)
    M = (d <= tol).to(dtype=dtype)
    M[:, m, m] = 1.0
    W = M.unsqueeze(1)
    denom = W.sum(dim=(1, 2, 3), keepdim=True) + eps
    W = W / denom
    return W


def _theta_to_k(hat_theta: torch.Tensor, n: int) -> torch.Tensor:
    theta_det = hat_theta.detach()
    phi = torch.remainder(theta_det, math.pi)
    k = (phi / (math.pi / n)).round().long().clamp(0, n - 1)
    return k.unsqueeze(1)


class LineEnergyHead7x7(nn.Module):
    """Line-integral energy on (B, C, H, W) -> hat_theta (B, H, W). Single scale s=7."""
    def __init__(self, Q: int = 24, N: int = 8, T_theta: float = 1.0, sigma_r: float = 1.0):
        super().__init__()
        self.Q = Q
        self.N = N
        self.T_theta = T_theta
        self.sigma_r = sigma_r
        self.r_s = 3.0

    def _make_flow_grid(self, H, W, dx, dy, device, dtype, B):
        y = torch.linspace(-1, 1, H, device=device, dtype=dtype)
        x = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(y, x, indexing="ij")
        gx_n = gx - 2 * dx / (W - 1) if W > 1 else gx
        gy_n = gy - 2 * dy / (H - 1) if H > 1 else gy
        grid = torch.stack([gx_n, gy_n], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        return grid

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B, C, H, W = X.shape
        device = X.device
        dtype = X.dtype
        F_map = X.norm(dim=1, keepdim=True)
        t_n = (torch.arange(1, self.N + 1, device=device, dtype=dtype) / self.N) * self.r_s
        w_n = torch.exp(-t_n * t_n / (2 * self.sigma_r ** 2))
        w_n = w_n / w_n.sum()
        theta_q = (2 * math.pi * torch.arange(self.Q, device=device, dtype=dtype) / self.Q)
        E_q = []
        for q in range(self.Q):
            c, s_ang = math.cos(theta_q[q].item()), math.sin(theta_q[q].item())
            samples = []
            for n in range(self.N):
                dy = t_n[n].item() * s_ang
                dx = t_n[n].item() * c
                grid = self._make_flow_grid(H, W, dx, dy, device, dtype, B)
                samp = F.grid_sample(F_map, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
                samples.append(samp)
            stack_s = torch.stack(samples, dim=0)
            E_n = (stack_s * w_n.view(self.N, 1, 1, 1, 1)).sum(dim=0)
            E_q.append(E_n)
        E_q = torch.stack(E_q, dim=1)
        pi_q = F.softmax(E_q / self.T_theta, dim=1)
        sin_q = torch.sin(theta_q).view(1, -1, 1, 1, 1)
        cos_q = torch.cos(theta_q).view(1, -1, 1, 1, 1)
        sy = (pi_q * sin_q).sum(dim=1).squeeze(1)
        cx = (pi_q * cos_q).sum(dim=1).squeeze(1)
        hat_theta = torch.atan2(sy, cx)
        hat_theta = torch.remainder(hat_theta + 2 * math.pi, 2 * math.pi)
        return hat_theta.squeeze(1)


class GPConv7x7PipelineB(nn.Module):
    """Pipeline B: line-energy -> k -> mask-only bank gather + learnable scalar gain. No Gaussian on mask."""
    def __init__(self, n_orientations: int = 4, Q: int = 24, N: int = 8, T_theta: float = 1.0, tol: float = 0.5):
        super().__init__()
        self.n = max(1, n_orientations)
        self.line_energy = LineEnergyHead7x7(Q=Q, N=N, T_theta=T_theta, sigma_r=1.0)
        W = build_bank_7x7(self.n, tol=tol)
        self.register_buffer("_W", W)
        self.gain = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 2, H, W). Returns (B, 1, H, W)."""
        hat_theta = self.line_energy(x)
        k = _theta_to_k(hat_theta, self.n)
        W = self._W
        w = W.expand(self.n, 2, 7, 7).clone()
        y = F.conv2d(x, w, padding=3)
        k = k.to(device=y.device, dtype=torch.long)
        if k.dim() == 3:
            k = k.unsqueeze(1)
        while k.dim() > 4:
            k = k.squeeze(1)
        out = torch.gather(y, 1, k)
        return self.gain.clamp(min=1e-3) * out

