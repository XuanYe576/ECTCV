"""
Pipeline B for mmmpaperablation: line-energy -> hat_theta -> differentiable orientation routing.
- Kernel = mask only (L1-normalized), no Gaussian prior on the mask.
- Learnable scalar gain on the mask output.
- Learnable routing temperature + learnable rho (effective top-k ratio).
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


def _circular_line_distance(theta: torch.Tensor, theta_centers: torch.Tensor) -> torch.Tensor:
    """
    Circular distance on line-orientation space [0, pi).
    theta: (B,1,H,W), theta_centers: (1,n,1,1)
    returns: (B,n,H,W) in [0, pi/2]
    """
    # Map delta to [-pi/2, pi/2], then abs for shortest line-orientation distance.
    delta = torch.remainder(theta - theta_centers + 0.5 * math.pi, math.pi) - 0.5 * math.pi
    return torch.abs(delta)


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
    """Pipeline B with differentiable orientation routing."""
    def __init__(
        self,
        n_orientations: int = 4,
        Q: int = 24,
        N: int = 8,
        T_theta: float = 1.0,
        tol: float = 0.5,
        route_temp_init: float = 0.08,
        rho_init: float = 0.5,
        rho_min: float = 0.05,
    ):
        super().__init__()
        self.n = max(1, n_orientations)
        self.line_energy = LineEnergyHead7x7(Q=Q, N=N, T_theta=T_theta, sigma_r=1.0)
        W = build_bank_7x7(self.n, tol=tol)
        self.register_buffer("_W", W)
        self.gain = nn.Parameter(torch.ones(1))
        self.rho_min = float(max(1e-3, min(rho_min, 0.99)))
        self.log_route_temp = nn.Parameter(torch.tensor(math.log(max(route_temp_init, 1e-3))))

        rho_init = float(max(self.rho_min + 1e-4, min(rho_init, 0.999)))
        rho_scaled = (rho_init - self.rho_min) / (1.0 - self.rho_min)
        rho_scaled = max(1e-4, min(rho_scaled, 1 - 1e-4))
        self.rho_logit = nn.Parameter(torch.tensor(math.log(rho_scaled / (1.0 - rho_scaled))))

        # Orientation bin centers in [0, pi): (k + 0.5) * pi/n
        Delta = math.pi / float(self.n)
        theta0 = 0.5 * Delta
        centers = theta0 + torch.arange(self.n, dtype=torch.float32) * Delta
        self.register_buffer("_theta_centers", centers.view(1, self.n, 1, 1))

    def _routing_weights(self, hat_theta: torch.Tensor) -> torch.Tensor:
        """
        hat_theta: (B,H,W) in [0, 2pi)
        returns soft routing weights: (B,n,H,W)
        """
        theta = torch.remainder(hat_theta, math.pi).unsqueeze(1)
        theta_centers = self._theta_centers.to(device=theta.device, dtype=theta.dtype)
        dist = _circular_line_distance(theta, theta_centers)

        route_temp = torch.exp(self.log_route_temp).clamp(min=1e-3, max=2.0)
        base = F.softmax(-dist / route_temp, dim=1)

        # rho in (rho_min, 1): controls sparsity (smaller rho -> sharper, top-k-like routing)
        rho = torch.sigmoid(self.rho_logit) * (1.0 - self.rho_min) + self.rho_min
        gamma = 1.0 / (rho + 1e-8)
        sharp = torch.pow(base + 1e-12, gamma)
        weights = sharp / (sharp.sum(dim=1, keepdim=True) + 1e-8)
        return weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 2, H, W). Returns (B, 1, H, W)."""
        hat_theta = self.line_energy(x)
        W = self._W
        w = W.expand(self.n, 2, 7, 7).clone()
        y = F.conv2d(x, w, padding=3)
        weights = self._routing_weights(hat_theta).to(dtype=y.dtype, device=y.device)
        out = (y * weights).sum(dim=1, keepdim=True)
        return self.gain.clamp(min=1e-3) * out

