import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage import measure


def SoftIoULoss(pred, target):
    pred = torch.sigmoid(pred)

    smooth = 1.0

    intersection = pred * target
    intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
    pred_sum = torch.sum(pred, dim=(1, 2, 3))
    target_sum = torch.sum(target, dim=(1, 2, 3))

    loss = (intersection_sum + smooth) / (
        pred_sum + target_sum - intersection_sum + smooth
    )

    loss = 1.0 - loss.mean()
    return loss


def Dice(pred, target, warm_epoch=1, epoch=1, layer=0):
    pred = torch.sigmoid(pred)

    smooth = 1.0

    intersection = pred * target
    intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
    pred_sum = torch.sum(pred, dim=(1, 2, 3))
    target_sum = torch.sum(target, dim=(1, 2, 3))

    loss = (2.0 * intersection_sum + smooth) / (
        pred_sum + target_sum + intersection_sum + smooth
    )

    loss = 1.0 - loss.mean()
    return loss


class SoftIoULossModule(nn.Module):
    """
    标准 Soft IoU Loss 的 Module 封装，用于训练。
    loss = 1 - (intersection + smooth) / (pred_sum + target_sum - intersection + smooth)
    """
    def __init__(self):
        super(SoftIoULossModule, self).__init__()

    def forward(self, pred_log, target, warm_epoch, epoch, with_shape=True):
        return SoftIoULoss(pred_log, target)


def LLoss(pred, target):
    """
    修正后的 Location Loss.

    与第一份代码保持同一设计思想：
      - 用中心角度差 + 中心长度比 来约束位置/形状
    但修正原实现中的关键问题：
      - 质心应使用质量归一化形式 sum(x*m)/sum(m)
      - 不能使用 mean(x*m)，否则位置项会与总面积/总响应强度耦合

    Args:
        pred   : [B, C, H, W] after sigmoid
        target : [B, C, H, W]

    Returns:
        scalar tensor
    """
    device = pred.device
    dtype = pred.dtype

    batch_size = pred.shape[0]
    h = pred.shape[2]
    w = pred.shape[3]
    smooth = 1e-8

    x_index = (
        torch.arange(0, w, 1, dtype=dtype, device=device)
        .view(1, 1, w)
        .repeat(1, h, 1)
    ) / w
    y_index = (
        torch.arange(0, h, 1, dtype=dtype, device=device)
        .view(1, h, 1)
        .repeat(1, 1, w)
    ) / h

    loss = pred.new_tensor(0.0)

    for i in range(batch_size):
        pi = pred[i]
        ti = target[i]

        pred_mass = pi.sum() + smooth
        target_mass = ti.sum() + smooth

        pred_centerx = (x_index * pi).sum() / pred_mass
        pred_centery = (y_index * pi).sum() / pred_mass

        target_centerx = (x_index * ti).sum() / target_mass
        target_centery = (y_index * ti).sum() / target_mass

        angle_loss = (4.0 / (torch.pi ** 2)) * torch.square(
            torch.arctan(pred_centery / (pred_centerx + smooth))
            - torch.arctan(target_centery / (target_centerx + smooth))
        )

        pred_length = torch.sqrt(
            pred_centerx * pred_centerx + pred_centery * pred_centery + smooth
        )
        target_length = torch.sqrt(
            target_centerx * target_centerx + target_centery * target_centery + smooth
        )

        length_loss = torch.min(pred_length, target_length) / (
            torch.max(pred_length, target_length) + smooth
        )

        loss = loss + (1.0 - length_loss + angle_loss) / batch_size

    return loss


class LLossOnlyLoss(nn.Module):
    """
    仅用 Location Loss (LLoss) 训练，无 IoU 项。
    用于对比：看纯 location/shape loss 会带来多少虚警。
    """
    def __init__(self):
        super(LLossOnlyLoss, self).__init__()

    def forward(self, pred_log, target, warm_epoch, epoch, with_shape=True):
        pred = torch.sigmoid(pred_log)
        return LLoss(pred, target)


class SLSIoULoss(nn.Module):
    """
    保留第一份定义。
    实质上与 L2 形式同型：
    alpha = (min + (Δ/2)^2) / (max + (Δ/2)^2)
    """
    def __init__(self):
        super(SLSIoULoss, self).__init__()

    def forward(self, pred_log, target, warm_epoch, epoch, with_shape=True):
        pred = torch.sigmoid(pred_log)
        smooth = 0.0

        intersection = pred * target
        intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
        pred_sum = torch.sum(pred, dim=(1, 2, 3))
        target_sum = torch.sum(target, dim=(1, 2, 3))

        dis = torch.pow((pred_sum - target_sum) / 2.0, 2)

        alpha = (torch.min(pred_sum, target_sum) + dis + smooth) / (
            torch.max(pred_sum, target_sum) + dis + smooth
        )

        loss = (intersection_sum + smooth) / (
            pred_sum + target_sum - intersection_sum + smooth
        )
        lloss = LLoss(pred, target)

        if epoch > warm_epoch:
            siou_loss = alpha * loss
            if with_shape:
                loss = 1.0 - siou_loss.mean() + lloss
            else:
                loss = 1.0 - siou_loss.mean()
        else:
            loss = 1.0 - loss.mean()
        return loss


class L1IoULoss(nn.Module):
    """
    L1-based IoU Loss
    alpha = (min(|A_p|, |A_t|) + |A_p - A_t|/2) / (max(|A_p|, |A_t|) + |A_p - A_t|/2)
    """
    def __init__(self):
        super(L1IoULoss, self).__init__()

    def forward(self, pred_log, target, warm_epoch, epoch, with_shape=True):
        pred = torch.sigmoid(pred_log)
        smooth = 0.0

        intersection = pred * target
        intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
        pred_sum = torch.sum(pred, dim=(1, 2, 3))
        target_sum = torch.sum(target, dim=(1, 2, 3))

        abs_diff = torch.abs(pred_sum - target_sum)
        m = torch.min(pred_sum, target_sum)
        M = torch.max(pred_sum, target_sum)

        alpha = (m + abs_diff / 2.0 + smooth) / (M + abs_diff / 2.0 + smooth)

        loss = (intersection_sum + smooth) / (
            pred_sum + target_sum - intersection_sum + smooth
        )
        lloss = LLoss(pred, target)

        if epoch > warm_epoch:
            l1_iou_loss = alpha * loss
            if with_shape:
                loss = 1.0 - l1_iou_loss.mean() + lloss
            else:
                loss = 1.0 - l1_iou_loss.mean()
        else:
            loss = 1.0 - loss.mean()
        return loss


class L2IoULoss(nn.Module):
    """
    L2-based IoU Loss
    alpha = (min(|A_p|, |A_t|) + (Δ/2)^2) / (max(|A_p|, |A_t|) + (Δ/2)^2)
    """
    def __init__(self):
        super(L2IoULoss, self).__init__()

    def forward(self, pred_log, target, warm_epoch, epoch, with_shape=True):
        pred = torch.sigmoid(pred_log)
        smooth = 0.0

        intersection = pred * target
        intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
        pred_sum = torch.sum(pred, dim=(1, 2, 3))
        target_sum = torch.sum(target, dim=(1, 2, 3))

        m = torch.min(pred_sum, target_sum)
        M = torch.max(pred_sum, target_sum)
        dis = torch.pow((pred_sum - target_sum) / 2.0, 2)

        alpha = (m + dis + smooth) / (M + dis + smooth)

        loss = (intersection_sum + smooth) / (
            pred_sum + target_sum - intersection_sum + smooth
        )
        lloss = LLoss(pred, target)

        if epoch > warm_epoch:
            l2_iou_loss = alpha * loss
            if with_shape:
                loss = 1.0 - l2_iou_loss.mean() + lloss
            else:
                loss = 1.0 - l2_iou_loss.mean()
        else:
            loss = 1.0 - loss.mean()
        return loss


class L3IoULoss(nn.Module):
    """
    L3-based IoU Loss (Mobius-like)
    alpha = (3*min(|A_p|, |A_t|) + max(|A_p|, |A_t|)) / (3*max(|A_p|, |A_t|) + min(|A_p|, |A_t|))
    """
    def __init__(self):
        super(L3IoULoss, self).__init__()

    def forward(self, pred_log, target, warm_epoch, epoch, with_shape=True):
        pred = torch.sigmoid(pred_log)
        smooth = 0.0

        intersection = pred * target
        intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
        pred_sum = torch.sum(pred, dim=(1, 2, 3))
        target_sum = torch.sum(target, dim=(1, 2, 3))

        m = torch.min(pred_sum, target_sum)
        M = torch.max(pred_sum, target_sum)

        alpha = (3.0 * m + M + smooth) / (3.0 * M + m + smooth)
        alpha = torch.clamp(alpha, 1.0 / 3.0, 1.0)

        loss = (intersection_sum + smooth) / (
            pred_sum + target_sum - intersection_sum + smooth
        )
        lloss = LLoss(pred, target)

        if epoch > warm_epoch:
            l3_iou_loss = alpha * loss
            if with_shape:
                loss = 1.0 - l3_iou_loss.mean() + lloss
            else:
                loss = 1.0 - l3_iou_loss.mean()
        else:
            loss = 1.0 - loss.mean()
        return loss


class L3WithDlossIoULoss(nn.Module):
    """
    L3-based IoU Loss with D_loss (from IR-SOIoU)
    Combines L3 alpha with IR-SOIoU-style center distance penalty.
    """
    def __init__(self, A_ref=0.01, beta=0.5, alpha=0.5, eps=1e-8):
        super(L3WithDlossIoULoss, self).__init__()
        self.A_ref = A_ref
        self.beta = beta
        self.alpha = alpha
        self.eps = eps

    def forward(self, pred_log, target, warm_epoch, epoch, with_shape=True):
        pred = torch.sigmoid(pred_log)
        smooth = 0.0

        intersection = pred * target
        intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
        pred_sum = torch.sum(pred, dim=(1, 2, 3))
        target_sum = torch.sum(target, dim=(1, 2, 3))

        m = torch.min(pred_sum, target_sum)
        M = torch.max(pred_sum, target_sum)

        alpha_l3 = (3.0 * m + M + smooth) / (3.0 * M + m + smooth)
        alpha_l3 = torch.clamp(alpha_l3, 1.0 / 3.0, 1.0)

        loss = (intersection_sum + smooth) / (
            pred_sum + target_sum - intersection_sum + smooth
        )

        h = pred.shape[2]
        w = pred.shape[3]
        A_img = h * w

        x_index = (
            torch.arange(0, w, 1, dtype=pred.dtype, device=pred.device)
            .view(1, 1, w)
            .repeat(1, h, 1)
        ) / w
        y_index = (
            torch.arange(0, h, 1, dtype=pred.dtype, device=pred.device)
            .view(1, h, 1)
            .repeat(1, 1, w)
        ) / h

        x_index = x_index.unsqueeze(0)
        y_index = y_index.unsqueeze(0)

        pred_center_x = torch.sum(x_index * pred, dim=(1, 2, 3)) / (pred_sum + self.eps)
        pred_center_y = torch.sum(y_index * pred, dim=(1, 2, 3)) / (pred_sum + self.eps)
        target_center_x = torch.sum(x_index * target, dim=(1, 2, 3)) / (target_sum + self.eps)
        target_center_y = torch.sum(y_index * target, dim=(1, 2, 3)) / (target_sum + self.eps)

        d_squared = (pred_center_x - target_center_x) ** 2 + (
            pred_center_y - target_center_y
        ) ** 2

        pred_bbox = self._get_bbox(
            pred, x_index.squeeze(0).squeeze(0), y_index.squeeze(0).squeeze(0), w, h
        )
        target_bbox = self._get_bbox(
            target, x_index.squeeze(0).squeeze(0), y_index.squeeze(0).squeeze(0), w, h
        )

        min_x = torch.min(pred_bbox[:, 0], target_bbox[:, 0])
        min_y = torch.min(pred_bbox[:, 1], target_bbox[:, 1])
        max_x = torch.max(pred_bbox[:, 2], target_bbox[:, 2])
        max_y = torch.max(pred_bbox[:, 3], target_bbox[:, 3])
        c_squared = (max_x - min_x) ** 2 + (max_y - min_y) ** 2 + self.eps

        A = target_sum / A_img
        gamma = torch.pow(self.A_ref / (A + self.eps), self.beta)
        D_loss = (d_squared / c_squared) * gamma

        if epoch > warm_epoch:
            l3_iou_loss = alpha_l3 * loss
            if with_shape:
                loss = 1.0 - l3_iou_loss.mean() + self.alpha * D_loss.mean()
            else:
                loss = 1.0 - l3_iou_loss.mean()
        else:
            loss = 1.0 - loss.mean()
        return loss

    def _get_bbox(self, mask, x_index, y_index, w, h):
        batch_size = mask.shape[0]
        bboxes = torch.zeros(batch_size, 4, dtype=mask.dtype, device=mask.device)
        threshold = 0.01

        for i in range(batch_size):
            single_mask = mask[i, 0] if mask.shape[1] > 0 else mask[i]
            active_mask = single_mask > threshold

            if active_mask.any():
                active_x = x_index[active_mask]
                active_y = y_index[active_mask]
                bboxes[i, 0] = active_x.min()
                bboxes[i, 1] = active_y.min()
                bboxes[i, 2] = active_x.max()
                bboxes[i, 3] = active_y.max()
            else:
                bboxes[i, 0] = 0.0
                bboxes[i, 1] = 0.0
                bboxes[i, 2] = 1.0
                bboxes[i, 3] = 1.0

        return bboxes


class L4IoULoss(nn.Module):
    """
    L4-based IoU Loss (strictly keep the FIRST version definition)

    IMPORTANT:
    This keeps your first-version quadratic penalty, not the second-version linear one.

    alpha = min(|A_p|, |A_t|) / (max(|A_p|, |A_t|) + 2*((A_p - A_t)/2)^2)
          = min / (max + (Δ^2)/2)
    """
    def __init__(self):
        super(L4IoULoss, self).__init__()

    def forward(self, pred_log, target, warm_epoch, epoch, with_shape=True):
        pred = torch.sigmoid(pred_log)
        smooth = 0.0

        intersection = pred * target
        intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
        pred_sum = torch.sum(pred, dim=(1, 2, 3))
        target_sum = torch.sum(target, dim=(1, 2, 3))

        var = torch.pow((pred_sum - target_sum) / 2.0, 2)
        m = torch.min(pred_sum, target_sum)
        M = torch.max(pred_sum, target_sum)

        alpha = (m + smooth) / (M + 2.0 * var + smooth)

        loss = (intersection_sum + smooth) / (
            pred_sum + target_sum - intersection_sum + smooth
        )
        lloss = LLoss(pred, target)

        if epoch > warm_epoch:
            l4_iou_loss = alpha * loss
            if with_shape:
                loss = 1.0 - l4_iou_loss.mean() + lloss
            else:
                loss = 1.0 - l4_iou_loss.mean()
        else:
            loss = 1.0 - loss.mean()
        return loss


class IRSOIoULoss(nn.Module):
    """
    Infrared Small-Object IoU Loss (IR-SOIoU)
    Region Energy-Based Dynamic Loss

    Formula:
        L_IR-SOIoU = 1 - IoU^γ + α * D_loss
    """
    def __init__(self, A_ref=0.01, beta=0.5, alpha=0.5, eps=1e-8):
        super(IRSOIoULoss, self).__init__()
        self.A_ref = A_ref
        self.beta = beta
        self.alpha = alpha
        self.eps = eps

    def forward(self, pred_log, target, warm_epoch, epoch, with_shape=True):
        pred = torch.sigmoid(pred_log)
        smooth = 0.0

        h = pred.shape[2]
        w = pred.shape[3]
        A_img = h * w

        intersection = pred * target
        intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
        pred_sum = torch.sum(pred, dim=(1, 2, 3))
        target_sum = torch.sum(target, dim=(1, 2, 3))

        iou = (intersection_sum + smooth) / (
            pred_sum + target_sum - intersection_sum + smooth
        )

        x_index = (
            torch.arange(0, w, 1, dtype=pred.dtype, device=pred.device)
            .view(1, 1, w)
            .repeat(1, h, 1)
        ) / w
        y_index = (
            torch.arange(0, h, 1, dtype=pred.dtype, device=pred.device)
            .view(1, h, 1)
            .repeat(1, 1, w)
        ) / h

        x_index = x_index.unsqueeze(0)
        y_index = y_index.unsqueeze(0)

        pred_center_x = torch.sum(x_index * pred, dim=(1, 2, 3)) / (pred_sum + self.eps)
        pred_center_y = torch.sum(y_index * pred, dim=(1, 2, 3)) / (pred_sum + self.eps)
        target_center_x = torch.sum(x_index * target, dim=(1, 2, 3)) / (target_sum + self.eps)
        target_center_y = torch.sum(y_index * target, dim=(1, 2, 3)) / (target_sum + self.eps)

        d_squared = (pred_center_x - target_center_x) ** 2 + (
            pred_center_y - target_center_y
        ) ** 2

        pred_bbox = self._get_bbox(
            pred, x_index.squeeze(0).squeeze(0), y_index.squeeze(0).squeeze(0), w, h
        )
        target_bbox = self._get_bbox(
            target, x_index.squeeze(0).squeeze(0), y_index.squeeze(0).squeeze(0), w, h
        )

        min_x = torch.min(pred_bbox[:, 0], target_bbox[:, 0])
        min_y = torch.min(pred_bbox[:, 1], target_bbox[:, 1])
        max_x = torch.max(pred_bbox[:, 2], target_bbox[:, 2])
        max_y = torch.max(pred_bbox[:, 3], target_bbox[:, 3])

        c_squared = (max_x - min_x) ** 2 + (max_y - min_y) ** 2 + self.eps

        A = target_sum / A_img
        gamma = torch.pow(self.A_ref / (A + self.eps), self.beta)
        D_loss = (d_squared / c_squared) * gamma

        iou_gamma = torch.pow(iou + self.eps, gamma)

        if epoch > warm_epoch:
            if with_shape:
                lloss = LLoss(pred, target)
                ir_soiou_loss = 1.0 - iou_gamma
                loss = ir_soiou_loss.mean() + self.alpha * lloss
            else:
                ir_soiou_loss = 1.0 - iou_gamma + self.alpha * D_loss
                loss = ir_soiou_loss.mean()
        else:
            loss = 1.0 - iou.mean()

        return loss

    def _get_bbox(self, mask, x_index, y_index, w, h):
        batch_size = mask.shape[0]
        bboxes = torch.zeros(batch_size, 4, dtype=mask.dtype, device=mask.device)
        threshold = 0.01

        for i in range(batch_size):
            single_mask = mask[i, 0] if mask.shape[1] > 0 else mask[i]
            active_mask = single_mask > threshold

            if active_mask.any():
                active_x = x_index[active_mask]
                active_y = y_index[active_mask]

                bboxes[i, 0] = active_x.min()
                bboxes[i, 1] = active_y.min()
                bboxes[i, 2] = active_x.max()
                bboxes[i, 3] = active_y.max()
            else:
                bboxes[i, 0] = 0.0
                bboxes[i, 1] = 0.0
                bboxes[i, 2] = 1.0
                bboxes[i, 3] = 1.0

        return bboxes


class FocalIoULoss(nn.Module):
    """
    保留接口兼容；你主实验不做 focal，就不要调用这个类。

    Focal IoU Loss = Ln-IoU(α加权) + LLoss + focal_w * FocalBCE
    variant in {'L1', 'L2', 'L3', 'L4'}
    """
    _VARIANTS = ("L1", "L2", "L3", "L4")

    def __init__(self, variant="L1", gamma=2.0, alpha=0.75, focal_w=1.0):
        super().__init__()
        variant = variant.upper()
        if variant not in self._VARIANTS:
            raise ValueError(
                f"FocalIoULoss variant must be one of {self._VARIANTS}, got {variant!r}"
            )
        self.variant = variant
        self.gamma = gamma
        self.alpha = alpha
        self.focal_w = focal_w

    def _alpha(self, ps, ts):
        m = torch.min(ps, ts)
        M = torch.max(ps, ts)

        if self.variant == "L1":
            d = torch.abs(ps - ts)
            return (m + d / 2.0) / (M + d / 2.0 + 1e-8)
        elif self.variant == "L2":
            d2 = ((ps - ts) / 2.0) ** 2
            return (m + d2) / (M + d2 + 1e-8)
        elif self.variant == "L3":
            a = (3.0 * m + M) / (3.0 * M + m + 1e-8)
            return torch.clamp(a, 1.0 / 3.0, 1.0)
        else:
            var = ((ps - ts) / 2.0) ** 2
            return m / (M + 2.0 * var + 1e-8)

    def _focal_bce(self, pred_log, target):
        bce = F.binary_cross_entropy_with_logits(pred_log, target, reduction="none")
        p = torch.sigmoid(pred_log)
        p_t = p * target + (1.0 - p) * (1.0 - target)
        a_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        w = a_t * (1.0 - p_t) ** self.gamma
        return (w * bce).mean()

    def forward(self, pred_log, target, warm_epoch, epoch, with_shape=True):
        pred = torch.sigmoid(pred_log)
        smooth = 0.0

        inter = (pred * target).sum(dim=(1, 2, 3))
        ps = pred.sum(dim=(1, 2, 3))
        ts = target.sum(dim=(1, 2, 3))
        iou_base = (inter + smooth) / (ps + ts - inter + smooth)
        alpha_w = self._alpha(ps, ts)
        lloss = LLoss(pred, target)
        focal = self._focal_bce(pred_log, target)

        if epoch > warm_epoch:
            iou_w = alpha_w * iou_base
            if with_shape:
                loss = 1.0 - iou_w.mean() + lloss + self.focal_w * focal
            else:
                loss = 1.0 - iou_w.mean() + self.focal_w * focal
        else:
            loss = 1.0 - iou_base.mean() + self.focal_w * focal

        return loss


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count