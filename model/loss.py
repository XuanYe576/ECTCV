import torch.nn as nn
import numpy as np
import  torch
import torch.nn.functional as F
from skimage import measure


def SoftIoULoss( pred, target):
        pred = torch.sigmoid(pred)
  
        smooth = 1

        intersection = pred * target
        intersection_sum = torch.sum(intersection, dim=(1,2,3))
        pred_sum = torch.sum(pred, dim=(1,2,3))
        target_sum = torch.sum(target, dim=(1,2,3))
        
        loss = (intersection_sum + smooth) / \
                    (pred_sum + target_sum - intersection_sum + smooth)
    
        loss = 1 - loss.mean()

        return loss

def Dice( pred, target,warm_epoch=1, epoch=1, layer=0):
        pred = torch.sigmoid(pred)
  
        smooth = 1

        intersection = pred * target
        intersection_sum = torch.sum(intersection, dim=(1,2,3))
        pred_sum = torch.sum(pred, dim=(1,2,3))
        target_sum = torch.sum(target, dim=(1,2,3))

        loss = (2*intersection_sum + smooth) / \
            (pred_sum + target_sum + intersection_sum + smooth)

        loss = 1 - loss.mean()

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


class SLSIoULoss(nn.Module):
    """L2 (SLS): alpha = (min + (Δ/2)^2) / (max + (Δ/2)^2), Δ = A_p - A_t."""
    def __init__(self):
        super(SLSIoULoss, self).__init__()

    def forward(self, pred_log, target,warm_epoch, epoch, with_shape=True):
        pred = torch.sigmoid(pred_log)
        smooth = 0.0
        intersection = pred * target
        intersection_sum = torch.sum(intersection, dim=(1,2,3))
        pred_sum = torch.sum(pred, dim=(1,2,3))
        target_sum = torch.sum(target, dim=(1,2,3))
        dis = torch.pow((pred_sum - target_sum) / 2.0, 2)
        
        
        alpha = (torch.min(pred_sum, target_sum) + dis + smooth) / (torch.max(pred_sum, target_sum) + dis + smooth) 
        
        loss = (intersection_sum + smooth) / \
                (pred_sum + target_sum - intersection_sum  + smooth)       
        lloss = LLoss(pred, target)

        if epoch>warm_epoch:       
            siou_loss = alpha * loss
            if with_shape:
                loss = 1 - siou_loss.mean() + lloss
            else:
                loss = 1 -siou_loss.mean()
        else:
            loss = 1 - loss.mean()
        return loss


class L2IoULoss(nn.Module):
    """
    L2-based IoU Loss（与 L1/L3/L4 并列，可公平比较 Lx-ONLY）.
    alpha = (min(|A_p|, |A_t|) + (Δ/2)^2) / (max(|A_p|, |A_t|) + (Δ/2)^2), Δ = |A_p - A_t|.
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
        # L2 alpha: (min + (Δ/2)^2) / (max + (Δ/2)^2)
        dis = torch.pow((pred_sum - target_sum) / 2.0, 2)
        alpha = (m + dis + smooth) / (M + dis + smooth)

        loss = (intersection_sum + smooth) / \
               (pred_sum + target_sum - intersection_sum + smooth)
        lloss = LLoss(pred, target)

        if epoch > warm_epoch:
            l2_iou_loss = alpha * loss
            if with_shape:
                loss = 1 - l2_iou_loss.mean() + lloss
            else:
                loss = 1 - l2_iou_loss.mean()
        else:
            loss = 1 - loss.mean()
        return loss


def LLoss(pred, target):
        """Location/shape loss. Centroids = sum(x*mass)/sum(mass), not mean over pixels."""
        loss = torch.tensor(0.0, requires_grad=True).to(pred)
        patch_size = pred.shape[0]
        h, w = pred.shape[2], pred.shape[3]
        x_index = torch.arange(0, w, 1, dtype=pred.dtype, device=pred.device).view(1, 1, w).repeat(1, h, 1)
        y_index = torch.arange(0, h, 1, dtype=pred.dtype, device=pred.device).view(1, h, 1).repeat(1, 1, w)
        smooth = 1e-8
        for i in range(patch_size):
            pi, ti = pred[i], target[i]
            psum = pi.sum() + smooth
            tsum = ti.sum() + smooth
            pred_centerx = (x_index * pi).sum() / psum
            pred_centery = (y_index * pi).sum() / psum
            target_centerx = (x_index * ti).sum() / tsum
            target_centery = (y_index * ti).sum() / tsum
            angle_loss = (4 / (torch.pi ** 2)) * (
                torch.square(torch.arctan(pred_centery / (pred_centerx + smooth))
                            - torch.arctan(target_centery / (target_centerx + smooth)))
            pred_length = torch.sqrt(pred_centerx * pred_centerx + pred_centery * pred_centery + smooth)
            target_length = torch.sqrt(target_centerx * target_centerx + target_centery * target_centery + smooth)
            length_loss = (torch.min(pred_length, target_length)) / (torch.max(pred_length, target_length) + smooth)
            loss = loss + (1 - length_loss + angle_loss) / patch_size
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


class L1IoULoss(nn.Module):
    """
    L1-based IoU Loss
    L1 = (min(|A_p|, |A_t|) + |A_p - A_t|/2) / (max(|A_p|, |A_t|) + |A_p - A_t|/2)
    """
    def __init__(self):
        super(L1IoULoss, self).__init__()

    def forward(self, pred_log, target, warm_epoch, epoch, with_shape=True):
        pred = torch.sigmoid(pred_log)
        smooth = 0.0

        intersection = pred * target
        intersection_sum = torch.sum(intersection, dim=(1,2,3))
        pred_sum = torch.sum(pred, dim=(1,2,3))
        target_sum = torch.sum(target, dim=(1,2,3))
        
        # 计算绝对差
        abs_diff = torch.abs(pred_sum - target_sum)
        m = torch.min(pred_sum, target_sum)
        M = torch.max(pred_sum, target_sum)
        
        # L1 alpha: (min + |P-GT|/2) / (max + |P-GT|/2)
        alpha = (m + abs_diff / 2.0 + smooth) / (M + abs_diff / 2.0 + smooth)
        
        loss = (intersection_sum + smooth) / \
                (pred_sum + target_sum - intersection_sum + smooth)
        lloss = LLoss(pred, target)

        if epoch > warm_epoch:
            l1_iou_loss = alpha * loss
            if with_shape:
                loss = 1 - l1_iou_loss.mean() + lloss
            else:
                loss = 1 - l1_iou_loss.mean()
        else:
            loss = 1 - loss.mean()
        return loss


class L3IoULoss(nn.Module):
    """
    L3-based IoU Loss (Mobius)
    L3 = (3*min(|A_p|, |A_t|) + max(|A_p|, |A_t|)) / (3*max(|A_p|, |A_t|) + min(|A_p|, |A_t|))
    """
    def __init__(self):
        super(L3IoULoss, self).__init__()

    def forward(self, pred_log, target, warm_epoch, epoch, with_shape=True):
        pred = torch.sigmoid(pred_log)
        smooth = 0.0

        intersection = pred * target
        intersection_sum = torch.sum(intersection, dim=(1,2,3))
        pred_sum = torch.sum(pred, dim=(1,2,3))
        target_sum = torch.sum(target, dim=(1,2,3))
        
        m = torch.min(pred_sum, target_sum)
        M = torch.max(pred_sum, target_sum)
        
        # L3 alpha: (3*min + max) / (3*max + min)
        alpha = (3.0 * m + M + smooth) / (3.0 * M + m + smooth)
        # 限制alpha在[1/3, 1]范围内
        alpha = torch.clamp(alpha, 1.0/3.0, 1.0)
        
        loss = (intersection_sum + smooth) / \
                (pred_sum + target_sum - intersection_sum + smooth)
        lloss = LLoss(pred, target)

        if epoch > warm_epoch:
            l3_iou_loss = alpha * loss
            if with_shape:
                loss = 1 - l3_iou_loss.mean() + lloss
            else:
                loss = 1 - l3_iou_loss.mean()
        else:
            loss = 1 - loss.mean()
        return loss


class L3WithDlossIoULoss(nn.Module):
    """
    L3-based IoU Loss with D_loss (from IR-SOIoU)
    Combines L3 alpha calculation with IR-SOIoU's D_loss for location penalty
    """
    def __init__(self, A_ref=0.01, beta=0.5, alpha=0.5, eps=1e-8):
        """
        Args:
            A_ref: Reference area for D_loss calculation (default: 0.01)
            beta: Exponent for area adaptive index, β ∈ (0, 1) (default: 0.5)
            alpha: Balancing factor for D_loss (default: 0.5)
            eps: Small constant for numerical stability (default: 1e-8)
        """
        super(L3WithDlossIoULoss, self).__init__()
        self.A_ref = A_ref
        self.beta = beta
        self.alpha = alpha
        self.eps = eps

    def forward(self, pred_log, target, warm_epoch, epoch, with_shape=True):
        pred = torch.sigmoid(pred_log)
        smooth = 0.0

        intersection = pred * target
        intersection_sum = torch.sum(intersection, dim=(1,2,3))
        pred_sum = torch.sum(pred, dim=(1,2,3))
        target_sum = torch.sum(target, dim=(1,2,3))
        
        m = torch.min(pred_sum, target_sum)
        M = torch.max(pred_sum, target_sum)
        
        # L3 alpha: (3*min + max) / (3*max + min)
        alpha_l3 = (3.0 * m + M + smooth) / (3.0 * M + m + smooth)
        # 限制alpha在[1/3, 1]范围内
        alpha_l3 = torch.clamp(alpha_l3, 1.0/3.0, 1.0)
        
        loss = (intersection_sum + smooth) / \
                (pred_sum + target_sum - intersection_sum + smooth)
        
        # Calculate D_loss (from IR-SOIoU)
        batch_size = pred.shape[0]
        h = pred.shape[2]
        w = pred.shape[3]
        A_img = h * w
        
        # Calculate centroids for center distance
        x_index = torch.arange(0, w, 1, dtype=torch.float32).view(1, 1, w).repeat(1, h, 1).to(pred.device) / w
        y_index = torch.arange(0, h, 1, dtype=torch.float32).view(1, h, 1).repeat(1, 1, w).to(pred.device) / h
        x_index = x_index.unsqueeze(0)
        y_index = y_index.unsqueeze(0)
        
        pred_center_x = torch.sum(x_index * pred, dim=(1,2,3)) / (pred_sum + self.eps)
        pred_center_y = torch.sum(y_index * pred, dim=(1,2,3)) / (pred_sum + self.eps)
        target_center_x = torch.sum(x_index * target, dim=(1,2,3)) / (target_sum + self.eps)
        target_center_y = torch.sum(y_index * target, dim=(1,2,3)) / (target_sum + self.eps)
        
        d_squared = (pred_center_x - target_center_x) ** 2 + (pred_center_y - target_center_y) ** 2
        
        # Get bounding boxes
        pred_bbox = self._get_bbox(pred, x_index.squeeze(0).squeeze(0), y_index.squeeze(0).squeeze(0), w, h)
        target_bbox = self._get_bbox(target, x_index.squeeze(0).squeeze(0), y_index.squeeze(0).squeeze(0), w, h)
        
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
                # Use D_loss
                loss = 1 - l3_iou_loss.mean() + self.alpha * D_loss.mean()
            else:
                # Without location loss
                loss = 1 - l3_iou_loss.mean()
        else:
            loss = 1 - loss.mean()
        return loss
    
    def _get_bbox(self, mask, x_index, y_index, w, h):
        """Get bounding box coordinates [min_x, min_y, max_x, max_y] for each sample in batch."""
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
    L4-based IoU Loss (Man Fung)
    L4 = min(|A_p|, |A_t|) / (max(|A_p|, |A_t|) + 2*var(|A_p|, |A_t|))
    """
    def __init__(self):
        super(L4IoULoss, self).__init__()

    def forward(self, pred_log, target, warm_epoch, epoch, with_shape=True):
        pred = torch.sigmoid(pred_log)
        smooth = 0.0

        intersection = pred * target
        intersection_sum = torch.sum(intersection, dim=(1,2,3))
        pred_sum = torch.sum(pred, dim=(1,2,3))
        target_sum = torch.sum(target, dim=(1,2,3))
        
        # variance 全局 abs: var = |Δ|/2
        var = torch.abs(pred_sum - target_sum) / 2.0
        m = torch.min(pred_sum, target_sum)
        M = torch.max(pred_sum, target_sum)
        # L4 alpha: min / (max + 2*var) = min / (max + |Δ|)
        alpha = (m + smooth) / (M + 2.0 * var + smooth)
        
        loss = (intersection_sum + smooth) / \
                (pred_sum + target_sum - intersection_sum + smooth)
        lloss = LLoss(pred, target)

        if epoch > warm_epoch:
            l4_iou_loss = alpha * loss
            if with_shape:
                loss = 1 - l4_iou_loss.mean() + lloss
            else:
                loss = 1 - l4_iou_loss.mean()
        else:
            loss = 1 - loss.mean()
        return loss


class IRSOIoULoss(nn.Module):
    """
    Infrared Small-Object IoU Loss (IR-SOIoU)
    Region Energy-Based Dynamic Loss
    
    Formula: L_IR-SOIoU = 1 - IoU^γ + α * D_loss
    
    Components:
    - A = wh / A_img (Region Coverage Ratio, Eq. 8)
    - γ = (A_ref / (A + ε))^β (Area Adaptive Index, Eq. 9)
    - D_loss = (d^2 / c^2) * γ (Center Distance Loss, Eq. 10)
    """
    def __init__(self, A_ref=0.01, beta=0.5, alpha=0.5, eps=1e-8):
        """
        Args:
            A_ref: Reference area (default: 0.01)
            beta: Exponent for area adaptive index, β ∈ (0, 1) (default: 0.5)
            alpha: Balancing factor for D_loss (default: 0.5)
            eps: Small constant for numerical stability (default: 1e-8)
        """
        super(IRSOIoULoss, self).__init__()
        self.A_ref = A_ref
        self.beta = beta
        self.alpha = alpha
        self.eps = eps

    def forward(self, pred_log, target, warm_epoch, epoch, with_shape=True):
        pred = torch.sigmoid(pred_log)
        smooth = 0.0
        
        batch_size = pred.shape[0]
        h = pred.shape[2]
        w = pred.shape[3]
        A_img = h * w  # Total image area
        
        # Calculate IoU
        intersection = pred * target
        intersection_sum = torch.sum(intersection, dim=(1,2,3))
        pred_sum = torch.sum(pred, dim=(1,2,3))
        target_sum = torch.sum(target, dim=(1,2,3))
        
        iou = (intersection_sum + smooth) / (pred_sum + target_sum - intersection_sum + smooth)
        
        # Calculate centroids for center distance (similar to LLoss)
        x_index = torch.arange(0, w, 1, dtype=torch.float32).view(1, 1, w).repeat(1, h, 1).to(pred.device) / w
        y_index = torch.arange(0, h, 1, dtype=torch.float32).view(1, h, 1).repeat(1, 1, w).to(pred.device) / h
        
        # Expand to batch dimension
        x_index = x_index.unsqueeze(0)  # [1, 1, h, w]
        y_index = y_index.unsqueeze(0)  # [1, 1, h, w]
        
        # Calculate centroids (weighted average, normalized to [0,1])
        pred_center_x = torch.sum(x_index * pred, dim=(1,2,3)) / (pred_sum + self.eps)
        pred_center_y = torch.sum(y_index * pred, dim=(1,2,3)) / (pred_sum + self.eps)
        target_center_x = torch.sum(x_index * target, dim=(1,2,3)) / (target_sum + self.eps)
        target_center_y = torch.sum(y_index * target, dim=(1,2,3)) / (target_sum + self.eps)
        
        # Calculate center distance d (Euclidean distance in normalized coordinates)
        d_squared = (pred_center_x - target_center_x) ** 2 + (pred_center_y - target_center_y) ** 2
        
        # Calculate minimum enclosing box diagonal c
        # For segmentation, compute bounding boxes and their union
        # Get bounding box coordinates for pred and target
        pred_bbox = self._get_bbox(pred, x_index.squeeze(0).squeeze(0), y_index.squeeze(0).squeeze(0), w, h)
        target_bbox = self._get_bbox(target, x_index.squeeze(0).squeeze(0), y_index.squeeze(0).squeeze(0), w, h)
        
        # Union bounding box (minimum box containing both)
        min_x = torch.min(pred_bbox[:, 0], target_bbox[:, 0])
        min_y = torch.min(pred_bbox[:, 1], target_bbox[:, 1])
        max_x = torch.max(pred_bbox[:, 2], target_bbox[:, 2])
        max_y = torch.max(pred_bbox[:, 3], target_bbox[:, 3])
        
        # Diagonal length c (normalized)
        c_squared = (max_x - min_x) ** 2 + (max_y - min_y) ** 2 + self.eps
        
        # Calculate Region Coverage Ratio A (Eq. 8)
        # A = wh / A_img, where wh is the area of ground truth object
        A = target_sum / A_img
        
        # Calculate Area Adaptive Index γ (Eq. 9)
        # γ = (A_ref / (A + ε))^β
        gamma = torch.pow(self.A_ref / (A + self.eps), self.beta)
        
        # Calculate Center Distance Loss D_loss (Eq. 10)
        # D_loss = (d^2 / c^2) * γ
        D_loss = (d_squared / c_squared) * gamma
        
        # Calculate IR-SOIoU Loss (Eq. 11)
        # L_IR-SOIoU = 1 - IoU^γ + α * location_loss
        iou_gamma = torch.pow(iou + self.eps, gamma)
        
        # Apply warm-up mechanism similar to other losses
        if epoch > warm_epoch:
            if with_shape:
                # Use LLoss instead of D_loss for comparison
                lloss = LLoss(pred, target)
                # LLoss is a scalar, so add it after taking mean
                ir_soiou_loss = 1.0 - iou_gamma
                loss = ir_soiou_loss.mean() + self.alpha * lloss
            else:
                # Use IR-SOIoU's D_loss (default)
                ir_soiou_loss = 1.0 - iou_gamma + self.alpha * D_loss
                loss = ir_soiou_loss.mean()
        else:
            # Warm-up: use standard IoU loss
            loss = 1.0 - iou.mean()
        
        return loss
    
    def _get_bbox(self, mask, x_index, y_index, w, h):
        """
        Get bounding box coordinates [min_x, min_y, max_x, max_y] for each sample in batch.
        Coordinates are normalized to [0, 1].
        
        Args:
            mask: [B, C, H, W] tensor
            x_index: [H, W] tensor with normalized x coordinates
            y_index: [H, W] tensor with normalized y coordinates
            w, h: image dimensions (unused, kept for compatibility)
        """
        batch_size = mask.shape[0]
        bboxes = torch.zeros(batch_size, 4, dtype=mask.dtype, device=mask.device)
        
        # Threshold for active pixels
        threshold = 0.01
        
        for i in range(batch_size):
            # Get single channel mask [H, W]
            single_mask = mask[i, 0] if mask.shape[1] > 0 else mask[i]
            
            # Find non-zero (active) pixels
            active_mask = single_mask > threshold
            
            if active_mask.any():
                # Get coordinates of active pixels
                active_x = x_index[active_mask]
                active_y = y_index[active_mask]
                
                bboxes[i, 0] = active_x.min()  # min_x
                bboxes[i, 1] = active_y.min()  # min_y
                bboxes[i, 2] = active_x.max()  # max_x
                bboxes[i, 3] = active_y.max()  # max_y
            else:
                # If no active pixels, use full image
                bboxes[i, 0] = 0.0  # min_x
                bboxes[i, 1] = 0.0  # min_y
                bboxes[i, 2] = 1.0  # max_x
                bboxes[i, 3] = 1.0  # max_y
        
        return bboxes


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