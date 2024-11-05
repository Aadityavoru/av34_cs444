import torch
import numpy as np
import random

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    
def get_detections(outs):
    B, BB, _, _ = outs[0][0].shape
    _, A, _, _ = outs[0][2].shape
    A = A // 4
    num_classes = BB // A
    
    pred_bboxes, pred_clss, anchors = [], [], []
    for pred_cls, pred_bbox, anchor in outs:
        # Get all the anchors, pred and bboxes
        H, W = pred_cls.shape[-2:]
        pred_cls = pred_cls.reshape(B, A, -1, H, W)
        pred_bbox = pred_bbox.reshape(B, A, -1, H, W)

        pred_clss.append(pred_cls.permute(0, 1, 3, 4, 2).reshape(B, -1, num_classes))
        pred_bboxes.append(pred_bbox.permute(0, 1, 3, 4, 2).reshape(B, -1, 4))

        anchor = anchor.reshape(B, A, -1, H, W)
        anchors.append(anchor.permute(0, 1, 3, 4, 2).reshape(B, -1, 4))
    pred_clss = torch.cat(pred_clss, dim=1)
    pred_bboxes = torch.cat(pred_bboxes, dim=1)
    anchors = torch.cat(anchors, dim=1)
    return pred_clss, pred_bboxes, anchors

def compute_bbox_iou(bbox1, bbox2, dim=1):
    """
    Args:
        bbox1: (N, 4) tensor of (x1, y1, x2, y2)
        bbox2: (M, 4) tensor of (x1, y1, x2, y2)
    Returns
        iou: (N, M) tensor of IoU values
    """
    bbox1 = bbox1.unsqueeze(1)
    bbox2 = bbox2.unsqueeze(0)
    max_min_x = torch.max(bbox1[...,0], bbox2[...,0])
    min_max_x = torch.min(bbox1[...,2], bbox2[...,2])
    max_min_y = torch.max(bbox1[...,1], bbox2[...,1])
    min_max_y = torch.min(bbox1[...,3], bbox2[...,3])
    intersection = torch.clamp(min_max_x - max_min_x, min=0) * torch.clamp(min_max_y - max_min_y, min=0)
    area1 = (bbox1[...,2] - bbox1[...,0]) * (bbox1[...,3] - bbox1[...,1])
    area2 = (bbox2[...,2] - bbox2[...,0]) * (bbox2[...,3] - bbox2[...,1])
    iou = intersection / (area1 + area2 - intersection)
    return iou

def compute_targets(anchor, cls, bbox):
    """
    Args:
        anchor: batch of anchors in the format (x1, y1, x2, y2); shape is (B, A, 4)
        cls: groundtruth object classes of shape (B, number of objects in the image, 1)
        bbox: groundtruth bounding boxes of shape (B, number of objects in the image, 4)
    Returns:
        gt_clss: groundtruth class targets of shape (B, A, 1)
        gt_bboxes: groundtruth bbox targets of shape (B, A, 4)
    """
    B, A, _ = anchor.shape
    gt_clss = torch.zeros((B, A, 1), dtype=torch.int, device=anchor.device)  # Shape (B, A, 1)
    gt_bboxes = torch.zeros((B, A, 4), device=anchor.device)

    for b in range(B):
        ious = compute_bbox_iou(anchor[b], bbox[b])  # (A, number of objects)
        max_ious, max_indices = torch.max(ious, dim=1)  # (A,)
        
        # Assign to background if IoU < 0.4
        gt_clss[b][max_ious < 0.4] = 0
        
        # Assign to ignore class if 0.4 <= IoU < 0.5
        ignore_mask = (max_ious >= 0.4) & (max_ious < 0.5)
        gt_clss[b][ignore_mask] = -1
        
        # Assign to ground truth if IoU >= 0.5
        valid_mask = max_ious >= 0.5
        if valid_mask.any():  # Check if there are valid indices
            assigned_indices = max_indices[valid_mask]
            gt_clss[b][valid_mask] = cls[b][assigned_indices].to(torch.int).view(-1, 1)  # Ensure correct shape and type
            gt_bboxes[b][valid_mask] = bbox[b][assigned_indices]

    return gt_clss, gt_bboxes


def compute_bbox_targets(anchors, gt_bboxes):
    """
    Args:
        anchors: anchors of shape (A, 4)
        gt_bboxes: groundtruth bounding boxes of shape (A, 4)
    Returns:
        bbox_reg_target: regression offset of shape (A, 4)
    """
    # Extract coordinates of anchors
    anchor_x1 = anchors[:, 0]
    anchor_y1 = anchors[:, 1]
    anchor_x2 = anchors[:, 2]
    anchor_y2 = anchors[:, 3]

    # Compute widths, heights, and center coordinates of anchors
    anchor_widths = anchor_x2 - anchor_x1
    anchor_heights = anchor_y2 - anchor_y1
    anchor_center_x = anchor_x1 + 0.5 * anchor_widths
    anchor_center_y = anchor_y1 + 0.5 * anchor_heights

    # Extract coordinates of ground truth boxes
    gt_x1 = gt_bboxes[:, 0]
    gt_y1 = gt_bboxes[:, 1]
    gt_x2 = gt_bboxes[:, 2]
    gt_y2 = gt_bboxes[:, 3]

    # Compute widths, heights, and center coordinates of ground truth boxes
    gt_widths = gt_x2 - gt_x1
    gt_heights = gt_y2 - gt_y1
    gt_center_x = gt_x1 + 0.5 * gt_widths
    gt_center_y = gt_y1 + 0.5 * gt_heights

    # Avoid division by zero and log of zero by clamping widths and heights
    eps = 1e-6
    anchor_widths = torch.clamp(anchor_widths, min=eps)
    anchor_heights = torch.clamp(anchor_heights, min=eps)
    gt_widths = torch.clamp(gt_widths, min=1.0)
    gt_heights = torch.clamp(gt_heights, min=1.0)

    # Compute regression targets
    delta_x = (gt_center_x - anchor_center_x) / anchor_widths
    delta_y = (gt_center_y - anchor_center_y) / anchor_heights
    delta_w = torch.log(gt_widths / anchor_widths)
    delta_h = torch.log(gt_heights / anchor_heights)

    # Stack the deltas to form the regression targets
    bbox_reg_target = torch.stack([delta_x, delta_y, delta_w, delta_h], dim=-1)

    return bbox_reg_target


def apply_bbox_deltas(boxes, deltas):
    """
    Args:
        boxes: (N, 4) tensor of (x1, y1, x2, y2)
        deltas: (N, 4) tensor of (dxc, dyc, dlogw, dlogh)
    Returns:
        boxes: (N, 4) tensor of (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = boxes.unbind(1)

    # Compute the center and size of the boxes
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1

    # Apply the deltas
    new_center_x = center_x + deltas[:, 0] * width
    new_center_y = center_y + deltas[:, 1] * height
    new_width = width * torch.exp(deltas[:, 2])
    new_height = height * torch.exp(deltas[:, 3])

    # Calculate the new boxes
    new_x1 = new_center_x - new_width / 2
    new_y1 = new_center_y - new_height / 2
    new_x2 = new_center_x + new_width / 2
    new_y2 = new_center_y + new_height / 2

    return torch.stack([new_x1, new_y1, new_x2, new_y2], dim=1)


def nms(bboxes, scores, threshold=0.5):
    """
    Args:
        bboxes: (N, 4) tensor of (x1, y1, x2, y2)
        scores: (N,) tensor of scores
    Returns:
        keep: (K,) tensor of indices to keep
    """
    # Sort the boxes by scores in descending order
    sorted_indices = scores.argsort(descending=True)
    keep = []

    while sorted_indices.numel() > 0:
        # Get the index of the box with the highest score
        current = sorted_indices[0]
        keep.append(current.item())

        if sorted_indices.numel() == 1:
            break

        # Compute IoUs of the current box with the remaining boxes
        ious = compute_bbox_iou(bboxes[current].unsqueeze(0), bboxes[sorted_indices[1:]])[0]

        # Select boxes with IoU less than the threshold
        sorted_indices = sorted_indices[1:][ious < threshold]

    return torch.tensor(keep, dtype=torch.int64)  # Ensure the output is of type int

