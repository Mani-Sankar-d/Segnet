import torch

def calculate_iou(preds,target,n_classes):
    preds = torch.argmax(preds,dim=1)
    ious = []
    for cls in n_classes:
        pred_inds = (preds==cls)
        target_inds = (target==cls)
        union = (pred_inds | target_inds).sum()
        intersection = (pred_inds & target_inds).sum()
        iou = union/intersection
        ious.append(iou)
    return ious