import torch

def dice_score(pred_mask, gt_mask):
    # Ensure binary masks with proper thresholding (detach from computation graph if needed)
    with torch.no_grad():
        # apply sigmoid the the prediction mask
        pred_mask = (torch.sigmoid(pred_mask) > 0.5).float().flatten()
        gt_mask = (gt_mask > 0.5).float().flatten()
        
        intersection = torch.sum(pred_mask * gt_mask)
        union = torch.sum(pred_mask) + torch.sum(gt_mask)
        
        dice = (2.0 * intersection) / (union + 1e-8)  # add a small epsilon to avoid division by zero
        
        return dice.item()