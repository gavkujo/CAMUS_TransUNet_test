import torch
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import directed_hausdorff

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(dataloader.dataset)

def evaluate(model, dataloader, criterion, device):
    """
    Evaluate the model on the validation or test dataset.
    """
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_hausdorff_95 = 0.0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1).cpu().numpy()
            labels = labels.cpu().numpy()

            for pred, label in zip(preds, labels):
                total_dice += dice_coeff(pred, label)
                total_iou += iou_score(pred, label)
                total_hausdorff_95 += hausdorff_distance(label, pred)

    num_samples = len(dataloader.dataset)
    return (total_loss / num_samples,
            total_dice / num_samples,
            total_iou / num_samples,
            total_hausdorff_95 / num_samples)

def dice_coeff(pred, target, epsilon=1e-6):
    """
    Compute the DICE coefficient for a single pair of prediction and target.
    """
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2.0 * intersection + epsilon) / (union + epsilon)

def iou_score(pred, target, epsilon=1e-6):
    """
    Compute the Intersection over Union (IoU) score.
    """
    intersection = (pred & target).sum()
    union = (pred | target).sum()
    return (intersection + epsilon) / (union + epsilon)

def hausdorff_distance(label, pred):
    """
    Compute the 95th percentile of the Hausdorff Distance between the label and prediction.
    """
    label_points = np.argwhere(label > 0)
    pred_points = np.argwhere(pred > 0)

    if label_points.size == 0 or pred_points.size == 0:
        return np.inf

    forward_hausdorff = directed_hausdorff(label_points, pred_points)[0]
    backward_hausdorff = directed_hausdorff(pred_points, label_points)[0]
    return max(forward_hausdorff, backward_hausdorff)
