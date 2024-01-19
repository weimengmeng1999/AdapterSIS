import torch
import torch.nn.functional as F

def iou(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-6) / (union + 1e-6)

def iou_loss(preds, labels, smooth=1e-6, num_classes=8):
    """
    Calculate the Intersection over Union (IoU) loss for multi-class segmentation.

    Args:
    preds (torch.Tensor): The predictions from the model. Shape: (batch_size, num_classes, height, width)
    labels (torch.Tensor): The ground truth labels. Shape: (batch_size, height, width)
    smooth (float): A small constant to avoid division by zero.
    num_classes (int): The number of classes.

    Returns:
    torch.Tensor: The mean IoU loss.
    """
    # Convert labels to one-hot encoding
    labels_one_hot = F.one_hot(labels, num_classes).permute(0, 3, 1, 2)
    # print(labels_one_hot.size())
    #.permute(0, 3, 1, 2)
    
    # Apply softmax to the predictions
    preds_softmax = F.softmax(preds, dim=1)

    loss = 0.0

    for cls in range(num_classes):
        # Extract the predictions and labels for the current class
        preds_cls = preds_softmax[:, cls, :, :]
        labels_cls = labels_one_hot[:, cls, :, :]

        # Compute the intersection and union
        intersection = torch.sum(preds_cls * labels_cls, dim=[1, 2])
        union = torch.sum(preds_cls, dim=[1, 2]) + torch.sum(labels_cls, dim=[1, 2]) - intersection

        # Compute the IoU and IoU loss for the current class
        iou = (intersection + smooth) / (union + smooth)
        iou_loss_cls = 1 - iou

        # Add the IoU loss for the current class to the total loss
        loss += iou_loss_cls.mean()

    # Compute the mean IoU loss over all classes
    return loss / num_classes
    
def ch_iou(y_true, y_pred):
    result = []

    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0

    for type_id in set(y_true.flatten()):
        if type_id == 0:
            continue
        result += [iou(y_true == type_id, y_pred == type_id)]

    return np.mean(result)

def isi_iou(y_true, y_pred, problem_type='instruments'):
    result = []

    if problem_type == 'binary':
        type_number = 2
    elif problem_type == 'parts':
        type_number = 4
    elif problem_type == 'instruments':
        type_number = 8

    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0

    for type_id in range(type_number):
        if type_id == 0:
            continue
        if (y_true == type_id).sum() != 0 or (y_pred == type_id).sum() != 0:
            result += [iou(y_true == type_id, y_pred == type_id)]

    return np.mean(result)