import torch
import numpy as np

def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i] # predict
        # pred_boxes = output['boxes']
        # pred_scores = output['scores']
        # pred_labels = output['labels']

        true_positives = torch.zeros(output['boxes'].shape[0])   
 
        annotations = targets[sample_i]  # actual
        target_labels = annotations['labels'] if len(annotations) else []
        if len(annotations):    # len(annotations) = 3
            detected_boxes = []
            target_boxes = annotations['boxes']

            for pred_i, (pred_box, pred_label) in enumerate(zip(output['boxes'], output['labels'])): 

                # If targets are found break
                if len(detected_boxes) == len(target_labels): # annotations -> target_labels
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)   
                if iou >= iou_threshold and box_index not in detected_boxes: 
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]  
        batch_metrics.append([true_positives, output['scores'], output['labels']])
    return batch_metrics

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (tensor).
        conf:  Objectness value from 0-1 (tensor).
        pred_cls: Predicted object classes (tensor).
        target_cls: True object classes (tensor).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Sort by objectness
    i = torch.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = torch.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = torch.cumsum(1 - tp[i], -1)
            tpc = torch.cumsum(tp[i], -1)

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1].item())  # Convert to Python scalar

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1].item())  # Convert to Python scalar

            # AP from recall-precision curve
            ap_curve = compute_ap(recall_curve, precision_curve)
            ap.append(ap_curve)  # Ensure compute_ap returns a tensor and convert to Python scalar

    # Convert lists to tensors
    p, r, ap = torch.tensor(p), torch.tensor(r), torch.tensor(ap)
    
    # Ensure the tensors are on the same device as the input tensors
    device = tp.device
    p, r, ap = p.to(device), r.to(device), ap.to(device)

    # Compute F1 score (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes

def compute_ap(recall, precision):
    """Compute the average precision, using PyTorch, given the recall and precision curves.
    Adapted from code originally in https://github.com/rbgirshick/py-faster-rcnn for PyTorch compatibility.
    # Arguments
        recall:    The recall curve (Tensor).
        precision: The precision curve (Tensor).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = torch.cat((torch.tensor([0.0], device=recall.device), recall, torch.tensor([1.0], device=recall.device)))
    mpre = torch.cat((torch.tensor([0.0], device=precision.device), precision, torch.tensor([0.0], device=precision.device)))

    # compute the precision envelope
    for i in range(mpre.size(0) - 1, 0, -1):
        mpre[i - 1] = torch.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = torch.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = torch.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap.item()  # Convert to Python scalar for compatibility with list append operations


