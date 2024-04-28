import json
import numpy as np
import os

def create_missing_dirs(path):
    os.makedirs(path, exist_ok=True)

def json_to_object(file_path):
    with open(file_path, 'r') as file_obj:
        json_object = json.load(file_obj)
        
    return json_object

def filter_preds_on_conf_thres(pred_json_obj, score_thres):
    filtered_preds = []
    assert type(pred_json_obj) == list, "Unable to filter conf scores as the pred JSON obj is not a list."
    for pred_item in pred_json_obj:
        if float(pred_item['score']) > score_thres:
            filtered_preds.append(pred_item)
    return filtered_preds

def gt_pred_category_id_consistency_check(gt_json_obj, pred_json_obj, ignore_check):
    gt_category_ids = [x['id'] for x in gt_json_obj['categories']]
    pred_category_ids = [x['category_id'] for x in pred_json_obj]
    is_pred_cats_within_gt_cats = set(pred_category_ids).issubset(set(gt_category_ids))
    if not ignore_check:
        assert is_pred_cats_within_gt_cats is True, 'Predictions have a category id that is not seen in GT categories list'
    return gt_category_ids

def category_filter(gt_annotations, pred_json_obj, category_id):
    cat_filtered_gt_annotations = []
    cat_filtered_pred_json_obj = []
    for annotation in gt_annotations:
        if annotation['category_id'] == category_id:
            cat_filtered_gt_annotations.append(annotation)
    for pred in pred_json_obj:
        if pred['category_id'] == category_id:
            cat_filtered_pred_json_obj.append(pred)
            
    return cat_filtered_gt_annotations, cat_filtered_pred_json_obj

def gt_normal_image_ids(gt_json_obj):
    img_with_annot = set([x['image_id'] for x in gt_json_obj['annotations']])
    all_imgs = set([x['id'] for x in gt_json_obj['images']])
    normal_imgs = all_imgs - img_with_annot
    return list(normal_imgs)

def iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    
    bb1 = [bb1[0], bb1[1], bb1[2]+bb1[0], bb1[3]+bb1[1]]
    bb2 = [bb2[0], bb2[1], bb2[2]+bb2[0], bb2[3]+bb2[1]]
    
    if not (bb1[0] < bb1[2]):
        return 0.0
    if not (bb1[1] < bb1[3]):
        return 0.0
    if not (bb2[0] < bb2[2]):
        return 0.0
    if not (bb2[1] < bb2[3]):
        return 0.0

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0] + 1) * (bb1[3] - bb1[1] + 1)
    bb2_area = (bb2[2] - bb2[0] + 1) * (bb2[3] - bb2[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def depr_iou(gt_bbox, pred_bbox): # remove depr_ from the function name here to use this function. Correspondingly, comment out the other function.
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Parameters:
    bbox1 (numpy.ndarray): Coordinates of the first bounding box in the format (x1, y1, w1, h1).
    bbox2 (numpy.ndarray): Coordinates of the second bounding box in the format (x2, y2, w2, h2).

    Returns:
    float: Intersection over Union (IoU) score.
    """
    
    if type(gt_bbox) is not np.ndarray:
        gt_bbox = np.array(gt_bbox)
    if type(pred_bbox) is not np.ndarray:
        pred_bbox = np.array(pred_bbox)
        
    assert type(gt_bbox) is np.ndarray, "gt_bbox in the utils.iou function is not of type numpy array."
    assert type(pred_bbox) is np.ndarray, "pred_bbox in the utils.iou function is not of type numpy array."
    
    # Extract coordinates
    x1, y1, w1, h1 = gt_bbox
    x2, y2, w2, h2 = pred_bbox
    
    # Calculate intersection coordinates
    x_left = np.maximum(x1, x2)
    y_top = np.maximum(y1, y2)
    x_right = np.minimum(x1 + w1, x2 + w2)
    y_bottom = np.minimum(y1 + h1, y2 + h2)
    
    # Calculate intersection area
    intersection_area = np.maximum(0, x_right - x_left) * np.maximum(0, y_bottom - y_top)
    
    
    # Calculate union area
    union_area = w1 * h1 + w2 * h2 - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area
    
    return iou