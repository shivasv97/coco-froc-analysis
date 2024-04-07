import json
import numpy as np
import matplotlib.pyplot as plt

import json
import numpy as np
import matplotlib.pyplot as plt

def calculate_iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    iou = intersection / union
    
    return iou

def calculate_froc(coco_gt, coco_predictions, iou_threshold):
    # Load the ground truth and predictions from COCO-formatted JSON files
    with open(coco_gt, 'r') as f:
        gt_data = json.load(f)
    with open(coco_predictions, 'r') as f:
        pred_data = json.load(f)
    
    # Extract the annotations and predictions
    gt_annotations = gt_data['annotations']
    pred_annotations = pred_data
    
    # Get the list of class IDs and class names
    class_ids = [category['id'] for category in gt_data['categories']]
    class_names = [category['name'] for category in gt_data['categories']]
    
    for class_id, class_name in zip(class_ids, class_names):
        tp = 0
        fp = 0
        fn = 0
        
        for gt_ann in gt_annotations:
            if gt_ann['category_id'] == class_id:
                ious = []
                
                for pred_ann in pred_annotations:
                    if pred_ann['category_id'] == class_id:
                        iou = calculate_iou(gt_ann['bbox'], pred_ann['bbox'])
                        ious.append(iou)
                
                if len(ious) > 0:
                    max_iou = max(ious)
                    
                    if max_iou >= iou_threshold:
                        tp += 1
                    else:
                        fn += 1
                else:
                    fn += 1
        
        for pred_ann in pred_annotations:
            if pred_ann['category_id'] == class_id:
                ious = []
                
                for gt_ann in gt_annotations:
                    if gt_ann['category_id'] == class_id:
                        iou = calculate_iou(gt_ann['bbox'], pred_ann['bbox'])
                        ious.append(iou)
                
                if len(ious) > 0:
                    max_iou = max(ious)
                    
                    if max_iou < iou_threshold:
                        fp += 1
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        avg_fp_per_image = fp / len(gt_data['images'])
        
        # Plot FROC curve for the current class
        plt.plot(avg_fp_per_image, sensitivity, marker='o')
        plt.xlabel('Average False Positives per Image')
        plt.ylabel('Sensitivity (Recall)')
        plt.title(f'FROC Curve - {class_name}')
        plt.grid(True)
        plt.savefig(fname=f"test_froc_{class_id}.png", dpi=150)

# Path to COCO-formatted ground truth and predictions JSON files
coco_gt_file = '/scratch/ssenth21/InternImage/detection/data/vindr/annotations/annotations_test.json'
coco_predictions_file = '/scratch/ssenth21/InternImage/detection/results.bbox.json'
# IOU threshold for calculating FROC curve
iou_threshold = 0.4

# Calculate and plot FROC curve for each class
calculate_froc(coco_gt_file, coco_predictions_file, iou_threshold)




""" def plot_froc_curve(sensitivity, avg_fp_per_image):
    plt.plot(avg_fp_per_image, sensitivity, marker='o')
    plt.xlabel('Average False Positives per Image')
    plt.ylabel('Sensitivity (Recall)')
    plt.title('FROC Curve')
    plt.grid(True)
    
    plt.savefig(fname="test_froc.png", dpi=150)
    #plt.show()

# Path to COCO-formatted ground truth and predictions JSON files
coco_gt_file = '/scratch/ssenth21/InternImage/detection/data/vindr/annotations/annotations_test.json'
coco_predictions_file = '/scratch/ssenth21/InternImage/detection/results.bbox.json'

# IOU threshold for calculating FROC curve
iou_threshold = 0.4

# Calculate FROC curve
sensitivity, avg_fp_per_image = calculate_froc(coco_gt_file, coco_predictions_file, iou_threshold)

# Plot FROC curve
plot_froc_curve(sensitivity, avg_fp_per_image) """
