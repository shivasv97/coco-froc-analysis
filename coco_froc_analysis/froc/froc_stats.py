from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..utils import get_iou_score


def init_stats(gt: dict, categories: dict) -> dict:
    """Initializing the statistics before counting leasion
       and non-leasion localiazations.

    Arguments:
        gt {dict} -- Ground truth COCO dataset
        categories {dict} -- Dictionary of categories in the COCO dataset

    Returns:
        stats {dict} -- Statistics to be updated, containing every information
                        necessary to evaluate a single FROC point
    """
    stats = {
        cat['id']: {
            'name': cat['name'],
            'LL': 0,
            'NL': 0,
            'n_images': [],
            'n_lesions': 0,
        }
        for cat in categories
    }
    for annotation in gt['annotations']:
        category_id = annotation['category_id']
        stats[category_id]['n_lesions'] += 1

    for image in gt['images']:
        image_id = image['id']
        for cat_id in stats:
            stats[cat_id]['n_images'].append(image_id)

    for cat_id in stats:
        stats[cat_id]['n_images'] = len(stats[cat_id]['n_images'])
    #print(stats)

    return stats


def update_stats(
    stats: dict,
    gt_id_to_annotation: dict,
    pr_id_to_annotation: dict,
    categories: dict,
    use_iou: bool,
    iou_thres: float,
    iou_thres_upper_bound:float
):
    """Updating statistics as going through images of the dataset.

    Arguments:
        stats {dict} -- FROC statistics
        gt_id_to_annotation {dict} -- Ground-truth image IDs to annotations.
        pr_id_to_annotation {dict} -- Prediction image IDs to annotations.
        categories {dict} -- COCO categories dictionary.
        use_iou {bool} -- Whether or not to use iou thresholding.
        iou_thres {float} -- IoU threshold when using IoU thresholding.
        iou_thres_upper_bound {float} -- IoU threshold upper bound.

    Returns:
        stats {dict} -- Updated FROC statistics
    """
    
    for image_id in gt_id_to_annotation:
        cat2anns: dict[int, dict[str, list]] = {}
        for cat in categories:
            # print("Category: ", cat)
            cat2anns[cat['id']] = {'gt': [], 'pr': []}
        
        for gt_ann in gt_id_to_annotation[image_id]:
            cat2anns[gt_ann['category_id']]['gt'].append(gt_ann)
        for pred_ann in pr_id_to_annotation.get(image_id, []):
            # print("Pred ann cat id: ", pred_ann['category_id'])
            try:
                cat2anns[pred_ann['category_id']]['pr'].append(pred_ann)
            except KeyError as ke:
                print(f"Keyerror for key {ke} occurred, possibly due to cat id {ke} being present in predictions, but not in GT. Since this will not affect the end result, ignoring...")

        for cat in categories:
            gt_anns = cat2anns[cat['id']]['gt']
            pr_anns = cat2anns[cat['id']]['pr']

            n_gt = len(gt_anns)
            n_pr = len(pr_anns)
            #print(n_gt, n_pr)
            
            # if n_gt == 0:
            #     if n_pr == 0:
            #         stats[cat['id']]['LL'] += 1 # ? Dont know if this is correct or not. This was not there before, so can be removed.
            #         pass
            #     else: ## newly added by me
            #         stats[cat['id']]['NL'] += n_pr
            # else:
            cost_matrix = np.zeros((n_gt, n_pr)) #* 1e6

            n_true_positives = 0
            n_false_positives = 0

            gt_hit = np.zeros(n_gt)
            pr_hit = np.zeros(n_pr)
            #print(len(gt_anns))
            #print(len(pr_anns))
            for gt_ind, gt_ann in enumerate(gt_anns):
                for pr_ind, pr_ann in enumerate(pr_anns):
                    if use_iou:
                        # if (image_id == 2999):
                        #     print(image_id, gt_ann['bbox'])
                        #     print(image_id, pr_ann['bbox'])
                        iou_score = get_iou_score(
                            gt_ann['bbox'],
                            pr_ann['bbox'],
                        )
                        #print(iou_score)
                        if iou_score >= iou_thres and iou_score < iou_thres_upper_bound and gt_hit[gt_ind] == 0 and pr_hit[pr_ind] == 0:
                            #print(image_id, cat, n_gt, n_pr, {gt_ind}, {pr_ind}, iou_score)
                            #print(iou_score, iou_thres)
                            #cost_matrix[gt_ind, pr_ind] = iou_score 
                            """ / (
                                np.random.uniform(0, 1) / 1e6
                            ) """
                            n_true_positives += 1
                            gt_hit[gt_ind] = 1
                            pr_hit[pr_ind] = 1
                        # else:
                        #     n_false_positives += 1
                        # elif iou_score < iou_thres: #and gt_hit[gt_ind] == 0: #and not(gt_hit[gt_ind] == 1 and pr_hit[pr_ind] == 0)
                        elif (iou_score < iou_thres and gt_hit[gt_ind] == 0) or (iou_score > iou_thres_upper_bound and gt_hit[gt_ind] == 0): #and not(gt_hit[gt_ind] == 1 and pr_hit[pr_ind] == 0)
                            n_false_positives += 1
                
                    else:
                        print("here")
                        gt_x, gt_y, gt_w, gt_h = gt_ann['bbox']

                        pr_x, pr_y, pr_w, pr_h = pr_ann['bbox']
                        pr_bbox_center = pr_x + pr_w / 2, pr_y + pr_h / 2

                        if (
                            pr_bbox_center[0] >= gt_x
                            and pr_bbox_center[0] <= gt_x + gt_w
                            and pr_bbox_center[1] >= gt_y
                            and pr_bbox_center[1] <= gt_y + gt_h
                        ):
                            print("here")
                            cost_matrix[gt_ind, pr_ind] = 1.0
                    
            #print(cost_matrix.shape, cost_matrix)
            # row_ind, col_ind = linear_sum_assignment(
            #     cost_matrix,
            # maximize=True)  # Hungarian-matching

            #n_true_positives = len(row_ind)
            #n_false_positives = max(n_pr - len(col_ind), 0)

            stats[cat['id']]['LL'] += n_true_positives
            #print(stats[cat['id']]['LL'])
            stats[cat['id']]['NL'] += n_false_positives
            #print(stats[cat['id']]['NL'])

    return stats
