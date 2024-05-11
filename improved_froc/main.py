'''

'''
import numpy as np
from tqdm import tqdm
import os.path as osp
import matplotlib.pyplot as plt
import pprint
import json
import argparse

from utils import json_to_object, filter_preds_on_conf_thres, gt_pred_category_id_consistency_check, category_filter, iou, gt_normal_image_ids, create_missing_dirs

def is_gt_or_eq_iou_thres(gt_bbox, pred_bbox, iou_thres):
    if (iou(gt_bbox, pred_bbox) >= iou_thres):
        return True
    else:
        return False

def per_cat_per_image_sens_fpr_calculator(per_cat_per_image_gt_annots, per_cat_per_image_preds, iou_thres):
    gt_len = len(per_cat_per_image_gt_annots)
    pred_len = len(per_cat_per_image_preds)
    
    gt_bboxes = np.array([x['bbox'] for x in per_cat_per_image_gt_annots])
    pred_bboxes = np.array([x['bbox'] for x in per_cat_per_image_preds])
    # pred_bboxes = np.expand_dims(pred_bboxes, axis=1)  # Shape: (n, 1, 4)
    # gt_bboxes = np.expand_dims(gt_bboxes, axis=0)  # Shape: (1, m, 4)
    
    TN_per_img_per_cat = 0
    per_cat_per_img_FP = 0
    # If either pred_len and/or gt_len are zero, we can determine TN(pred_len and gt_len are zero) or FN(if pred_len only zero) or FP(if gt_len only zero)
    if gt_len==0 and pred_len==0:
        TN_per_img_per_cat += 1 # This will be used for the No Findings FROC curve and values. Since we would be calculating the TD and FPR for some particular category, we will have to consider all the categories. Only if all the cats report the TN, then this would be used for the No Findings FROC.
        return 0,0,TN_per_img_per_cat, gt_len # TODO: have to fill in the True Detection and FPR return values here instead of return None only.
    elif gt_len==0 and pred_len>0:
        per_cat_per_img_FP += 1
        return 0, pred_len, TN_per_img_per_cat, gt_len # TODO: have to fill in the True Detection and FPR return values here instead of return None only.: DONE
    elif gt_len>0 and pred_len==0: # For false negatives, consider the case where conf thres is 1. We have no predictions and only GTs. This makes the sens go to 0 as there are no detections at all to say if they are correct. Similarly, for FPR, there are no predictions, hence no FP preds, thus FPR is 0.
        pass # TODO: have to fill in the True Detection and FPR return values here instead of pass and return None only.: DONE
        return 0,0, TN_per_img_per_cat, gt_len
    
    else:
        sens_fpr_table = np.zeros((pred_len, gt_len))
        
        # Using numpy broadcasting for efficiently applying IoU calc on the numpy 'table'
        for i in range(sens_fpr_table.shape[0]): # loop over pred's len
            for j in range(sens_fpr_table.shape[1]): # loop over gt's len
                sens_fpr_table[i][j] = int(is_gt_or_eq_iou_thres(gt_bboxes[j], pred_bboxes[i], iou_thres))
                
        num_TD_per_img_per_cat = np.sum(np.any(sens_fpr_table, axis=0)) # np.any makes sure that even overlapping predictions that are above conf thres and iou thres are NOT considered as FP detections. 
        num_FP_per_img_per_cat = np.sum(np.all(sens_fpr_table == 0, axis=1)) # np.all makes sure that a particular prediction is not a positive prediction for any GT in the image for that category.
        return num_TD_per_img_per_cat, num_FP_per_img_per_cat, TN_per_img_per_cat, gt_len

def per_cat_sens_fpr_calculator(image_ids, cat_filtered_gt_annots, cat_filtered_preds, iou_thres):
    per_cat_per_img_stats = {}
    for image_id in image_ids:
        per_cat_per_img_stats.setdefault(image_id, {'TD per image per cat': 0, 'FP per image per cat': 0, 'TN per image per cat': 0, 'GT per image per cat':0})
        per_cat_per_img_gt_annots = [x for x in cat_filtered_gt_annots if x['image_id']==image_id]
        per_cat_per_img_preds = [x for x in cat_filtered_preds if x['image_id']==image_id]
        num_TD_per_img_per_cat, num_FP_per_img_per_cat, TN_per_img_per_cat, num_GT_per_img_per_cat = per_cat_per_image_sens_fpr_calculator(per_cat_per_img_gt_annots, per_cat_per_img_preds, iou_thres)
        per_cat_per_img_stats[image_id]['TD per image per cat'] += num_TD_per_img_per_cat
        per_cat_per_img_stats[image_id]['FP per image per cat'] += num_FP_per_img_per_cat
        per_cat_per_img_stats[image_id]['TN per image per cat'] += TN_per_img_per_cat
        per_cat_per_img_stats[image_id]['GT per image per cat'] += num_GT_per_img_per_cat
        # per_cat_per_img_stats[image_id] = {
        #     'TD per image per cat':num_TD_per_img_per_cat,
        #     'FP per image per cat':num_FP_per_img_per_cat,
        #     'TN per image per cat':TN_per_img_per_cat, 
        #     'GT per image per cat':num_GT_per_img_per_cat
        # }
    return per_cat_per_img_stats

def normal_image_sens_fpr_calculator(normal_img_ids, normal_preds, all_img_ids, imgs_with_preds_set):
    gt_normal_set = set(normal_img_ids)
    pred_normal_set = set(normal_preds)
    diseased_gt_set = set(all_img_ids) - gt_normal_set
    # print(f'Inside function Diseased GT, normal GT, total images, normal preds: {len(diseased_gt_set)}, {len(gt_normal_set)}, {len(all_img_ids)}, {len(pred_normal_set)}')

    # Sensitivity of normal images = Correctly detected normal images / Total number of GT normal images. 
    # FPR of normal images = Images that are predicted normal but GT is diseased / Total number of diseased images. == 1 - specificity. 
    # Specificity of normal images = Images that are correctly identified as diseased / Total number of diseased images.
    
    TP_normal = gt_normal_set & pred_normal_set # Images that are correctly identified as normal.
    FP_normal = pred_normal_set - gt_normal_set # Images that are falsely identified as normal. Remaining elements of predictions that are not present in the GT normal set.
    TN_normal = imgs_with_preds_set & diseased_gt_set # TN_normal implies True negatives for normals, which is basically diseased images that are identified correctly as so.
    # print(f'TP_normal, FP_normal, TN_normal: {len(TP_normal)}, {len(FP_normal)}, {len(TN_normal)}')
    
    Sens_normal = len(TP_normal) / len(gt_normal_set)
    FPR_normal = len(FP_normal) / len(diseased_gt_set)
    Specificity_normal = len(TN_normal) / len(diseased_gt_set)
    assert Specificity_normal + FPR_normal == 1.0 , f"Specificity != 1-FPR for normal images. Check logic in code. Spec: {Specificity_normal}, FPR: {FPR_normal}" # Due to numbers like Spec: 0.9556025369978859, FPR: 0.04439746300211417 which are never ending, the sum is the only way rather than direct Specificity_normal == 1.0 - FPR_normal as it will throw assert error as it cant work for more than few decimal places? 
    # TODO: Bug in python subtraction? Doing Specificity_normal == 1.0 - FPR_normal throws error for Spec: 0.9556025369978859, FPR: 0.04439746300211417
    # But doing Specificity_normal + FPR_normal == 1.0 for those values are not throwing error.
    
    # TN_normal = gt_normal_set & pred_normal_set # Basically TP for 'normal' images, but technically it is TN as normal is the lack of other classes.
    # FN_normal = gt_normal_set - pred_normal_set
    # FP_normal = pred_normal_set - gt_normal_set # prediction is normal, but GT is not normal
    
    # sens_normal = len(TN_normal)/len(gt_normal_set)
    # fpr_normal = len(FP_normal)/len(all_img_ids)
    #print(sens_normal, fpr_normal)
    
    return Sens_normal, Specificity_normal
    # return Sens_normal, FPR_normal

def plot_per_cat_froc_curve(cat_label, fpr_list, sens_list, save_dir, iou_thres, eval_thresholds, interp_sens, plot_eval_thres, eval_thres_markers, plot_sota_markers, sota_fpr_list, sota_sens_list, extrapolate_plot, plot_type='FROC'):
    cat_label = cat_label.replace('/', '-')
    cat_label = cat_label.replace('.', '_')
    save_dir = osp.join(save_dir, str(iou_thres).replace('.','_'))
    save_path = osp.join(save_dir, f'FROC_{cat_label}.png')
    create_missing_dirs(save_dir)
    
    # Reverse the FPR and Sens list for proper zero based plot. 
    # fpr_list.reverse()
    # sens_list.reverse()
    
    if extrapolate_plot:
        fpr_list_extra=np.arange(1, 1.4+0.1, 0.1).tolist()
        sens_list_extra = np.interp(fpr_list_extra, fpr_list, sens_list)
        fpr_list.extend(fpr_list_extra)
        sens_list.extend(sens_list_extra)
    elif not extrapolate_plot:
        fpr_list_extra = fpr_list
        sens_list_extra = sens_list
    
    
    if plot_eval_thres and extrapolate_plot:
        sens_list.extend(interp_sens.tolist())
        fpr_list.extend(list(eval_thresholds))
        # Check if there are border cases where the sort can screw up the ordering of the sens_list and the corresponding fpr_list elements.
        # sens_list = sorted(sens_list) 
        # fpr_list = sorted(fpr_list)
    
    # elif plot_eval_thres:
        
    
    x_ticks = np.arange(0, 1.4+0.1, 0.1).tolist()
    x_ticks_w_eval_thres = sorted(x_ticks + list(eval_thresholds))
    if plot_eval_thres and plot_type == "FROC":
        final_xticks = list(eval_thresholds)
    if not plot_eval_thres and plot_type == "FROC":
        final_xticks = [x for x in x_ticks_w_eval_thres if x <= 1.4]
    if plot_type == "ROC":
        final_xticks = x_ticks_w_eval_thres[::-1]
    
    sensitivities_text = "\n".join([f'Sens={interp_sens[i]:.4f}@{eval_thresholds[i]} FP/image' for i in range(len(eval_thresholds))])
    if plot_type == "ROC":
        sensitivities_text = "\n".join([f'Sens={interp_sens[i]:.4f}@{eval_thresholds[i]} Specificity' for i in range(len(eval_thresholds))])
    #print(sensitivities_text)
    
    plt.plot(fpr_list, sens_list, linestyle='-')
    if eval_thres_markers:
        if extrapolate_plot:
            for i, x_val in enumerate(fpr_list):
                if x_val in eval_thresholds:
                    plt.scatter(x_val, sens_list[i], color='red', marker='*', label='Eval thresholds')  # Mark points in list A
                    plt.text(x_val, sens_list[i], f'{x_val} FPR', fontsize=5, ha='right', va='bottom') # Seems there is some repetition for elements in the FPR, sens list. Check this out based on the processing being done above.
        elif not extrapolate_plot:
            for i, x_val in enumerate(eval_thresholds):
                plt.scatter(x_val, interp_sens[i], color='red', marker='*', label='Eval thresholds')  # Mark points in list A
                plt.text(x_val, interp_sens[i], f'{x_val} FPR', fontsize=5, ha='right', va='bottom') # Seems there is some repetition for elements in the FPR, sens list. Check this out based on the processing being done above.
            # if len(eval_thresholds) == 0:
            if plot_type == "FROC":
                plt.scatter(fpr_list[0], sens_list[0], color='blue', marker='*', label='final point')  # Mark points in list A
                plt.text(fpr_list[0], sens_list[0], f'{fpr_list[0]:.2f} FPR', fontsize=5, ha='right', va='bottom') # Seems there is some repetition for elements in the FPR, sens list. Check this out based on the processing being done above.
            if plot_type == "ROC":
                pass
                plt.scatter(fpr_list[-1], sens_list[-1], color='blue', marker='*', label='final point')  # Mark points in list A
                plt.text(fpr_list[-1], sens_list[-1], f'{fpr_list[-1]:.4f} Specificity', fontsize=5, ha='right', va='bottom') # Seems there is some repetition for elements in the FPR, sens list. Check this out based on the processing being done above.
                
            # else:
            # plt.plot(x_val, sens_list[i], color='blue', marker=' ', linestyle='-')  # Don't mark other points
    if plot_sota_markers:
        for i_, sota_x_val in enumerate(sota_fpr_list):
            plt.scatter(sota_x_val, sota_sens_list[i_], color='green', marker='*', label='SOTA values')  # Mark points in list A
            if plot_type == "FROC":
                plt.text(sota_x_val, sota_sens_list[i_], f'SOTA {sota_x_val} FPR', fontsize=5, ha='right', va='bottom') 
            if plot_type == "ROC":
                plt.text(sota_x_val, sota_sens_list[i_], f'SOTA Sens: {sota_sens_list[i_]},\n Specificity: {sota_x_val}', fontsize=5, ha='right', va='bottom') 

    # else:
    # plt.plot(fpr_list, sens_list, marker='o', linestyle='-')
    # plt.plot(fpr_list_extra, sens_list_extra, 'rx', linestyle='-')
    if plot_type == "ROC":
        plt.xlabel('Specificity')
    else:    
        plt.xlabel('FP/image')
    plt.ylabel('Sens')
    plt.title(f'Class {cat_label}')
    plt.grid(True)
    # if plot_eval_thres:
    #     plt.xlim(left=-0.1, right=sens_list[-1])
    plt.yticks(np.arange(0, 1.2, 0.1))
    if plot_type == "ROC":
        plt.xlim(max(fpr_list+eval_thresholds)+0.1, min(fpr_list+eval_thresholds))
    if not plot_eval_thres and plot_type == "FROC":
        plt.xticks(final_xticks, rotation=45)
        plt.xlim(left=-0.1, right=1.4)
        plt.ylim(bottom=-0.1, top=1.1)
    # plt.text(0.5, -2, sensitivities_text, ha='center', fontsize=10)
    # plt.text(1.1, 0.5, sensitivities_text, fontsize=8, verticalalignment='bottom', horizontalalignment='left')
    plt.text(1.05, 0.5, sensitivities_text, fontsize=8, verticalalignment='center', horizontalalignment='left', transform=plt.gca().transAxes)
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()

def main():
    gt_obj = json_to_object(gt_file_path)
    pred_obj = json_to_object(pred_file_path)
    gt_image_ids = [x['id'] for x in gt_obj['images']]
    gt_cat_id_to_labels = {x['id']:x['name'] for x in gt_obj['categories']}
    normal_img_ids = gt_normal_image_ids(gt_obj)
    num_images = len(gt_image_ids)
    gt_cat_ids = gt_pred_category_id_consistency_check(gt_obj, pred_obj, IGNORE_STRICT_CHECK)
    cat_stats={}
    normal_stats={}
    for conf_score_thres in tqdm(np.linspace(0, 1, num=num_sample_points, endpoint=True, retstep=False, dtype=None, axis=0)):
        conf_score_filtered_preds = filter_preds_on_conf_thres(pred_obj, conf_score_thres)
        imgs_with_preds_set = set([x['image_id'] for x in conf_score_filtered_preds]) # diseased predictions
        normal_preds = list(set(gt_image_ids) - imgs_with_preds_set)
        # print(f'Diseased preds, normal preds, total images: {len(imgs_with_preds_set)}, {len(normal_preds)}, {len(gt_image_ids)}')
        cat_stats[conf_score_thres] = {}
        normal_stats[conf_score_thres] = {}
        for category_id in gt_cat_ids:
            cat_filtered_gt_annots, cat_filtered_preds = category_filter(gt_obj['annotations'], conf_score_filtered_preds, category_id)
            per_cat_per_img_stats = per_cat_sens_fpr_calculator(gt_image_ids, cat_filtered_gt_annots, cat_filtered_preds, iou_thres)
            TD_per_cat = 0
            FP_per_cat = 0
            GTs_per_cat = 0
            for image_id, stats in per_cat_per_img_stats.items():
                #print(category_id, image_id, stats)
                TD_per_cat += stats['TD per image per cat']
                FP_per_cat += stats['FP per image per cat']
                GTs_per_cat += stats['GT per image per cat']
            sens_per_cat =  TD_per_cat/GTs_per_cat
            fpr_per_cat = FP_per_cat/num_images
            cat_stats[conf_score_thres][category_id] = {
            'sensitivity': sens_per_cat,
            'fpr': fpr_per_cat
            }   
        if len(normal_img_ids)>0:  
            sens_normal, fpr_normal = normal_image_sens_fpr_calculator(normal_img_ids, normal_preds, gt_image_ids, imgs_with_preds_set) #TODO: change fpr_normal to specificity_normal as the function normal_image_sens_fpr_calculator() now returns the specificity instead of FPR.
            normal_stats[conf_score_thres] = {
            'sensitivity': sens_normal,
            'fpr': fpr_normal,
            'youdens_index': sens_normal+fpr_normal-1
            }     
    
    if len(normal_img_ids)>0:  
        # pprint.pprint(normal_stats)
        normal_plot_dict = {}
        normal_sens_list = []
        normal_fpr_list = []
        normal_youdens_list = []
        for conf_score in normal_stats.keys():
            normal_sens_list.append(normal_stats[conf_score]['sensitivity'])
            normal_fpr_list.append(normal_stats[conf_score]['fpr'])
            normal_youdens_list.append(normal_stats[conf_score]['youdens_index'])
        normal_plot_dict['derived_normal'] = {
            'sens_list':normal_sens_list,
            'fpr_list':normal_fpr_list,
            'max_youdens_index': max(normal_youdens_list),
            'optimal_threshold': ""
        }
        # normal_fpr_list.reverse()
        # normal_sens_list.reverse()
        eval_thresholds = []
        normal_interp_sens = []
        # eval_thresholds = [x for x in og_eval_thresholds if x >= min(normal_fpr_list) and x <= max(normal_fpr_list)]
        # normal_interp_sens = np.interp(eval_thresholds, np.array(normal_fpr_list)[::-1], np.array(normal_sens_list)[::-1])
        plot_per_cat_froc_curve("Derived_normal", normal_fpr_list, normal_sens_list, save_dir, iou_thres, eval_thresholds, normal_interp_sens, PLOT_EVAL_THRES, INSERT_EVAL_THRES_MARKERS, PLOT_SOTA_MARKERS, sota_points[sota_no_finding_id]['fpr_list'], sota_points[sota_no_finding_id]['sens_list'], EXTRAPOLATE_PLOT, "ROC")
    
        
    # print(f'{cat_stats}')
    per_cat_plot_dict = {}
    for cat_id in gt_cat_ids:
        sens_list = []
        fpr_list = []
        for conf_score in cat_stats.keys():
            sens_list.append(cat_stats[conf_score][cat_id]['sensitivity'])
            fpr_list.append(cat_stats[conf_score][cat_id]['fpr'])
        per_cat_plot_dict[cat_id] = {
            'sens_list':sens_list,
            'fpr_list':fpr_list
        }
        eval_thresholds = [x for x in og_eval_thresholds if x >= min(fpr_list) and x <= max(fpr_list)]
        interp_sens = np.interp(eval_thresholds, np.array(fpr_list)[::-1], np.array(sens_list)[::-1])
        # print(gt_cat_id_to_labels[cat_id], interp_sens) 
        
        plot_per_cat_froc_curve(gt_cat_id_to_labels[cat_id], fpr_list, sens_list, save_dir, iou_thres, eval_thresholds, interp_sens, PLOT_EVAL_THRES, INSERT_EVAL_THRES_MARKERS, PLOT_SOTA_MARKERS, sota_points[cat_id]['fpr_list'], sota_points[cat_id]['sens_list'], EXTRAPOLATE_PLOT)
        
    plot_info = {}
    plot_info.update(per_cat_plot_dict)
    if len(normal_img_ids)>0:
        plot_info.update(normal_plot_dict)
    plot_info['mappings'] = gt_cat_id_to_labels
    
    outfile_path = osp.join(save_dir,str(iou_thres).replace('.','_'), "plot_info.json")
    with open(outfile_path, 'w') as outfile:
        json.dump(plot_info, outfile, indent=4)
    print(f'Plot info file at: {outfile_path}')
    
    return

if __name__ == '__main__':
    # argparse
    parser = argparse.ArgumentParser(description="Description of your script")

    # Boolean flags
    parser.add_argument('--IGNORE_STRICT_CHECK', type=bool, default=False, help='Description of IGNORE_STRICT_CHECK')
    parser.add_argument('--PLOT_EVAL_THRES', type=bool, default=True, help='Description of PLOT_EVAL_THRES')
    parser.add_argument('--PLOT_SOTA_MARKERS', type=bool, default=True, help='Description of PLOT_SOTA_MARKERS')
    parser.add_argument('--INSERT_EVAL_THRES_MARKERS', type=bool, default=True, help='Description of INSERT_EVAL_THRES_MARKERS')
    parser.add_argument('--EXTRAPOLATE_PLOT', type=bool, default=False, help='Description of EXTRAPOLATE_PLOT')

    # Integer arguments
    parser.add_argument('--num_sample_points', type=int, default=100, help='Description of num_sample_points')

    # Float arguments
    parser.add_argument('--iou_thres', type=float, default=0.4, help='Description of iou_thres')

    parser.add_argument('--sota_no_finding_id', type=float, default='', help='Description of sota_no_finding_id')
    
    # Tuple arguments
    parser.add_argument('--og_eval_thresholds', nargs='+', type=float, default=(0.25, 0.5, 1, 2, 4), help='Description of og_eval_thresholds')

    # String arguments
    parser.add_argument('--gt_file_path', type=str, default='', help='Description of gt_file_path')
    parser.add_argument('--pred_file_path', type=str, default='', help='Description of pred_file_path')
    parser.add_argument('--save_dir', type=str, default='', help='Description of save_dir')

    args = parser.parse_args()
    
    
    IGNORE_STRICT_CHECK = args.IGNORE_STRICT_CHECK #True
    PLOT_EVAL_THRES = args.PLOT_EVAL_THRES #True
    PLOT_SOTA_MARKERS = args.PLOT_SOTA_MARKERS #True
    INSERT_EVAL_THRES_MARKERS = args.INSERT_EVAL_THRES_MARKERS #True
    EXTRAPOLATE_PLOT = args.EXTRAPOLATE_PLOT #False
    
    # print(f'{IGNORE_STRICT_CHECK}, {PLOT_EVAL_THRES}, {PLOT_SOTA_MARKERS}, {INSERT_EVAL_THRES_MARKERS}, {EXTRAPOLATE_PLOT}')
    
    num_sample_points = args.num_sample_points #100 # Number of confidence thresholds between 0 and 1.
    
    iou_thres = args.iou_thres #0.4
    og_eval_thresholds=tuple(args.og_eval_thresholds) #(0.25, 0.5, 1, 2, 4)
    
    # To have the sota reference on the graph, you will have to manually enter the points with the correct category IDs based on the GT file. The ordering of the sens and fpr list must be same as the order of the eval_thresholds.
    sota_points = {
        0:{ # Aortic enlargement
            'sens_list':[0.838, 0.882, 0.905, 0.905, 0.909],
            'fpr_list':[0.25, 0.5, 1, 2, 4]               
        },
        1:{ # Atelectasis
            'sens_list':[0.642, 0.698, 0.772, 0.772, 0.772],
            'fpr_list':[0.25, 0.5, 1, 2, 4]               
        },
        2:{ # Calcification
            'sens_list':[0.598, 0.685, 0.774, 0.802, 0.802],
            'fpr_list':[0.25, 0.5, 1, 2, 4]               
        },
        3:{ # Cardiomegaly
            'sens_list':[0.965, 0.965, 0.968, 0.968, 0.968],
            'fpr_list':[0.25, 0.5, 1, 2, 4]               
        },
        4:{ # Consolidation
            'sens_list':[0.841, 0.898, 0.937, 0.937, 0.937],
            'fpr_list':[0.25, 0.5, 1, 2, 4]               
        },
        5:{ # ILD
            'sens_list':[0.664, 0.782, 0.858, 0.923, 0.937],
            'fpr_list':[0.25, 0.5, 1, 2, 4]               
        },
        6:{ # Infiltration
            'sens_list':[0.801, 0.861, 0.911, 0.911, 0.911],
            'fpr_list':[0.25, 0.5, 1, 2, 4]               
        },
        7:{ # Lung Opacity
            'sens_list':[0.617, 0.756, 0.842, 0.895, 0.906],
            'fpr_list':[0.25, 0.5, 1, 2, 4]               
        },
        8:{ # Nodule/Mass
            'sens_list':[0.579, 0.663, 0.735, 0.769, 0.773],
            'fpr_list':[0.25, 0.5, 1, 2, 4]               
        },
        9:{ # Other Lesion
            'sens_list':[0.265, 0.311, 0.372, 0.484, 0.551],
            'fpr_list':[0.25, 0.5, 1, 2, 4]               
        },
        10:{ # Pleural Effusion
            'sens_list':[0.898, 0.927, 0.934, 0.942, 0.942],
            'fpr_list':[0.25, 0.5, 1, 2, 4]               
        },
        11:{ # Pleural Thickening
            'sens_list':[0.494, 0.608, 0.714, 0.797, 0.850],
            'fpr_list':[0.25, 0.5, 1, 2, 4]               
        },
        12:{ # Pneumothorax
            'sens_list':[0.639, 0.680, 0.680, 0.680, 0.680],
            'fpr_list':[0.25, 0.5, 1, 2, 4]               
        },
        13:{ # Pulmonary Fibrosis
            'sens_list':[0.568, 0.627, 0.707, 0.775, 0.796],
            'fpr_list':[0.25, 0.5, 1, 2, 4]               
        },
        14:{ # No findings
            'sens_list':[0.92],
            'fpr_list':[0.91] # This is not FPR, this is specificity from the VinDr paper.              
            # 'sens_list':[0.921, 0.921, 0.921, 0.921, 0.921],
            # 'fpr_list':[0.25, 0.5, 1, 2, 4]               
        },
    }
    sota_no_finding_id = args.sota_no_finding_id #14
    
    gt_file_path = args.gt_file_path #'/scratch/ssenth21/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0/annotations/VinDrCXR_Kaggle_14Diseases_TEST.json'
    pred_file_path = args.pred_file_path #'/scratch/ssenth21/EXPERIMENTS/bench_boost_localizers/experiments/exp_1/ensemble_vindr/ensemble_1/Swin_S-Swin_L-Intern_B-R50-FasterRCNN.json'
    save_dir = args.save_dir #'/scratch/ssenth21/EXPERIMENTS/bench_boost_localizers/experiments/exp_1/ensemble_vindr/ensemble_1/froc_curves/'
    
    # gt_file_path = '/scratch/ssenth21/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0/annotations/VinDrCXR_Kaggle_14Diseases_TEST.json'
    # pred_file_path = '/scratch/ssenth21/EXPERIMENTS/bench_boost_localizers/misc/vindr_kaggle_1stplace/submission_disease_only.json'
    # save_dir = '/scratch/ssenth21/EXPERIMENTS/bench_boost_localizers/misc/vindr_kaggle_1stplace/froc_curves/improved_froc/only_diseases'
    
    # gt_file_path = '/scratch/ssenth21/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0/annotations/kaggle_sota_tester_annotations_test_full.json'
    # pred_file_path = '/scratch/ssenth21/EXPERIMENTS/bench_boost_localizers/misc/vindr_kaggle_1stplace/class_14_cleaned_submission.json'
    # save_dir = '/scratch/ssenth21/EXPERIMENTS/bench_boost_localizers/misc/vindr_kaggle_1stplace/froc_curves/improved_froc/class_14_cleaned_submission'
    
    # gt_file_path = '/scratch/ssenth21/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0/annotations/kaggle_sota_tester_annotations_test_full.json'
    # pred_file_path = '/scratch/ssenth21/EXPERIMENTS/bench_boost_localizers/misc/vindr_kaggle_1stplace/submission.json'
    # save_dir = '/scratch/ssenth21/EXPERIMENTS/bench_boost_localizers/misc/vindr_kaggle_1stplace/froc_curves/improved_froc/'
    
    # gt_file_path = '/scratch/ssenth21/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0/annotations/VinDrCXR_Kaggle_14Diseases_TEST.json'
    # pred_file_path = '/scratch/ssenth21/EXPERIMENTS/bench_boost_localizers/experiments/exp_1/ensemble_vindr/codetesting/test.json'
    # save_dir = '/scratch/ssenth21/EXPERIMENTS/bench_boost_localizers/experiments/exp_1/ensemble_vindr/codetesting/improved_froc/'
    
    # gt_file_path = '/scratch/ssenth21/coco-froc-analysis/tests/improved_froc_testsuite/sample_coco_json/gt.json'
    # pred_file_path = '/scratch/ssenth21/coco-froc-analysis/tests/improved_froc_testsuite/sample_coco_json/preds.json'
    # save_dir = '/scratch/ssenth21/coco-froc-analysis/tests/improved_froc_testsuite/froc_curves/'
    main()