'''Rough idea
1) CSV reader to get the ground truths. TODO: organize the GT in a format suitable for processing.
2) Fetch predictions from the MMDET model for the unique images in the CSV entries. TODO: organize these predictions in a suitable format.
3) For each image:
    1) For each class:
        1) Fetch the IoU of the prediction and GT bboxes. 
            1) For each of the ground truth box in that particular class, arrange the predicted bboxes in highest to lowest IoU value.
            2) This implies that each GT bbox will have multiple predictions and IoU rank.
            3) From these multiple entries of GT:pred IoUs for a single class, merge them all together and take the pairs that have the IoU greater than the threshold.

'''

from coco_froc_analysis.count import generate_bootstrap_count_curves
from coco_froc_analysis.count import generate_count_curve
from coco_froc_analysis.froc import generate_bootstrap_froc_curves
from coco_froc_analysis.froc import generate_froc_curve

import plots_merger
import os

def create_missing_dirs(path):
    os.makedirs(path, exist_ok=True)


gt_ann = '/scratch/jliang12/data/ASUMayoDBsplit_jpg/annotations/val_detc_annotations_coco.json'
pr_ann = '/data/jliang12/shared/ssenth21/toma/org-dino-pipeline/detection/asumayopolyp/ASUMayoDBPolyp_detection_variant_4_run_01_eval/cocoeval_pred.json'
root_dir = '/scratch/ssenth21/coco-froc-analysis/data/swin_dino_asumayo_polyp/validation_set'

for iou_thres_val in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9]:
    iou_folder_name = str(iou_thres_val).replace('.', '_')
    plot_dir_path = f'{root_dir}/iou_{iou_folder_name}'
    create_missing_dirs(plot_dir_path)
    plot_output_path = plot_dir_path + '/froc_'
    merged_plot_img_path = f'{root_dir}/merged_plots_iou_{iou_folder_name}.png'
    image_resolution = (600, 600)  # Resolution of each individual image
    final_resolution = (2410, 2410)  # Resolution of the combined image

    # For single FROC curve

    generate_froc_curve(
                gt_ann=gt_ann,
                pr_ann=pr_ann,
                use_iou=True,
                iou_thres=iou_thres_val,
                n_sample_points=50,
                plot_title='FROC' ,
                plot_output_path=plot_output_path,
            )

    plots_merger.create_combined_image(plot_dir_path, merged_plot_img_path, image_resolution, final_resolution)