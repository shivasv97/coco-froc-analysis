'''This script is for sbatch-based and bash-based generation of curves.

Commandline argument capability is incorporated for easy manipulation.

'''

from coco_froc_analysis.count import generate_bootstrap_count_curves
from coco_froc_analysis.count import generate_count_curve
from coco_froc_analysis.froc import generate_bootstrap_froc_curves
from coco_froc_analysis.froc import generate_froc_curve

import plots_merger
import os
import argparse

def create_missing_dirs(path):
    os.makedirs(path, exist_ok=True)


parser = argparse.ArgumentParser(description='Plotting FROC curves using COCO-formatted JSON files for predictions and ground truths.')

parser.add_argument('-g', '--gt', type=str, help='Enter the absolute path to the ground file that is coco JSON formatted.')
parser.add_argument('-p', '--pred', type=str, help='Enter the absolute path to the predictions file that is coco JSON formatted.')
parser.add_argument('-d', '--dir', type=str, help='Enter the absolute path to the dir where graphs will be stored.')
parser.add_argument('--iou_list', nargs='+', type=float, help='Enter the iou thresholds for which FROC curves are to be plotted, separated by a single space.')
parser.add_argument('--sample_points', default=100, type=int, help='Enter the number of sample points to be present on the FROC curve in each plot. Default: 100')
parser.add_argument('--iou_upper_bound', type=float, default=1, help='Enter the upper value of the iou threshold. Default: 1')
args = parser.parse_args()

gt_ann = args.gt #'/scratch/ssenth21/InternImage/detection/data/coco/annotations/instances_val2017.json'
pr_ann = args.pred #'/data/jliang12/ssenth21/bench_boost_localizers/experiments/exp_0/DINO/Swin/Evaluate_pretrained_published/run02/coco_val_preds.bbox.json'
root_dir = args.dir #'/data/jliang12/ssenth21/bench_boost_localizers/experiments/exp_0/DINO/Swin/Evaluate_pretrained_published/run02/froc_curves'

for iou_thres_val in args.iou_list:
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
                iou_thres_upper_bound=args.iou_upper_bound,
                n_sample_points=args.sample_points,
                plot_title='FROC' ,
                plot_output_path=plot_output_path,
            )

    plots_merger.create_combined_image(plot_dir_path, merged_plot_img_path, image_resolution, final_resolution)