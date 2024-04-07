from __future__ import annotations
import json

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import math

from ..utils import build_gt_id2annotations
from ..utils import build_pr_id2annotations
from ..utils import COLORS
from ..utils import load_json_from_file
from ..utils import transform_gt_into_pr
from ..utils import update_scores
from .froc_stats import init_stats
from .froc_stats import update_stats


def froc_point(gt_ann, pr_ann, score_thres, use_iou, iou_thres, iou_thres_upper_bound):
    #print(iou_thres)
    gt = load_json_from_file(gt_ann)
    pr = load_json_from_file(pr_ann)
    #print(score_thres)
    pr = update_scores(pr, score_thres)
    # if score_thres > 0.5 and score_thres < 0.52:
    #     for item in pr:
    #         if item['score'] < score_thres:
    #             print("Something fishy here.")

    categories = gt['categories']

    stats = init_stats(gt, categories)

    gt_id_to_annotation = build_gt_id2annotations(gt)
    pr_id_to_annotation = build_pr_id2annotations(pr)
    #print(len(gt_id_to_annotation), len(pr_id_to_annotation))

    stats = update_stats(
        stats, gt_id_to_annotation, pr_id_to_annotation,
        categories, use_iou, iou_thres,iou_thres_upper_bound
    )
    #print(stats)

    return stats


def calc_scores(stats, lls_accuracy, nlls_per_image):
    for category_id in stats:
        if stats[category_id]['n_lesions'] != 0:
            if lls_accuracy.get(category_id, None):
                lls_accuracy[category_id].append(
                    stats[category_id]['LL'] /
                    stats[category_id]['n_lesions'],
                )
            else:
                lls_accuracy[category_id] = []
                lls_accuracy[category_id].append(
                    stats[category_id]['LL'] /
                    stats[category_id]['n_lesions'],
                )

        if nlls_per_image.get(category_id, None):
            nlls_per_image[category_id].append(
                stats[category_id]['NL'] /
                stats[category_id]['n_images'],
            )
        else:
            nlls_per_image[category_id] = []
            nlls_per_image[category_id].append(
                stats[category_id]['NL'] /
                stats[category_id]['n_images'],
            )

    return lls_accuracy, nlls_per_image


def generate_froc_curve(
    gt_ann,
    pr_ann,
    use_iou=False,
    iou_thres=0.5,
    iou_thres_upper_bound=1.0,
    n_sample_points=50,
    plot_title='FROC curve',
    plot_output_path='froc.png',
    test_ann=None,
    bounds=None,
):

    lls_accuracy = {}
    nlls_per_image = {}

    with open(f'stat_log_{iou_thres}:{iou_thres_upper_bound}.json', 'w') as f: 
        dict_to_dump = {}
        # for score_thres in tqdm(
        #         [0.8],
        # ):
        for score_thres in tqdm(
                np.linspace(0.0, 1.0, n_sample_points, endpoint=True),
        ):
            #print('IoU: ',iou_thres)
            #stats = froc_point(gt_ann, pr_ann, iou_thres, use_iou, score_thres)
            #print("Linspace: ", score_thres)
            stats = froc_point(gt_ann, pr_ann, score_thres, use_iou, iou_thres, iou_thres_upper_bound)
            lls_accuracy, nlls_per_image = calc_scores(
                stats, lls_accuracy,
                nlls_per_image,
            )
            dict_to_dump[score_thres] = [stats, lls_accuracy, nlls_per_image]
        json.dump(dict_to_dump, f)
        
        # create FROC curve for all classes combined
        list_length = len(next(iter(lls_accuracy.values())))
        mean_lls = [0] * list_length
        mean_nlls = [0] * list_length
        for values in lls_accuracy.values():
            if len(values) != list_length:
                raise ValueError("All lists in the dictionary must have equal length")
            mean_lls = [avg + val for avg, val in zip(mean_lls, values)]
        mean_lls = [avg / len(lls_accuracy) for avg in mean_lls]
        
        for values in nlls_per_image.values():
            if len(values) != list_length:
                raise ValueError("All lists in the dictionary must have equal length")
            mean_nlls = [avg + val for avg, val in zip(mean_nlls, values)]
        mean_nlls = [avg / len(nlls_per_image) for avg in mean_nlls]
            
        x_axis_end_point = max(mean_nlls)
        x_axis_end_point_index = mean_nlls.index(x_axis_end_point)
        lls_end_point_val = mean_lls[x_axis_end_point_index]
        final_point_x = max(1.0, math.ceil(x_axis_end_point))
        final_point_y = lls_end_point_val
        mean_lls.insert(0,final_point_y)
        mean_nlls.insert(0,final_point_x)
        
        if plot_title:
            fig, ax = plt.subplots(figsize=[27, 18])
            ax.plot(
                #np.concatenate([nlls, x_extrapolate]),
                #np.concatenate([lls, y_extrapolate]),
                mean_nlls,#x_smooth,#
                mean_lls,#y_smooth,#
                'o-',
                label='Mean FROC for all classes', linewidth=2, markersize=5
            )
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

            ax.legend(
                loc='lower left', bbox_to_anchor=(.1, .1),
                fancybox=True, shadow=True, ncol=1, fontsize=30,
            )
            

            ax.set_title(plot_title, fontdict={'fontsize': 35})
            ax.grid(True)
            ax.set_ylabel('Sensitivity', fontdict={'fontsize': 30})
            ax.set_xlabel(f'FP / image ', fontdict={'fontsize': 30})
            #nlls_max:{max(nlls)}
            ax.tick_params(axis='both', which='major', labelsize=30)
            """ ins.tick_params(axis='both', which='major', labelsize=20) """

            if bounds is not None:
                x_min, x_max, y_min, y_max = bounds
                """ ax.set_ylim([0, 1])
                ax.set_xlim([0, 1]) """
                ax.set_ylim([y_min, y_max])
                ax.set_xlim([x_min, x_max])
            else:
                """ ax.set_ylim(bottom=0, top=1)
                ax.set_xlim(left=0, right=1) """
                ax.set_ylim(bottom=-0.1, top=1.1)
                ax.set_xlim(left=-0.1, right=max(1.0, math.ceil(max(mean_nlls))))
            fig.tight_layout()
            plot_output_name = f'{plot_output_path}_mean_FROC.png'
            fig.savefig(fname=plot_output_name, dpi=150)
            plt.close(fig)
        
        for category_id in lls_accuracy:
            """ if category_id == 8:
                print(lls_accuracy[category_id], nlls_per_image[category_id]) """
            if plot_title:
                fig, ax = plt.subplots(figsize=[27, 18])
                """ ins = ax.inset_axes([0.55, 0.05, 0.45, 0.4])
                ins.set_xticks(
                    [0.1, 1.0, 2.0], [
                        0.1, 1.0, 2.0
                    ], fontsize=30,
                ) #, 3.0, 4.0   , 3.0, 4.0,

                if bounds is not None:
                    _, x_max, _, y_max = bounds
                    ins.set_xlim([0, 1])
                else:
                    ins.set_xlim([0, 1]) """

            lls = lls_accuracy[category_id]
            nlls = nlls_per_image[category_id]
            x_axis_end_point = max(nlls)
            x_axis_end_point_index = nlls.index(x_axis_end_point)
            lls_end_point_val = lls[x_axis_end_point_index]
            final_point_x = math.ceil(x_axis_end_point)
            final_point_y = lls_end_point_val
            lls.insert(0,final_point_y)
            nlls.insert(0,final_point_x)
            # print(f"lls: {x_axis_end_point_index} {len(lls)} \n nlls: {x_axis_end_point_index} {len(nlls)}")
            # Polynomial interpolation
            """ degree = 2  # Degree of the polynomial
            coefficients = np.polyfit(nlls, lls, degree)
            polynomial = np.poly1d(coefficients)
            x_smooth = np.linspace(nlls[0], lls[-1], 100)
            y_smooth = polynomial(x_smooth)
            
            degree = 1  # Degree of the polynomial
            coefficients = np.polyfit(nlls, lls, degree)
            polynomial = np.poly1d(coefficients)
            x_extrapolate = np.linspace(nlls[-1], plt.xlim()[1], 100)
            y_extrapolate = polynomial(x_extrapolate) """
            if plot_title:
                ax.plot(
                    #np.concatenate([nlls, x_extrapolate]),
                    #np.concatenate([lls, y_extrapolate]),
                    nlls,#x_smooth,#
                    lls,#y_smooth,#
                    'x-',
                    label='AI ' + stats[category_id]['name'], linewidth=2, markersize=5
                )
                """ ins.plot(
                    nlls,
                    lls,
                    'x--',
                    label='AI ' + stats[category_id]['name'],
                ) """

                if test_ann is not None:
                    print("here")
                    for t_ann, c in zip(test_ann, COLORS):
                        t_ann, label = t_ann
                        t_pr = transform_gt_into_pr(t_ann, gt_ann)
                        stats = froc_point(gt_ann, t_pr, .5, use_iou, iou_thres)
                        _lls_accuracy, _nlls_per_image = calc_scores(stats, {}, {})
                        if plot_title:
                            ax.plot(
                                _nlls_per_image[category_id][0],
                                _lls_accuracy[category_id][0],
                                'D',
                                markersize=15,
                                markeredgewidth=3,
                                label=label +
                                f' (FP/image = {np.round(_nlls_per_image[category_id][0], 2)})',
                                c=c,
                            )
                            ins.plot(
                                _nlls_per_image[category_id][0],
                                _lls_accuracy[category_id][0],
                                'D',
                                markersize=12,
                                markeredgewidth=2,
                                label=label +
                                f' (FP/image = {np.round(_nlls_per_image[category_id][0], 2)})',
                                c=c,
                            )
                            ax.hlines(
                                y=_lls_accuracy[category_id][0],
                                xmin=np.min(nlls),
                                xmax=np.max(nlls),
                                linestyles='dashed',
                                colors=c,
                            )
                            ins.hlines(
                                y=_lls_accuracy[category_id][0],
                                xmin=np.min(nlls),
                                xmax=np.max(nlls),
                                linestyles='dashed',
                                colors=c,
                            )
                            ax.text(
                                x=_nlls_per_image[category_id][0], y=_lls_accuracy[category_id][0],
                                s=f' FP/image = {np.round(_nlls_per_image[category_id][0], 2)}',
                                fontdict={'fontsize': 20, 'fontweight': 'bold'},
                            )

            if plot_title:
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

                ax.legend(
                    loc='lower left', bbox_to_anchor=(.1, .1),
                    fancybox=True, shadow=True, ncol=1, fontsize=30,
                )
                

                ax.set_title(plot_title, fontdict={'fontsize': 35})
                ax.grid(True)
                ax.set_ylabel('Sensitivity', fontdict={'fontsize': 30})
                ax.set_xlabel(f'FP / image ', fontdict={'fontsize': 30})
                #nlls_max:{max(nlls)}
                ax.tick_params(axis='both', which='major', labelsize=30)
                """ ins.tick_params(axis='both', which='major', labelsize=20) """

                if bounds is not None:
                    x_min, x_max, y_min, y_max = bounds
                    """ ax.set_ylim([0, 1])
                    ax.set_xlim([0, 1]) """
                    ax.set_ylim([y_min, y_max])
                    ax.set_xlim([x_min, x_max])
                else:
                    """ ax.set_ylim(bottom=0, top=1)
                    ax.set_xlim(left=0, right=1) """
                    ax.set_ylim(bottom=-0.1, top=1.1)
                    ax.set_xlim(left=-0.1, right=max(1.0, math.ceil(max(nlls))))
                    # print(f'cat {category_id}: {max(1.0, math.ceil(max(nlls)))}, {math.ceil(max(nlls))}, {max(nlls)}, {nlls}')
                fig.tight_layout()
                plot_output_name = f'{plot_output_path}_froc_category_{category_id}.png'
                fig.savefig(fname=plot_output_name, dpi=150)
                plt.close(fig)
            else:
                return lls_accuracy, nlls_per_image
