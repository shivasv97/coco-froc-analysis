from .froc_curve import generate_froc_curve
from copy import deepcopy

def generate_bootstrap_curves(gt_ann, pr_ann, n_bootstrap_samples,
              use_iou, iou_thres, n_sample_points,
              plot_title, plot_output_path):
    with open(gt_ann, 'r+') as fp:
        GT_ANN = json.load(fp)

    with open(pr_ann, 'r+') as fp:
        PRED_ANN = json.load(fp)

    n_images = len(GT_ANN['images'])

    plt.figure(figsize=(15, 15))

    collected_frocs = {"lls": {}, "nlls": {}}

    for _ in tqdm.tqdm(range(n_bootstrap_samples)):
        selected_images = random.choices(GT_ANN['images'], k=n_images)
        bootstrap_gt = deepcopy(GT_ANN)

        del bootstrap_gt['images']

        bootstrap_gt['images'] = selected_images

        gt_annotations = bootstrap_gt['annotations']
        
        del bootstrap_gt['annotations']

        bootstrap_gt['annotations'] = []
        for _gt_ann_ in gt_annotations:
            img_id = _gt_ann_['image_id']
            for selected_image in selected_images:
                if selected_image['id'] == img_id:
                    bootstrap_gt['annotations'].append(_gt_ann_)

        with open('/tmp/tmp_bootstrap_gt.json', 'w') as fp:
            json.dump(bootstrap_gt, fp)

        with open('/tmp/tmp_bootstrap_pred.json', 'w') as fp:
            json.dump(PRED_ANN, fp)

        tmp_gt_ann = '/tmp/tmp_bootstrap_gt.json'
        tmp_pred_ann = '/tmp/tmp_bootstrap_pred.json'

        lls, nlls = create_froc_curve(tmp_gt_ann, tmp_pred_ann, use_iou, iou_thres, n_sample_points)

        for cat_id in lls:
            plt.semilogx(lls[cat_id],
                         nlls[cat_id], '-', alpha=.05)
            collected_frocs['lls'].get(cat_id, []).append(lls[cat_id])
            collected_frocs['nlls'].get(cat_id, []).append(nlls[cat_id])

        collected_frocs.append((lls, nlls))

    mean_froc_lls = {}
    mean_froc_nlls = {}

    for cat_id in collected_frocs['lls']:
        mean_froc_lls[cat_id] = np.mean(np.array(collected_frocs['lls'][cat_id]).reshape(
            args.n_samples, args.n_sample_points), axis=0)
        mean_froc_nlls[cat_id] = np.mean(np.array(collected_frocs['nlls'][cat_id]).reshape(
            args.n_samples, args.n_sample_points), axis=0)

    mean_froc_curve = {}
    for cat_id in collected_frocs['lls']:
        mean_froc_curve[cat_id] = np.stack((mean_froc_nlls[cat_id], mean_froc_lls[cat_id]), axis=-1)

        plt.semilogx(mean_froc_curve[cat_id][:, 0],
                     mean_froc_curve[cat_id][:, 1], 'x-')

    plt.xlabel('FP/image')
    plt.ylabel('Sensitivity')

    os.remove('/tmp/tmp_bootstrap_gt.json')
    os.remove('/tmp/tmp_bootstrap_pred.json')

    plt.title(plot_title)

    plt.savefig(plot_output_path, dpi=100)