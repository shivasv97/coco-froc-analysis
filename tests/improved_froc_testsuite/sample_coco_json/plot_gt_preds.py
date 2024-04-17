import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_bboxes(inp_img_path, gt_annotations, pred_annotations, save_filename):
    fig, ax = plt.subplots()

    # Set axis limits based on the maximum bounding box coordinates
    max_x = 0
    max_y = 0
    # Check if gt_annotations is not empty
    if gt_annotations:
        max_x = max(max(gt['bbox'][0] for gt in gt_annotations), max_x)
        max_y = max(max(gt['bbox'][1] for gt in gt_annotations), max_y)

    # Check if pred_annotations is not empty
    if pred_annotations:
        max_x = max(max(pred['bbox'][0] for pred in pred_annotations), max_x)
        max_y = max(max(pred['bbox'][1] for pred in pred_annotations), max_y)
    ax.set_xlim(0, max_x + 100)  # Add padding for better visualization
    ax.set_ylim(0, max_y + 100)  # Add padding for better visualization

    # Plot ground truth bounding boxes
    for gt in gt_annotations:
        if gt is not None:
            bbox = gt['bbox']
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
            ax.text(bbox[0], bbox[1], f"GT: {gt['category_id']}", color='g', fontsize=8)
    #plt.show()
    # Plot predicted bounding boxes
    for pred in pred_annotations:
        if pred is not None:
            bbox = pred['bbox']
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(bbox[0], bbox[1], f"Pred: {pred['category_id']}", color='r', fontsize=8)

    # plt.show()
    plt.savefig(save_filename)
    plt.close()


# Load ground truth and predicted annotations
with open('./gt.json', 'r') as f:
    gt_data = json.load(f)

with open('./preds.json', 'r') as f:
    pred_data = json.load(f)


# Save filename for the plot (replace with desired filename)

inp_img_path = '/scratch/jliang12/data/tbx11k/tbx11k/TBX11K/imgs/health/h0001.png'

# Plot bounding boxes and save the plot
for image in gt_data['images']:
    
    save_filename = f'./bbox_plot_{image["id"]}.png'
    gt_annotations = [x for x in gt_data['annotations'] if x['image_id']==image['id']]
    _pred_data = [x for x in pred_data if x['image_id']==image['id']]
    # print(gt_annotations)
    # print(pred_data)
    plot_bboxes(inp_img_path, gt_annotations, _pred_data, save_filename)
