import os
from PIL import Image

def calculate_height_with_aspect_ratio(image, desired_width):
    aspect_ratio = image.width / image.height
    return int(desired_width / aspect_ratio)

def create_combined_image(folder_path, output_path, image_resolution, final_resolution):
    image_files = [file for file in os.listdir(folder_path) if file.endswith('.png')]
    image_files = sorted(image_files)

    # Calculate the number of images that can fit in a row based on image resolution and final resolution
    images_per_row = final_resolution[0] // image_resolution[0]

    # Calculate the number of rows required to accommodate all the images
    num_rows = (len(image_files) - 1) // images_per_row + 1

    # Calculate the actual height for each row based on the available width
    row_height = final_resolution[1] // num_rows

    # Create a blank canvas for the combined image
    combined_image = Image.new('RGB', final_resolution)

    # Paste each image onto the combined image
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(folder_path, image_file)
        image = Image.open(image_path)

        # Calculate the appropriate height while maintaining the aspect ratio
        image_height = calculate_height_with_aspect_ratio(image, image_resolution[0])
        image = image.resize((image_resolution[0], image_height))

        row_index = i // images_per_row
        col_index = i % images_per_row
        x = col_index * image_resolution[0]
        y = row_index * row_height
        combined_image.paste(image, (x, y))

    # Save the combined image
    combined_image.save(output_path)


# Example usage
""" folder_path = '/scratch/ssenth21/coco-froc-analysis/data/faster_rcnn_vindr/only_diseased/iou_0_1'
output_path = '/scratch/ssenth21/coco-froc-analysis/data/faster_rcnn_vindr/only_diseased/merged_plots_iou_0_1.png'
image_resolution = (600, 600)  # Resolution of each individual image
final_resolution = (2410, 2410)  # Resolution of the combined image """

""" create_combined_image(folder_path, output_path, image_resolution, final_resolution) """
