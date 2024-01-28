import os
import cv2

input_directory = '/cluster/work/cvl/jiezcao/jiameng/3D-Gaussian/tandt_db/tandt/train/images'
output_directory_base = '/cluster/work/cvl/jiezcao/jiameng/3D-Gaussian/tandt_db/tandt/train/images_'

downsamp_ratios = [2, 4, 8]

# Loop through each downsampling ratio
for downsamp_ratio in downsamp_ratios:
    output_directory = os.path.join(output_directory_base + str(downsamp_ratio))

    # Create the output directory if it does not exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Loop through each file in the input directory
    for filename in os.listdir(input_directory):
        input_path = os.path.join(input_directory, filename)
        output_path = os.path.join(output_directory, filename)

        img = cv2.imread(input_path)
        height, width = img.shape[:2]
        resized_img = cv2.resize(img, (width // downsamp_ratio, height // downsamp_ratio), interpolation=cv2.INTER_AREA)
        cv2.imwrite(output_path, resized_img)
