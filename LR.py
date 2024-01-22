import cv2
import os

input_directory = './dataset/tandt_db/tandt/train/images/'
output_directory = './dataset/tandt_db/tandt/train_LR/input/images/'


if not os.path.exists(output_directory):
    os.makedirs(output_directory)
downsamp_ratio = 2
for filename in os.listdir(input_directory):
    input_path = os.path.join(input_directory, filename)
    output_path = os.path.join(output_directory, filename)

    img = cv2.imread(input_path)
    height, width = img.shape[:2]
    resized_img = cv2.resize(img, (width // downsamp_ratio, height // downsamp_ratio), interpolation=cv2.INTER_AREA)
    cv2.imwrite(output_path, resized_img)
