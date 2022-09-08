import os
import torch
from dataloader import dataloader
# from utils.custom_parser import *
import filetype
import cv2
import numpy as np
import sys

data_dir = sys.argv[1]
target_data_dir = sys.argv[2]

def get_file_type(data_dir: str, file_type_list=("crawling", "drawing", "studio")) -> str:
    for file_type in file_type_list:
        if file_type in data_dir.split("/"):
            return file_type
    return "drawing"  # if not defined file type in whole directory
    # assert "unknown file_type. please confirm directory : ", data_dir

train_dataset = dataloader.DatasetWithBgSegAug(data_dir=None,
                                               background_aug_dir=None,
                                               edge_detection_model=torch.load("./utils/model_edge_detection.pt"),
                                               segmentation_model=torch.load("./utils/u2net.pt"),
                                               )

total_file_num = 0
file_num_counter = 0

for root, dirs, files in os.walk(data_dir):
    for file in files:
        original_file_dir = os.path.join(root, file)
        if filetype.is_image(original_file_dir):
            total_file_num += 1

print("total file number : ", total_file_num)

for root, dirs, files in os.walk(data_dir):
    for file in files:
        original_file_dir = os.path.join(root, file)
        if filetype.is_image(original_file_dir):
            target_root = root.replace(data_dir, target_data_dir)
            os.makedirs(target_root, exist_ok=True)
            file_type = get_file_type(root)
            image = cv2.imread(original_file_dir)

            file_num_counter += 1
            if (file_num_counter % 1000) == 999:
                print("file processing : {} / {}".format(file_num_counter, total_file_num))

            if file_type == "drawing":
                image = train_dataset.edge_detection_from_numpy(image)
                image = train_dataset.resize_with_pad_for_square(image, tgt_size=1014)
                image = np.pad(image, ((5, 5),
                                       (5, 5),
                                       (0, 0)),
                               constant_values=255)
                cv2.imwrite(os.path.join(target_root, file), image)
            elif file_type == "studio":
                image = train_dataset.resize_with_pad_for_square(image)
                image, _ = train_dataset.segmentation_module(image)

                image = train_dataset.edge_detection_from_numpy(image)  # downsizing with same ratio, largest value is 512
                image = train_dataset.resize_with_pad_for_square(image, tgt_size=1014)
                image = np.pad(image, ((5, 5),
                                       (5, 5),
                                       (0, 0)),
                               constant_values=255)
                cv2.imwrite(os.path.join(target_root, file), image)
            else:
                image = train_dataset.edge_detection_from_numpy(image)
                image = train_dataset.resize_with_pad_for_square(image, tgt_size=1014)
                image = np.pad(image, ((5, 5),
                                       (5, 5),
                                       (0, 0)),
                               constant_values=255)
                cv2.imwrite(os.path.join(target_root, file), image)