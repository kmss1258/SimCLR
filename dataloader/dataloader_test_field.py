import random

import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
import os
import filetype

class DatasetTestField(Dataset):
    def __init__(self, data_dir, edge_detection_model=None, segmentation_model=None, detection_class=None):

        self.test_list = []
        self.classes = []

        self.test_list = self.get_data_list(data_dir) # IT/Fridge/drawing or IT/Fridge/real

        if edge_detection_model is not None:
            self.edge_detection_model = edge_detection_model # _MS_ arrived in cpu model
            self.edge_detection_model.eval()
            self.edge_detection_model = self.edge_detection_model.cuda()
        if segmentation_model is not None:
            self.segmentation_model = segmentation_model #
            self.segmentation_model.eval() #
            self.segmentation_model = self.segmentation_model.cuda()

        self.detection_class = detection_class

        # if detection_class is not None:
        #     self.detection_class = detection_class

    def get_data_list(self, data_dir : str):
        test_list = list()
        self.classes = self.class_list_up_from_directory_for_classification(data_dir)
        if len(self.classes) == 0:
            assert True, print("please confirm target directory : ", data_dir)

        print("target class number : ", len(self.classes))
        print("target class list : ", self.classes)

        for root, dirs, files in os.walk(data_dir):
            if data_dir == root:
                continue
            elif len(files) == 0:
                continue

            for file in files:
                if file.endswith("xml"):
                    continue
                label_dict = dict()

                for idx, class_str in enumerate(self.classes):
                    if class_str in root.split("/"):
                        label_dict["class_num"] = idx
                        break

                if len(label_dict) == 0:
                    assert print("class number error")


                if filetype.is_image(os.path.join(root, file)):
                    label_dict["file_dir"] = os.path.join(root, file)
                else:
                    print("Image Read error occured. image name : ", self.test_list[idx]["file_dir"])
                    continue

                test_list.append(label_dict)

        return test_list

    def class_list_up_from_directory_for_classification(self, data_dir : str):
        classes = os.listdir(data_dir)
        for idx, class_str in enumerate(classes):
            if os.path.isdir(os.path.join(data_dir, class_str)) == False:
                classes.pop(idx)
        classes.sort()
        return classes

    def resize_with_pad_for_square(self, src_img, tgt_size=256):

        if src_img.shape[0] == tgt_size and src_img.shape[1] == tgt_size:
            return src_img

        if src_img.shape[0] > src_img.shape[1]:
            resize_ratio = tgt_size / src_img.shape[0]
            pad_array = [0, int(tgt_size - src_img.shape[1] * resize_ratio + 1) // 2]
        else:
            resize_ratio = tgt_size / src_img.shape[1]
            pad_array = [int(tgt_size - src_img.shape[0] * resize_ratio + 1) // 2, 0]

        tgt_img = cv2.resize(src_img, dsize=(0, 0), fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_AREA)
        tgt_img = np.pad(tgt_img, ((pad_array[0], (tgt_size - tgt_img.shape[0] - pad_array[0])),
                                   (pad_array[1], (tgt_size - tgt_img.shape[1] - pad_array[1])), (0, 0)),
                         constant_values=255)

        return tgt_img

    def edge_detection_from_numpy(self, original_img : np.ndarray, resize=(1024, 1024)):
        """
        Downsizing image (with ratio) and generate edge image
        :param original_img:
        :return:
        """

        if original_img.shape[1] > original_img.shape[0]: # larger width
            fx_ratio = 1.0
            fy_ratio = original_img.shape[0] / original_img.shape[1]
        else:
            fx_ratio = original_img.shape[1] / original_img.shape[0]
            fy_ratio = 1.0

        input_image = original_img.copy()
        normalization_value = 3.5 if 128 / input_image.mean() > 3.5 else 128 / input_image.mean() # _MS_ normalization
        input_image = np.clip(input_image * normalization_value, 0, 255).astype(np.uint8) # _MS_ normalization
        input_image = cv2.resize(input_image, dsize=resize)
        input_image = np.array(input_image, dtype=np.float32)  # .astype(np.float32)
        input_image -= [103.939, 116.779, 123.68]
        input_image = input_image.transpose(2, 0, 1)[None]
        input_tensor = torch.from_numpy(input_image).cuda()

        # input_tensor = self.edge_detection_tensor(torch.from_numpy(input_image))

        output = self.edge_detection_model(input_tensor)
        output = output[-1].detach().cpu().numpy()
        output = 1 / (1 + np.exp(-output))
        output = np.uint8(self.image_normalization(output[0][0]))
        output = cv2.bitwise_not(output).astype(np.uint8)
        output = cv2.resize(output, dsize=(0, 0), fx=fx_ratio, fy=fy_ratio)
        output = np.expand_dims(output, axis=2)

        return np.concatenate([output, output, output], axis=2)

    def image_normalization(self, img, img_min=0, img_max=255, epsilon=1e-12):
        img = np.float32(img)
        img = (img - np.min(img)) * (img_max - img_min) / \
              ((np.max(img) - np.min(img)) + epsilon) + img_min
        return img

    def get_crop_image(self, image, detection_result, padding_ratio=0.01):
        height, width, _ = image.shape
        cx_original, cy_original, w_original, h_original = detection_result[0] * width, detection_result[1] * height, \
                                                           detection_result[2] * width, detection_result[3] * height

        left = int(cx_original - w_original // 2)
        right = int(cx_original + w_original // 2)
        top = int(cy_original - h_original // 2)
        bottom = int(cy_original + h_original // 2)

        left = int(left - width * padding_ratio) if int(left - width * padding_ratio) > 0 else 0
        right = int(right + width * padding_ratio) if int(right + width * padding_ratio) < width else width
        top = int(top - height * padding_ratio) if int(top - height * padding_ratio) > 0 else 0
        bottom = int(bottom + height * padding_ratio) if int(bottom + height * padding_ratio) < height else height

        return image[top:bottom, left:right, :].copy()

    def __getitem__(self, idx):
        label_num = self.test_list[idx]["class_num"]
        input_image = cv2.imread(self.test_list[idx]["file_dir"])
        if self.detection_class is not None:
            result = self.detection_class.object_discovery_inference(input_image)
            input_image = self.get_crop_image(input_image, result, padding_ratio=0.015)

        input_image = self.edge_detection_from_numpy(input_image, resize=(1024, 1024))
        input_image = self.resize_with_pad_for_square(input_image, tgt_size=512)

        input_numpy = (input_image.transpose(2, 0, 1) / 255.).astype(np.float32)
        # output_numpy = self.session.run(None, { self.session.get_inputs()[0].name: input_numpy } )[0]
        output_tensor = torch.from_numpy(input_numpy)[None]
        # output_tensor.to("cuda")

        return output_tensor, torch.tensor(data=label_num, dtype=torch.int64)

    def __len__(self):
        return len(self.test_list)





def annoy_test(anchor_list : list, test_list : list, feature_num = 100):
    """

    :param anchor_list: list( dict{"label" : int, "feature" : numpy.ndarray} )
    :param test_list: list( dict{"label" : int, "feature" : numpy.ndarray} )
    :param feature_num: number of embedded feature channel
    :return -> None:
    """

    from annoy import AnnoyIndex

    distance_calculate_list = ["angular", "euclidean", "manhattan", "hamming", "dot"]
    log_str_list = []

    max_top_1_counter = -1

    for distance_name in distance_calculate_list:
        annoy_table = AnnoyIndex(feature_num, distance_name)
        for idx, anchor_dict in enumerate(anchor_list):
            annoy_table.add_item(idx, anchor_dict["feature"])
        annoy_table.build(n_trees=1)

        test_total = len(test_list)
        top_1_counter = 0
        top_3_counter = 0

        for idx_test, test_dict in enumerate(test_list):
            result = annoy_table.get_nns_by_vector(test_dict["feature"], 3)
            for idx_result, rank in enumerate(result):
                if idx_result == 0 and anchor_list[rank]["label"] == test_list[idx_test]["label"]:
                    top_1_counter += 1
                    top_3_counter += 1
                    break
                elif anchor_list[rank]["label"] == test_list[idx_test]["label"]:
                    top_3_counter += 1
                    break

        if max_top_1_counter < top_1_counter:
            max_top_1_counter = top_1_counter

        log_str = "[{}]\t\t Test result : top 1 acc : {}/{} | top 3 acc : {}/{}".format(distance_name, top_1_counter, test_total,
                                                                                        top_3_counter, test_total)
        print(log_str)
        log_str_list.append(log_str)

    return log_str_list, max_top_1_counter

if __name__ == "__main__":

    edge_detection_model = torch.load("../utils/model_edge_detection.pt")
    # segmentation_model = torch.load("../utils/u2net.pt")

    Dataset_Test_Drawing = DatasetTestField(data_dir="/home/ms-neo/Pictures/_CUSTOM2/생활가전/가습기/drawing",
                                    edge_detection_model=edge_detection_model,
                                    segmentation_model=None)


    Dataset_Test_Real = DatasetTestField(data_dir="/home/ms-neo/Pictures/_CUSTOM2/생활가전/가습기/real",
                                         edge_detection_model=edge_detection_model,
                                         segmentation_model=None)
