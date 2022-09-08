import random

import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
import os
import filetype

# torch.multiprocessing.set_start_method('spawn')

## SIMILARITY : jaccard_similarity
## return len(s1&s2)/len(s1|s2)

# if torch.cuda.device_count() > 1:
#   model = nn.DataParallel(model)
# Nvidia Apex

# Structure
# TARGET FOLDER
# - CLASS1 FOLDER
# -- DRAWING IMAGE FOLDER
# --- .jpg ONLY DRAWING (with not need for augmentation)
# -- REAL IMAGE FOLDER (with need for augmentation)
# --- .jpg (REAL IMAGES)
# -- CRAWLING IMAGE FOLDER
# --- .jpg (Crawling Images)
# - CLASS2 FOLDER
# .......

class DatasetWithBgSegAug(Dataset):
    def __init__(self, data_dir="", background_aug_dir="",
                 edge_detection_model : torch.nn.Module = None,
                 segmentation_model : torch.nn.Module = None,
                 is_train=True, class_each_drawing=False,
                 augmentation_num=1, is_debug=False):

        self.test_list = []
        self.classes = []
        self.is_train = is_train
        self.is_debug = is_debug
        self.augmentation_num = augmentation_num

        if edge_detection_model is not None:
            self.edge_detection_model = edge_detection_model # _MS_ arrived in cpu model
            self.edge_detection_model.eval()
            self.edge_detection_model = self.edge_detection_model.cuda()
        if segmentation_model is not None:
            self.segmentation_model = segmentation_model #
            self.segmentation_model.eval() #
            self.segmentation_model = self.segmentation_model.cuda()

        # if torch.cuda.is_available():
        #     self.input_tensor_edge_detection = torch.zeros(size=(3, 512, 512), dtype=torch.float32).cuda()
        #     self.input_tensor_segmentation = torch.zeros(size=(3, 256, 256), dtype=torch.float32).cuda()
        # else:
        #     self.input_tensor_edge_detection = torch.zeros(size=(3, 512, 512), dtype=torch.float32)
        #     self.input_tensor_segmentation = torch.zeros(size=(3, 256, 256), dtype=torch.float32)

        if is_train is True:
            self.train_list = []

        if data_dir is not None:
            self.test_list = self.get_data_list(data_dir, class_each_drawing, except_top_bottom=False)
        if background_aug_dir is not None:
            self.background_aug_list = self.get_aug_background_data_list(background_aug_dir)

        if is_train:  # divide into train and test dataloader
            pass # TODO :

    def get_aug_background_data_list(self, background_aug_dir: str) -> list:
        aug_background_list = list()
        for root, dirs, files in os.walk(background_aug_dir):
            for file in files:
                if cv2.imread(os.path.join(root, file)) is None:
                    print(os.path.join(root, file) + " file has an error")
                    continue

                if filetype.is_image(os.path.join(root, file)) is True:
                    aug_background_list.append(os.path.join(root, file))

        print("Augmentation Background Image list number : ", len(aug_background_list))
        return aug_background_list

    def get_data_list(self, data_dir : str, class_each_drawing=False, except_top_bottom=True) -> list:
        """
        :param data_dir: training data directory or validation data directory
            * input example : "sample_dataset/train/category1"
        :return test_list: test_list include as below:
            test_list[0] : label_dict -> dict()
                * label_dict[string]
                 - "class_num" : idx of class (ex, 3019870019525 's index is 0, 3019654897185 's index is 1 ...)
                 - "file_dir" : train/val image file directory (ex, "train/category1/3019870019525/crawling/1.png")
                 - "file_type" : image file type : (ex, "crawling", "drawing", "studio")
            test_list[1] : label_dict
                ...
        """
        test_list = list()
        if class_each_drawing is False:
            self.classes = self.class_list_up_from_directory_for_classification(data_dir)
        else:
            self.classes = self.class_list_up_from_drawing_for_classification(data_dir)
        if len(self.classes) == 0:
            assert True, print("please confirm target directory : ", data_dir)

        print("target class number : ", len(self.classes))
        # print("target class list : ", self.classes)

        for root, dirs, files in os.walk(data_dir):
            files.sort()
            for idx, file in enumerate(files):
                if file.endswith("xml") or file.endswith("txt"):
                    continue
                if idx > 4 and except_top_bottom is True:
                    break
                label_dict = dict() # _MS_ labeled dictionary

                for idx, class_str in enumerate(self.classes):
                    # if class_str in root.split("/"):
                    target_file = os.path.join(root, file)
                    if target_file.find(class_str) != -1:
                        label_dict["class_num"] = idx
                        break

                if len(label_dict) == 0:
                    assert print("class number error")

                if filetype.is_image(os.path.join(root, file)):
                    label_dict["file_dir"] = os.path.join(root, file)
                    label_dict["file_type"] = self.get_file_type(root)
                else:
                    print("Image Read error occured. image name : ", os.path.join(root, file))
                    continue

                test_list.append(label_dict)

        return test_list


    def get_file_type(self, data_dir : str, file_type_list=("crawling", "drawing", "studio")) -> str:
        for file_type in file_type_list:
            if file_type in data_dir.split("/"):
                return file_type
        return "drawing" # if not defined file type
        # assert "unknown file_type. please confirm directory : ", data_dir

    def class_list_up_from_directory_for_classification(self, data_dir : str):
        classes = os.listdir(data_dir)
        for idx, class_str in enumerate(classes):
            if os.path.isdir(os.path.join(data_dir, class_str)) == False:
                classes.pop(idx)
        classes.sort()
        return classes

    def class_list_up_from_drawing_for_classification(self, data_dir : str):
        classes = list()
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                target_file = os.path.join(root, file)
                if filetype.is_image(target_file):
                    classes.append(os.path.join(target_file.split("/")[-2], target_file.split("/")[-1]))
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

    def floodfill_image_to_mask(self, image : np.ndarray):

        th, im_th = cv2.threshold(image[:, :, 0:1], 220, 255, cv2.THRESH_BINARY_INV)
        im_floodfill = im_th.copy()
        h, w = im_th.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(im_floodfill, mask, (0, 0), 255) # TODO : (0,0) (0,H) (W,0) (W,H) all????
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        image_mask = im_th | im_floodfill_inv

        image_mask = image_mask > 50
        image_mask = np.expand_dims(image_mask, axis=2)

        return image_mask

    def augmentation_image_from_type(self, image : np.ndarray, image_type : str, return_image_only=False, output_size=1024):

        ## AUG MODULE -> Flipping (UP-Down), Flipping (Left-Right), Rotation, Warping, Zoom-in/out 10%
        # DRAWING : EDGE DETECTION -> AUG MODULE -> BG sunthesis
        # STUDIO : Segmentation -> EDGE DETECTION -> AUG MODULE -> BG synthesis
        # CRAWLING : EDGE DETECTION -> AUG MODULE

        augmented_image = np.zeros((1, ))

        if image_type == "drawing":
            # image = self.edge_detection_from_numpy(image) # downsizing with same ratio, largest value is 512
            image = self.resize_with_pad_for_square(image, tgt_size=output_size)
            image = self.augmentation_image_module(image, zoom=(-0.1 - random.random() * 0.3), shift=(0.1, 0.1),
                                                   do_rotation_first=True, rotation=5)
            if return_image_only is True:
                return image
            mask = self.floodfill_image_to_mask(image) # AFTER AUGMENTATION

            bg_image = cv2.imread(self.background_aug_list[random.randint(0, len(self.background_aug_list) - 1)])
            # bg_image = self.edge_detection_from_numpy(bg_image) #  downsizing with same ratio, largest value is 512
            # bg_image = self.resize_with_pad_for_square(bg_image)
            bg_image = cv2.resize(bg_image, dsize=(output_size, output_size))
            bg_image = self.augmentation_image_module(bg_image, zoom=(0.1 + random.random() * 0.2), warping=(0., 0.), rotation=0)

            augmented_image = np.zeros(shape=bg_image.shape, dtype=np.uint8)
            augmented_image += image * mask
            augmented_image += bg_image * ~mask

        elif image_type == "studio": # crawling / segmentation
            image = self.resize_with_pad_for_square(image)
            image, mask_image = self.segmentation_module(image)

            image = self.edge_detection_from_numpy(image)  # downsizing with same ratio, largest value is 512
            image = self.resize_with_pad_for_square(image)
            image, mask_image = self.augmentation_image_module_from_array([image, mask_image],
                                                                          zoom=(-0.1 - random.random() * 0.3),
                                                                          transform_border_black_tuple=(False, True),
                                                                          do_rotation_first=True)
            if return_image_only is True:
                return image
            mask = (mask_image > 35)
            mask = np.expand_dims(mask, axis=2)
            # No FloodFill for mask

            bg_image = cv2.imread(self.background_aug_list[random.randint(0, len(self.background_aug_list) - 1)])
            bg_image = self.edge_detection_from_numpy(bg_image) # downsizing with same ratio, largest value is 512
            bg_image = self.resize_with_pad_for_square(bg_image)
            bg_image = self.augmentation_image_module(bg_image, zoom=(0.5 -random.random() * 0.2))

            augmented_image = np.zeros(shape=bg_image.shape, dtype=np.uint8)
            augmented_image += image * mask
            augmented_image += bg_image * ~mask

        elif image_type == "crawling":
            # image = self.edge_detection_from_numpy(image)  # downsizing with same ratio, largest value is 512
            image = self.resize_with_pad_for_square(image)
            augmented_image = self.augmentation_image_module(image, zoom=(-0.1 - random.random() * 0.3))
        else:
            assert print("Could not parse image type : ", image_type)

        return augmented_image

    def augmentation_image_module(self, input_image : np.ndarray, flipping_up_down=True, flipping_left_right=True,
                                  rotation=45, warping=(0.1, 0.1), shift=(0.2, 0.2), zoom=-0.3, do_rotation_first=False):
        """
        :param input_image : image HAS WHITE BACKGROUND with object
        :param flipping_up_down: do flip up-down direction
        :param flipping_left_right: do flip left-right direction
        :param rotation: rotation degree (randint(-rotation, rotation))
        :param warping (x,y): -rand.int(-input_shape * x, +input_shape * x ...) and warping[]y axis, warping images
        :param zoom: resizing image ratio drawing -> minus | crawling, studio -> plus
        :return -> augmented image:
        """
        processed_image = input_image.copy()

        if rotation != 0 or zoom != 0 and do_rotation_first is True:
            rotation_value = random.randint(-rotation, rotation)
            M = cv2.getRotationMatrix2D((input_image.shape[1] // 2, input_image.shape[0] // 2), rotation_value,
                                        zoom + 1.0)
            processed_image = cv2.warpAffine(processed_image, M, (input_image.shape[1], input_image.shape[0]),
                                             borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

        if flipping_up_down is True and flipping_left_right:
            if random.random() > 0.5:
                processed_image = processed_image[::-1]
        # if flipping_left_right is True:
        #     if random.random() > 0.5:
                processed_image = processed_image[:, ::-1]
        if shift[0] != 0 or shift[1] != 0:
            x_rand = random.randint(int(-processed_image.shape[1] * shift[0]), int(processed_image.shape[1] * shift[0]))
            y_rand = random.randint(int(-processed_image.shape[0] * shift[1]), int(processed_image.shape[0] * shift[1]))
            M = np.float32([[1, 0, x_rand],
                            [0, 1, y_rand]])
            processed_image = cv2.warpAffine(processed_image, M, (processed_image.shape[1], processed_image.shape[0]),
                                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        if warping[0] != 0 or warping[1] != 0:
            corners = np.array([[0, 0], [0, processed_image.shape[0]], [processed_image.shape[1], 0],
                                [processed_image.shape[1], processed_image.shape[0]]], dtype=np.float32)
            warp_x = np.random.randint(-processed_image.shape[1] * warping[0], +processed_image.shape[1] * warping[0],
                                       size=(4, 1)).astype(np.float32)
            warp_y = np.random.randint(-processed_image.shape[0] * warping[1], +processed_image.shape[0] * warping[1],
                                       size=(4, 1)).astype(np.float32)
            warp = np.concatenate((warp_x, warp_y), axis=1)
            M = cv2.getPerspectiveTransform(corners, corners + warp)
            processed_image = cv2.warpPerspective(processed_image, M, (processed_image.shape[1], processed_image.shape[0]),
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        if rotation != 0 or zoom != 0 and do_rotation_first is False:
            rotation_value = random.randint(-rotation, rotation)
            M = cv2.getRotationMatrix2D((input_image.shape[1] // 2, input_image.shape[0] // 2), rotation_value, zoom + 1.0)
            processed_image = cv2.warpAffine(processed_image, M, (input_image.shape[1], input_image.shape[0]),
                                             borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        # if zoom != 0.:
        #     # white_background = np.zeros(input_image.shape, dtype=np.uint8) + 255
        #     temp_image = cv2.resize(processed_image, dsize=(0, 0), fx=1+zoom, fy=1+zoom)
        #     if zoom < 0:
        #         y_axis_padding_half = (processed_image.shape[0] - temp_image.shape[0]) // 2
        #         x_axis_padding_half = (processed_image.shape[1] - temp_image.shape[1]) // 2
        #         processed_image = np.pad(temp_image, (
        #                                 (y_axis_padding_half, input_image.shape[0] - temp_image.shape[0] - y_axis_padding_half),
        #                                 (x_axis_padding_half, input_image.shape[1] - temp_image.shape[1] - x_axis_padding_half),
        #                                 (0, 0)), "constant", constant_values=128
        #                                 )
        #     else:
        #         y_axis_padding_half = (temp_image.shape[0] - processed_image.shape[0]) // 2
        #         x_axis_padding_half = (temp_image.shape[1] - processed_image.shape[1]) // 2
        # 
        #         processed_image = temp_image[y_axis_padding_half:processed_image.shape[0] + y_axis_padding_half,
        #                           x_axis_padding_half:processed_image.shape[1] + x_axis_padding_half]

        return processed_image

    def augmentation_image_module_from_array(self, input_image_list : list, flipping_up_down=True, flipping_left_right=True,
                                             rotation=45, warping=(0.1, 0.1), shift=(0.2, 0.2), zoom=-0.3, do_rotation_first=True,
                                             transform_border_black_tuple=(False, True)):
        """
        generate augmentation image with same transform matrix
        :param input_image : image HAS WHITE BACKGROUND with object
        :param flipping_up_down: do flip up-down direction
        :param flipping_left_right: do flip left-right direction
        :param rotation: rotation degree (randint(-rotation, rotation))
        :param warping (x,y): -rand.int(-input_shape * x, +input_shape * x ...) and warping[]y axis, warping images
        :param zoom: resizing image ratio drawing -> minus | crawling, studio -> plus
        :return -> augmented image:
        """
        processed_image_list = []
        border_color_list = []
        assert len(transform_border_black_tuple) == len(input_image_list), print("argument length is not same in blacklist and input list")

        for idx, input_image_ in enumerate(input_image_list):
            processed_image_list.append(input_image_.copy())
            if transform_border_black_tuple[idx] is False: # White BORDER
                border_color_list.append((255, 255, 255))
            else: # White BORDER
                border_color_list.append((0, 0, 0))
        input_image_shape = processed_image_list[0].shape

        if rotation != 0 or zoom != 0 and do_rotation_first is True:
            rotation_value = random.randint(-rotation, rotation)
            for idx, input_image in enumerate(processed_image_list):
                M = cv2.getRotationMatrix2D((input_image.shape[1] // 2, input_image.shape[0] // 2), rotation_value, zoom + 1.0)
                processed_image_list[idx] = cv2.warpAffine(input_image, M, (input_image.shape[1], input_image.shape[0]),
                                                           borderMode=cv2.BORDER_CONSTANT,
                                                           borderValue=border_color_list[idx])
        if flipping_up_down is True:
            if random.random() > 0.5:
                for idx, input_image in enumerate(processed_image_list):
                    processed_image_list[idx] = input_image[::-1]
        if flipping_left_right is True:
            if random.random() > 0.5:
                for idx, input_image in enumerate(processed_image_list):
                    processed_image_list[idx] = input_image[:, ::-1]
        if shift[0] != 0 or shift[1] != 0:
            x_rand = random.randint(int(-input_image_shape[1] * shift[0]), int(input_image_shape[1] * shift[0]))
            y_rand = random.randint(int(-input_image_shape[0] * shift[1]), int(input_image_shape[0] * shift[1]))
            M = np.float32([[1, 0, x_rand],
                            [0, 1, y_rand]])
            for idx, input_image in enumerate(processed_image_list):
                processed_image_list[idx] = cv2.warpAffine(input_image, M, (input_image.shape[1], input_image.shape[0]),
                                                      borderMode=cv2.BORDER_CONSTANT, borderValue=border_color_list[idx])
        if warping[0] != 0 or warping[1] != 0:
            corners = np.array([[0, 0], [0, input_image_shape[0]], [input_image_shape[1], 0],
                                [input_image_shape[1], input_image_shape[0]]], dtype=np.float32)
            warp_x = np.random.randint(-input_image_shape[1] * warping[0], +input_image_shape[1] * warping[0],
                                       size=(4, 1)).astype(np.float32)
            warp_y = np.random.randint(-input_image_shape[0] * warping[1], +input_image_shape[0] * warping[1],
                                       size=(4, 1)).astype(np.float32)
            warp = np.concatenate((warp_x, warp_y), axis=1)
            M = cv2.getPerspectiveTransform(corners, corners + warp)
            for idx, input_image in enumerate(processed_image_list):
                processed_image_list[idx] = cv2.warpPerspective(input_image, M, (input_image.shape[1], input_image.shape[0]),
                                            borderMode=cv2.BORDER_CONSTANT, borderValue=border_color_list[idx])
        if rotation != 0 or zoom != 0 and do_rotation_first is False:
            rotation_value = random.randint(-rotation, rotation)
            for idx, input_image in enumerate(processed_image_list):
                M = cv2.getRotationMatrix2D((input_image.shape[1] // 2, input_image.shape[0] // 2), rotation_value, zoom + 1.0)
                processed_image_list[idx] = cv2.warpAffine(input_image, M, (input_image.shape[1], input_image.shape[0]),
                                                 borderMode=cv2.BORDER_CONSTANT, borderValue=border_color_list[idx])

        return processed_image_list

    def normalization_numpy(self, input_numpy: np.ndarray) -> np.ndarray:
        np_max = np.max(input_numpy)
        np_min = np.min(input_numpy)
        return (input_numpy - np_min) / (np_max - np_min)

    def segmentation_module(self, original_img : np.ndarray):

        input_img = original_img.copy()
        input_shape = (input_img.shape[0], input_img.shape[1])
        input_img = input_img[:, :, ::-1]
        input_img = cv2.resize(input_img, dsize=(320, 320))
        input_img = input_img / 255.
        input_img[:, :, 0] = (input_img[:, :, 0] - 0.485) / 0.229
        input_img[:, :, 1] = (input_img[:, :, 1] - 0.456) / 0.224
        input_img[:, :, 2] = (input_img[:, :, 2] - 0.406) / 0.225
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = torch.tensor(input_img[None], dtype=torch.float32, device=torch.device("cuda"))
        mask_img = (self.segmentation_model(input_tensor)[0].squeeze()).detach().cpu().numpy()
        mask_img = self.normalization_numpy(mask_img) * 255
        mask_img = mask_img.astype(np.uint8)
        mask_img = cv2.resize(mask_img, dsize=input_shape)

        # mask_result = cv2.resize(np.concatenate((mask_img[None], mask_img[None], mask_img[None])), dsize=input_shape,
        #                     interpolation=cv2.INTER_LINEAR) # COLOR RETURN

        mask_filter = (mask_img > 35)
        mask_filter = np.expand_dims(mask_filter, axis=2)

        mask_result = original_img * mask_filter + (np.random.rand(*input_shape, 3) + 255).astype(np.uint8) * ~mask_filter

        return mask_result, mask_img

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
        input_image = cv2.resize(input_image, dsize=resize)
        input_image = np.array(input_image, dtype=np.float32)  # .astype(np.float32)
        input_image -= [103.939, 116.779, 123.68]
        input_image = input_image.transpose(2, 0, 1)[None]
        input_tensor = torch.from_numpy(input_image).cuda()

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

    def __getitem__(self, idx):
        label_num = self.test_list[idx]["class_num"]
        original_image = cv2.imread(self.test_list[idx]["file_dir"])
        original_image = self.resize_with_pad_for_square(original_image, tgt_size=512)

        augmentation_list = list()

        for _ in range(self.augmentation_num):
            image = original_image.copy()
            image = self.augmentation_image_from_type(image,
                                                      self.test_list[idx]["file_type"],
                                                      return_image_only=False,
                                                      output_size=512)
            # resize?
            image = image.astype(np.float32)
            image /= 255.
            input_numpy = (image.transpose(2, 0, 1)).astype(np.float32)
            output_tensor = torch.from_numpy(input_numpy)
            augmentation_list.append(output_tensor)

        # cv2.imwrite("temp_original.png", original_image)
        # cv2.imwrite("temp.png", image)

        original_image = original_image.astype(np.float32)
        original_image /= 255.
        original_numpy = (original_image.transpose(2, 0, 1)).astype(np.float32)
        output_tensor_original = torch.from_numpy(original_numpy)

        # image -= [103.939, 116.779, 123.68]
        # image = self.resize_with_pad_for_square(cv2.imread(self.test_list[idx]["file_dir"]), tgt_size=256)


        if self.is_debug is True:
            os.makedirs("dataloader_result", exist_ok=True)
            cv2.imwrite(os.path.join("dataloader_result", "{:06}_original.png".format(idx)),
                        (output_tensor_original * 255).numpy().astype(np.uint8).transpose(1, 2, 0))
            for temp_idx, temp_image in enumerate(augmentation_list):
                cv2.imwrite(os.path.join("dataloader_result", "{:06}_augmented_{}.png".format(idx, temp_idx)),
                            (temp_image * 255).numpy().astype(np.uint8).transpose(1, 2, 0))

        if self.is_train is True:
            return torch.cat((output_tensor_original, *augmentation_list), dim=0), label_num, \
                   self.test_list[idx]["file_type"] # torch.tensor((label_num, ))
        else:
            return torch.cat((output_tensor_original, *augmentation_list), dim=0), label_num, \
                   self.test_list[idx]["file_type"], self.test_list[idx]["file_dir"]
        # return : (C * N, H, W), (integer, ), (str, ), (str, )
    def __len__(self):
        return len(self.test_list)

















# Structure

# TARGET FOLDER
# - CLASS1 FOLDER
# -- ANY FOLDER OR IMAGES ...
# - CLASS2 FOLDER
# -- ANY FOLDER OR IMAGES ...

class uvcgan_dataset_simple(Dataset):
    def __init__(self, data_dir, is_train=False):
        # self.session = ort.InferenceSession("gan_ab_ext.onnx", providers=["CUDAExecutionProvider"])
        # self.session = ort.InferenceSession("gan_ab_ext.onnx", providers=["CPUExecutionProvider"])
        self.test_list = []
        self.classes = []

        if is_train is True:
            self.train_list = []

        self.test_list = self.get_data_list(data_dir)

        if is_train:  # divide into train and test dataloader
            pass # TODO

    def __getitem__(self, idx):
        # label_num = self.test_list[idx]["class_num"]
        # image = self.resize_with_pad_for_square(cv2.imread(self.test_list[idx]["file_dir"]), tgt_size=256)
        # input_numpy = (image.transpose(2, 0, 1) / 255.).astype(np.float32)
        # # output_numpy = self.session.run(None, { self.session.get_inputs()[0].name: input_numpy } )[0]
        # output_tensor = torch.from_numpy(input_numpy)
        # # output_tensor.to("cuda")

        return output_tensor, label_num

    def __len__(self):
        return len(self.test_list)

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
                    assert True, print("class number error")

                # org_image = cv2.imread(os.path.join(root, file))
                # if org_image is not None:

                if filetype.is_image(os.path.join(root, file)):
                    label_dict["file_dir"] = os.path.join(root, file)
                else:
                    # print("Image Read error occured. image name : ", self.test_list[idx]["file_dir"])
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

if __name__ == "__main__":

    edge_detection_model = torch.load("../utils/model_edge_detection.pt")
    segmentation_model = torch.load("../utils/u2net.pt")

    Dataset_Test = DatasetWithBgSegAug(data_dir="../dataset/background", background_aug_dir="../dataset/background",
                                       edge_detection_model=edge_detection_model,
                                       segmentation_model=segmentation_model)

    output_image = Dataset_Test.augmentation_image_from_type(cv2.imread("../sample_images/real_a/sample_36.png"), "drawing")
    cv2.imwrite("output.png", output_image)

    # 1070256845