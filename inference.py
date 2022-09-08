import torch
import torch.nn as nn
from dataloader import dataloader
from tqdm import tqdm

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
            result = annoy_table.get_nns_by_vector(test_dict["feature"], 3 * 7) # 7 for chance (drawing average image number)
            search_label_list = []
            compare_counter = 1
            for idx_result, rank in enumerate(result):

                if idx_result == 0 and anchor_list[rank]["label"] == test_list[idx_test]["label"]:
                    top_1_counter += 1
                    top_3_counter += 1
                    break
                elif anchor_list[rank]["label"] == test_list[idx_test]["label"]:
                    top_3_counter += 1
                    break
                elif anchor_list[rank]["label"] in search_label_list: # one more chance
                    continue
                elif compare_counter == 3:
                    break
                else:
                    search_label_list.append(anchor_list[rank]["label"])
                    compare_counter += 1

        if max_top_1_counter < top_1_counter:
            max_top_1_counter = top_1_counter

        log_str = "[{}]\t\t Test result : top 1 acc : {}/{} | top 3 acc : {}/{}".format(distance_name, top_1_counter, test_total,
                                                                                        top_3_counter, test_total)
        print(log_str)
        log_str_list.append(log_str)

    return log_str_list, max_top_1_counter

if __name__ == "__main__":

    model = torch.load("result_model/model_SimCLR_resnet50")
    model = model.cuda()
    model.eval()

    testdataset = dataloader.DatasetWithBgSegAug(
        data_dir="/media/ms-neo/ms-ssd1/classifier_test_220823_target/가습기_2022726_170",
        # data_dir="/media/ms-neo/ms-ssd1/classifier_test_220823_target/무선이어폰_2022726_170",
        background_aug_dir="./dataset/background",
        edge_detection_model=None,
        segmentation_model=None,
        class_each_drawing = False,
        augmentation_num = 1,
        is_train = False,
        is_debug = False,
    )

    # image = cv2.imread("/media/ms-neo/ms-ssd1/classifier_test_220823_target/무선이어폰_2022726_170/"
    #                    "3020190004452/00001.jpg")
    # # image = train_dataset.edge_detection_from_numpy(image)
    # image = train_dataset.resize_with_pad_for_square(image, tgt_size=512)
    # image = image / 255.
    # output, feature = model(torch.from_numpy(image.astype(np.float32).transpose(2, 0, 1)[None]).cuda())
    # result_index = output.argmax(dim=1, keepdim=True)
    # result = result_index.detach().cpu().squeeze().numpy()
    # print(result)
    # print(train_dataset.classes[result])

    # numpy_concat = np.zeros(shape=(1, 64), dtype=np.float32)

    correct = 0

    anchor_list = []
    test_list = []
    for data, target, image_type, file_directory in tqdm(testdataset):

        data = data[None]
        data_splitted = data.split(3, dim=1) # aug 2 suppose

        data = torch.cat(data_splitted, dim=0)

        _, _, x_i, x_j = model(data_splitted[0].cuda(), data_splitted[1].cuda())
        anchor_list.append({"label" : target, "feature" : x_i.detach().squeeze().cpu().numpy()})
        test_list.append({"label" : target, "feature" : x_j.detach().squeeze().cpu().numpy()})

        # target_tensor = torch.tensor(data=target, dtype=torch.int64, device=torch.device("cuda"))[None]
        # target_tensor = torch.cat((target_tensor, target_tensor), dim=0)

        # pred = output.argmax(dim=1, keepdim=True)
        # correct += pred.eq(target_tensor.view_as(pred)).sum().item()

    annoy_test(anchor_list, test_list, 64)
    # print("correct : {} / {}".format(correct, len(anchor_list) * 2))


    # output_feature = np.concatenate(feature_list, axis=0)
    #
    # tsne_result = TSNE(n_components=2, verbose=1, n_iter=500).fit_transform(output_feature)
    #
    # plt.figure(figsize=(20, 20))
    #
    # plt.xlim(tsne_result[:, 0].min(), tsne_result[:, 0].max() + 1)
    # plt.ylim(tsne_result[:, 1].min(), tsne_result[:, 1].max() + 1)
    #
    # for idx in range(len(target_list)):
    #     # if idx in target_list_idx:
    #     #
    #     #else:
    #     if target_list[idx] == 0:
    #         plt.scatter(tsne_result[idx, 0], tsne_result[idx, 1], c='#0000FF')
    #     elif target_list[idx] == 1:
    #         plt.scatter(tsne_result[idx, 0], tsne_result[idx, 1], c='#FF0000')
    #     else:
    #         plt.scatter(tsne_result[idx, 0], tsne_result[idx, 1], c='#00FF00')
    #
    # # plt.scatter(tsne_result[0, 0], tsne_result[0, 1], c='#FF0000')
    #
    # # for idx in range(len(similarity_numpy.shape[0])):
    # #     plt.text(digits_tsne[i, 0], digits_tsne[i, 1], str(digits.target[i]),
    # #              color=colors[digits.target[i]],
    # #              fontdict={'weight': 'bold', 'size': 9})
    #
    # plt.xlabel("t-SNE 1")
    # plt.ylabel("t-SNE 2")
    #
    # plt.show()
