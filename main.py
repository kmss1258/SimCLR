import os
import numpy as np
import torch
import torchvision
import argparse

# distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

# SimCLR_custom
from simclr import SimCLR
from simclr.modules import NT_Xent, get_resnet
from simclr.modules.transformations import TransformsSimCLR
from simclr.modules.sync_batchnorm import convert_model

from model import load_optimizer, save_model
from utils import yaml_config_hook

from dataloader.dataloader import DatasetWithBgSegAug
from dataloader.dataloader_test_field import DatasetTestField
from tqdm import tqdm
from object_discovery import object_discovery

# --dataset MS_CUSTOM --dataset_dir "/media/ms-neo/ms-ssd1/classifier_test_220823_target/가습기_2022726_170"

def train(args, train_loader, model, criterion, optimizer, writer):
    scaler = torch.cuda.amp.GradScaler() # _MS_ fixed
    loss_epoch = 0
    model.train()
    for step, loaded_data in enumerate(tqdm(train_loader)):

        if args.dataset == "CIFAR10":
            ((x_i, x_j), _) = loaded_data
        elif args.dataset == "MS_CUSTOM":
            (data, target, image_type) = loaded_data
            data_splitted = data.split(3, dim=1)
            assert len(data_splitted) >= 2, print("Confirm your batch size!")
            original_list = list()
            for _ in range(len(data_splitted) - 1):
                original_list.append(data_splitted[0]) # original batch == augmented batch
            x_i = torch.cat(original_list, dim=0) # original (not augmented)
            x_j = torch.cat(data_splitted[1:], dim=0) # augmented

        optimizer.zero_grad()
        x_i = x_i.cuda(non_blocking=True)
        x_j = x_j.cuda(non_blocking=True)

        # positive pair, with encoding
        with torch.cuda.amp.autocast():
            h_i, h_j, z_i, z_j = model(x_i, x_j)

            loss = criterion(z_i, z_j)

        scaler.scale(loss).backward() # loss.backward()

        scaler.step(optimizer) # optimizer.step()

        scaler.update() # None

        if dist.is_available() and dist.is_initialized():
            loss = loss.data.clone()
            dist.all_reduce(loss.div_(dist.get_world_size()))

        if args.nr == 0 and step % 1000 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")

        if args.nr == 0:
            writer.add_scalar("Loss/train_epoch", loss.item(), args.global_step)
            args.global_step += 1

        loss_epoch += loss.item()
    return loss_epoch

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

def test(model, device, criterion, test_loader):

    model.eval()

    test_loss = 0
    anchor_list = list()
    test_list = list()

    with torch.no_grad():
        for loaded_data in tqdm(test_loader):

            if args.dataset == "CIFAR10":
                ((x_i, x_j), _) = loaded_data
            elif args.dataset == "MS_CUSTOM":
                (data, target, image_type, file_directory) = loaded_data
                data_splitted = data.split(3, dim=1)
                assert len(data_splitted) >= 2, print("Confirm your batch size!")
                original_list = list()
                target_list = list()
                for _ in range(len(data_splitted) - 1):
                    original_list.append(data_splitted[0])  # original batch == augmented batch
                    target_list.append(target)
                x_i = torch.cat(original_list, dim=0)  # original (not augmented)
                x_j = torch.cat(data_splitted[1:], dim=0)  # augmented

                target = torch.cat(target_list, dim=0)
                target.to(device)

            x_i = x_i.cuda(non_blocking=True)
            x_j = x_j.cuda(non_blocking=True)

            # positive pair, with encoding
            h_i, h_j, z_i, z_j = model(x_i, x_j)
            loss = criterion(z_i, z_j)
            test_loss += loss * x_i.shape[0]

            output_feature_anchor = z_i.detach().cpu().numpy()
            output_feature_target = z_j.detach().cpu().numpy()
            target_numpy = target.detach().cpu().numpy().squeeze()

            for idx in range(len(output_feature_anchor)): # idx : batch?
                anchor_feature = output_feature_anchor[idx]
                test_feature = output_feature_target[idx]

                anchor_label = int(target_numpy[idx])
                test_label = int(target_numpy[idx])

                anchor_list.append({"feature" : anchor_feature, "label" : anchor_label})
                test_list.append({"feature" : test_feature, "label" : test_label})

    log_str_list, max_top1 = annoy_test(anchor_list, test_list, feature_num=len(anchor_list[0]["feature"]))

    del anchor_list, test_list

    test_loss /= (len(test_loader.dataset) * (1 + test_loader.dataset.augmentation_num))
    test_log_loss_dict = {"log" : log_str_list, "test_loss" : test_loss}
    return test_log_loss_dict, max_top1

def test_field(model, device, criterion, test_field_dataset_drawing, test_field_dataset_real):

    model.eval()

    anchor_list = list()
    test_list = list()

    print("Field Test Result : ")

    with torch.no_grad():
        for loaded_data in test_field_dataset_drawing:

            if args.dataset == "CIFAR10":
                ((x_i, x_j), _) = loaded_data
            elif args.dataset == "MS_CUSTOM": # 1 batch
                (data, target) = loaded_data
                x_i, x_j = data, data

            x_i = x_i.cuda(non_blocking=True)
            x_j = x_j.cuda(non_blocking=True)

            # positive pair, with encoding
            h_i, h_j, z_i, z_j = model(x_i, x_j)

            anchor_feature = z_i.detach().squeeze().cpu().numpy()
            target_numpy = target.detach().cpu().numpy().squeeze()
            target_numpy = int(target_numpy)

            anchor_list.append({"feature" : anchor_feature, "label" : target_numpy})

    with torch.no_grad():
        for loaded_data in test_field_dataset_real:

            if args.dataset == "CIFAR10":
                ((x_i, x_j), _) = loaded_data
            elif args.dataset == "MS_CUSTOM": # 1 batch
                (data, target) = loaded_data
                x_i, x_j = data, data

            x_i = x_i.cuda(non_blocking=True)
            x_j = x_j.cuda(non_blocking=True)

            # positive pair, with encoding
            h_i, h_j, z_i, z_j = model(x_i, x_j)

            test_feature = z_i.detach().squeeze().cpu().numpy()
            target_numpy = target.detach().cpu().numpy().squeeze()
            target_numpy = int(target_numpy)

            test_list.append({"feature" : test_feature, "label" : target_numpy})

    log_str_list, test_loss = annoy_test(anchor_list, test_list, feature_num=len(anchor_list[0]["feature"]))

    return {"log": log_str_list, "test_loss": test_loss}

def main(gpu, args):
    rank = args.nr * args.gpus + gpu

    if args.nodes > 1:
        dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
        torch.cuda.set_device(gpu)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset == "STL10":
        augmentation_num = 1
        train_dataset = torchvision.datasets.STL10(
            args.dataset_dir,
            split="unlabeled",
            download=True,
            transform=TransformsSimCLR(size=args.image_size),
        )
    elif args.dataset == "CIFAR10":
        augmentation_num = 1
        train_dataset = torchvision.datasets.CIFAR10(
            args.dataset_dir,
            download=True,
            transform=TransformsSimCLR(size=args.image_size),
        )
    elif args.dataset == "MS_CUSTOM":
        augmentation_num = 1
        class_each_drawing = False

        class_each_drawing_str = "each" if class_each_drawing is True else "group"
        train_dataset = DatasetWithBgSegAug(args.dataset_dir,
                                            background_aug_dir="./dataset/background",
                                            edge_detection_model=None,
                                            segmentation_model=None,
                                            class_each_drawing=class_each_drawing,
                                            augmentation_num=augmentation_num,
                                            is_train=True,
                                            is_debug=False,
                                            )
    else:
        raise NotImplementedError

    if args.nodes > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=rank, shuffle=True
        )
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        drop_last=True,
        num_workers=args.workers,
        sampler=train_sampler,
    )

    test_dataset = DatasetWithBgSegAug(args.dataset_dir,
                                      background_aug_dir="./dataset/background",
                                      edge_detection_model=None,
                                      segmentation_model=None,
                                      class_each_drawing=class_each_drawing,
                                      augmentation_num=1,
                                      is_train=False,
                                      is_debug=False
                                      )

    test_kwargs = {"batch_size": args.batch_size * augmentation_num,
                   "num_workers": args.batch_size,
                   "shuffle": True,
                   "pin_memory": True,
                   "drop_last": True,
                   }

    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
    edge_detection_model = torch.load("./utils/model_edge_detection.pt")
    detection_class = object_discovery(model_dir="./utils/object_discovery.pt")

    test_dataset_drawing = DatasetTestField(data_dir="./dataset/_CUSTOM2/생활가전/가습기/drawing",
                                            edge_detection_model=edge_detection_model,
                                            segmentation_model=None)

    test_dataset_real = DatasetTestField(data_dir="./dataset/_CUSTOM2/생활가전/가습기/real",
                                         edge_detection_model=edge_detection_model,
                                         segmentation_model=None,
                                         detection_class=detection_class)

    # initialize ResNet
    encoder = get_resnet(args.resnet, pretrained=True)
    if args.resnet in ["resnet50", "resnext50"]:
        n_features = encoder.fc.in_features
    elif args.resnet in ["NAVER_CGD"]:
        n_features = 1536 # encoder.fc.in_features  # get dimensions of fc layer

    # initialize model
    model = SimCLR(encoder, args.projection_dim, n_features)
    if args.reload:
        model_fp = os.path.join(
            args.model_path, "checkpoint_{}.tar".format(args.epoch_num)
        )
        model.load_state_dict(torch.load(model_fp, map_location=args.device.type))
    model = model.to(args.device)

    # optimizer / loss
    optimizer, scheduler = load_optimizer(args, model)
    criterion = NT_Xent(args.batch_size * augmentation_num, args.temperature, args.world_size) # _MS_ fixed

    # DDP / DP
    if args.dataparallel:
        model = convert_model(model)
        model = DataParallel(model)
    else:
        if args.nodes > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model, device_ids=[gpu])

    model = model.to(args.device)

    writer = None
    if args.nr == 0:
        writer = SummaryWriter()

    args.global_step = 0
    args.current_epoch = 0

    top_max_top1 = -1

    for epoch in range(args.start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        lr = optimizer.param_groups[0]["lr"]
        # if epoch % 10 == 0:
        #     _, _ = test_field(model, args.device, criterion, test_dataset_drawing, test_dataset_real)
        loss_epoch = train(args, train_loader, model, criterion, optimizer, writer)
        test_log_loss_dict, max_top1 = test(model, args.device, criterion, test_loader)
        os.makedirs("result_model", exist_ok=True)
        torch.save(model, os.path.join("result_model", "model_{}_dim{}_{}_{}_latest.pt".format(args.resnet,
                                                                                                 args.projection_dim,
                                                                                                 class_each_drawing_str,
                                                                                                 args.description)))

        if top_max_top1 < max_top1:
            torch.save(model, os.path.join("result_model", "model_{}_dim{}_{}_{}_max_top.pt".format(args.resnet,
                                                                                                 args.projection_dim,
                                                                                                 class_each_drawing_str,
                                                                                                 args.description)))
            top_max_top1 = max_top1

        if args.nr == 0 and scheduler:
            scheduler.step()

        # if args.nr == 0 and epoch % 10 == 0:
        #     save_model(args, model, optimizer)

        if args.nr == 0:
            writer.add_scalar("Loss/train", loss_epoch / len(train_loader), epoch)
            writer.add_scalar("Misc/learning_rate", lr, epoch)
            writer.add_scalar("Loss/test", test_log_loss_dict["test_loss"], epoch)
            print(
                f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(train_loader)}\t lr: {round(lr, 5)}"
            )
            args.current_epoch += 1


    print("max top 1 number : ", top_max_top1)
    model = torch.load(os.path.join("result_model", "model_{}_dim{}_{}_{}_max_top.pt".format(args.resnet,
                                                                                             args.projection_dim,
                                                                                             class_each_drawing_str,
                                                                                             args.description)))
    model = model.cuda()
    model.eval()
    _, _ = test_field(model, args.device, criterion, test_dataset_drawing, test_dataset_real)

    ## end training
    # save_model(args, model, optimizer)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SimCLR_custom")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    # Master address for distributed data parallel
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "8000"

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.num_gpus = torch.cuda.device_count()
    args.world_size = args.gpus * args.nodes

    if args.nodes > 1:
        print(
            f"Training with {args.nodes} nodes, waiting until all nodes join before starting training"
        )
        mp.spawn(main, args=(args,), nprocs=args.gpus, join=True)
    else:
        print(args.description)
        main(0, args)

    import socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(("kmss1258.synology.me", 7777))
    client_socket.sendall(b"MSG" + b"Hello world!")  # b"MSG" 다음 b + 원하는 메세지
    client_socket.close()

    # description yes dim 128 : topbottom yes + hard warping batch 48 worker 12 38XX tqdm 72
    # des no dim 64 : topbottom yes + no hard warping 38XX
    # des no dim 128 : topbottom no + no hard warping 2448