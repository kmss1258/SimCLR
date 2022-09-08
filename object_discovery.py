import torch
import numpy as np
import cv2
from torchvision import transforms as pth_transforms
from scipy.linalg import eigh
from scipy import ndimage
import torch.nn.functional as F

feat_out = {}

def hook_fn_forward_qkv(module, input, output):
    feat_out["qkv"] = output

class object_discovery():
    def __init__(self, model_dir):
        self.detection_model = self.object_discovery_init(model_dir)

    def object_discovery_init(self, model_dir) -> torch.nn.Module:

        # model._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)
        model = torch.load(model_dir)

        self.feat_out = {}
        def hook_fn_forward_qkv(module, input, output):
            self.feat_out["qkv"] = output
        model._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)

        if torch.cuda.is_available():
            model = model.cuda()

        return model.get_last_selfattention

    def ncut(self, feats, dims, scales, init_image_size, tau=0, eps=1e-5, im_name='', no_binary_graph=False):
        """
        Implementation of NCut Method.
        Inputs
          feats: the pixel/patche features of an image
          dims: dimension of the map from which the features are used
          scales: from image to map scale
          init_image_size: size of the image
          tau: thresold for graph construction
          eps: graph edge weight
          im_name: image_name
          no_binary_graph: ablation study for using similarity score as graph edge weight
        """
        cls_token = feats[0, 0:1, :].cpu().numpy()

        feats = feats[0, 1:, :]
        feats = F.normalize(feats, p=2)
        A = (feats @ feats.transpose(1, 0))
        A = A.cpu().numpy()
        if no_binary_graph:
            A[A < tau] = eps
        else:
            A = A > tau
            A = np.where(A.astype(float) == 0, eps, A)
        d_i = np.sum(A, axis=1)
        D = np.diag(d_i)

        # Print second and third smallest eigenvector
        _, eigenvectors = eigh(D - A, D, subset_by_index=[1, 2])
        eigenvec = np.copy(eigenvectors[:, 0])

        # Using average point to compute bipartition
        second_smallest_vec = eigenvectors[:, 0]
        avg = np.sum(second_smallest_vec) / len(second_smallest_vec)
        bipartition = second_smallest_vec > avg

        seed = np.argmax(np.abs(second_smallest_vec))

        if bipartition[seed] != 1:
            eigenvec = eigenvec * -1
            bipartition = np.logical_not(bipartition)
        bipartition = bipartition.reshape(dims).astype(float)

        # predict BBox
        pred, _, objects, cc = self.detect_box(bipartition, seed, dims, scales=scales,
                                          initial_im_size=init_image_size[1:])  ## We only extract the principal object BBox
        mask = np.zeros(dims)
        mask[cc[0], cc[1]] = 1

        return np.asarray(pred), objects, mask, seed, None, eigenvec.reshape(dims)


    def detect_box(self, bipartition, seed, dims, initial_im_size=None, scales=None, principle_object=True):
        """
        Extract a box corresponding to the seed patch. Among connected components extract from the affinity matrix, select the one corresponding to the seed patch.
        """
        w_featmap, h_featmap = dims
        objects, num_objects = ndimage.label(bipartition)
        cc = objects[np.unravel_index(seed, dims)]

        if principle_object:
            mask = np.where(objects == cc)
            # Add +1 because excluded max
            ymin, ymax = min(mask[0]), max(mask[0]) + 1
            xmin, xmax = min(mask[1]), max(mask[1]) + 1
            # Rescale to image size
            r_xmin, r_xmax = scales[1] * xmin, scales[1] * xmax
            r_ymin, r_ymax = scales[0] * ymin, scales[0] * ymax
            pred = [r_xmin, r_ymin, r_xmax, r_ymax]

            # Check not out of image size (used when padding)
            if initial_im_size:
                pred[2] = min(pred[2], initial_im_size[1])
                pred[3] = min(pred[3], initial_im_size[0])

            # Coordinate predictions for the feature space
            # Axis different then in image space
            pred_feats = [ymin, xmin, ymax, xmax]

            return pred, pred_feats, objects, mask
        else:
            raise NotImplementedError

    def object_discovery_inference(self, image): # BGR image

        # image = cv2.imread(image_directory)[:, :, ::-1].copy()
        image = image[:, :, ::-1].copy()
        transform_resize = pth_transforms.Compose(
            [
                pth_transforms.ToTensor(),
                pth_transforms.Resize(640),
                pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]
        )
        img = transform_resize(image)

        init_image_size = img.shape
        size_im = (
            img.shape[0],
            int(np.ceil(img.shape[1] / 16) * 16),
            int(np.ceil(img.shape[2] / 16) * 16),
        )
        paded = torch.zeros(size_im)
        paded[:, : img.shape[1], : img.shape[2]] = img
        img = paded

        if torch.cuda.is_available():
            img = img.cuda(non_blocking=True)

        w_featmap = img.shape[-2] // 16
        h_featmap = img.shape[-1] // 16

        attentions = self.detection_model(img[None])
        scales = [16, 16]

        nb_im = attentions.shape[0]  # Batch size
        nh = attentions.shape[1]  # Number of heads
        nb_tokens = attentions.shape[2]  # Number of tokens

        qkv = (
            self.feat_out["qkv"]
            .reshape(nb_im, nb_tokens, 3, nh, -1 // nh)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        k = k.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
        # q = q.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
        # v = v.transpose(1, 2).reshape(nb_im, nb_tokens, -1)

        feats = k

        pred, objects, foreground, seed, bins, eigenvector = self.ncut(feats, [w_featmap, h_featmap], scales, init_image_size,
                                                                       0.2, 1e-05, None,
                                                                       no_binary_graph=False)

        pred_relative = (pred[2] + pred[0]) / 2 / init_image_size[2], \
                        (pred[3] + pred[1]) / 2 / init_image_size[1], \
                        (pred[2] - pred[0]) / init_image_size[2], \
                        (pred[3] - pred[1]) / init_image_size[1]

        return pred_relative # CX, CY, W, H

if __name__ == "__main__":

    model = object_discovery("utils/object_discovery.pt")

    import time
    start_time = time.time()
    for _ in range(1):
        image = cv2.imread("utils/deleteme.png")
        result = model.object_discovery_inference(image) #
    print("end time : {:.04} seconds".format(time.time() - start_time))

    print(result)