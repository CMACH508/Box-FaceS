import torch
import torch.nn.functional as F
import numpy as np

# MAX_BOX = [[9, 9], [13, 11], [11, 6], [11, 8], [7, 4], [7, 5]] # >100%, 32px, fan

# MAX_BOX = [[6, 6], [4, 9], [3, 7], [3, 7], [3, 6], [3, 6]]  # >80%, 32px

HIST_BOX = [[7, 7], [6, 9], [4, 8], [4, 8], [4, 6], [4, 6]]  # from hist, 32px


# AVG_BOX = [[6, 6], [8, 4], [7, 3], [7, 3], [5, 3], [5, 3]]  # avg, 32px, fan


# MAX_BOX = [[4, 5], [6, 5], [6, 3], [6, 3], [4, 3], [4, 2]] # >100%, 16px, fan


# MAX_BOX = [[5, 4], [5, 6], [3, 6], [3, 6], [3, 4], [2, 4]]  # >100%, 16px


def feature_align(raw_feature, masks, counts):
    """
    Perform feature align from the raw feature map.
    :param raw_feature: raw feature map
    :param P: point set containing point coordinates
    :param ori_size: size of the original image
    :return: out
    """
    all_nodes = []

    for idx, (count, mask) in enumerate(zip(counts, masks)):
        node_feature = (raw_feature * mask.to(raw_feature)).sum(dim=2).sum(dim=2, keepdim=True) / count

        all_nodes.append(node_feature)

    all_nodes = torch.cat(all_nodes, dim=2)

    return all_nodes


def get_node_box(fs, ori_size, boxes):
    scale = ori_size / fs
    rescaled_boxes = torch.ceil((boxes - scale / 2) / scale).int()
    rescaled_boxes = torch.clamp(rescaled_boxes, 0, fs - 1).to(torch.int32)
    return rescaled_boxes


def size_interpolate(image, size):
    _, _, h, w = image.shape
    max_h, max_w = size

    if np.floor(w * (max_h / h)) <= max_w:
        w = max(1, int(w * (max_h / h)))
        out = F.interpolate(image, size=[max_h, w])
    else:
        h = max(1, int(h * (max_w / w)))
        out = F.interpolate(image, size=[h, max_w])
    return out


def size_padding(image, size):
    _, _, h, w = image.shape
    gap1 = size[0] - h  # height
    gap2 = size[1] - w  # width
    ph1 = gap1 // 2
    ph2 = gap1 - ph1
    pw1 = gap2 // 2
    pw2 = gap2 - pw1
    # print(h,w,size[0],size[1],pw1,pw2,ph1,ph2)
    resized = F.pad(image, [pw1, pw2, ph1, ph2])  # left, right, up, down
    return resized


def get_node_feats(raw_feature, bbox, layers):
    all_nodes = []

    for idx in range(bbox.shape[1]):
        # print(box)
        if idx == 4 or idx == 5:
            layer = layers[3]
        elif idx == 2 or idx == 3:
            layer = layers[2]
        else:
            layer = layers[idx]
        node_features = []
        for batch_idx in range(raw_feature.shape[0]):

            box = bbox[batch_idx, idx]

            node_feature = raw_feature[batch_idx:batch_idx + 1, :, box[0]:box[2], box[1]:box[3]]
            maxh, maxw = HIST_BOX[idx]
            fh, fw = node_feature.size(2), node_feature.size(3)

            # print(idx, fh, fw)

            if fh <= maxh and fw <= maxw:
                node_feature = size_padding(node_feature, HIST_BOX[idx])
            else:
                if fh > maxh and fw <= maxw:
                    # if idx == 0:
                    #     print(fh, fw)
                    fw = max(1, int(fw * (maxh / fh)))
                    node_feature = F.interpolate(node_feature, size=[maxh, fw])
                    # if idx == 0:
                    #     print(node_feature.shape)
                elif fh <= maxh and fw > maxw:
                    # if idx == 0:
                    #     print(fh, fw)
                    fh = max(1, int(fh * (maxw / fw)))
                    node_feature = F.interpolate(node_feature, size=[fh, maxw])
                    # if idx == 0:
                    #     print(node_feature.shape)
                else:
                    node_feature = size_interpolate(node_feature, HIST_BOX[idx])
                node_feature = size_padding(node_feature, HIST_BOX[idx])

            # print(idx, node_feature.shape)
            node_features.append(node_feature)

        node_features = torch.cat(node_features, dim=0).flatten(1)
        node_features = layer(node_features).unsqueeze(1)
        all_nodes.append(node_features)

    return torch.cat(all_nodes, dim=1)
