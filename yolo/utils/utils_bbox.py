import numpy as np
import torch
import pkg_resources as pkg


def check_version(current: str = "0.0.0",
                  minimum: str = "0.0.0",
                  name: str = "version ",
                  pinned: bool = False) -> bool:
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    return result


TORCH_1_10 = check_version(torch.__version__, '1.10.0')


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing='ij') if TORCH_1_10 else torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    # 左上右下
    lt, rb = torch.split(distance, 2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def bbox_iou(box1, box2, xywh=True, CIoU=False, eps=1e-7):
    """
    计算IOU/CIoU
    :param box1: 预测框 [B, 4] or [4,]
    :param box2: 真实框 [B, 4] or [4,]
    :param xywh: 输入格式是否为xywh
    :param CIoU: 是否计算CIoU
    :param eps: 防止除零
    :return: IOU/CIoU值
    """
    # 转换为xyxy格式
    if xywh:
        # x1y1 = 中心 - 宽高/2, x2y2 = 中心 + 宽高/2
        box1 = torch.cat([box1[..., :2] - box1[..., 2:] / 2,
                          box1[..., :2] + box1[..., 2:] / 2], dim=-1)
        box2 = torch.cat([box2[..., :2] - box2[..., 2:] / 2,
                          box2[..., :2] + box2[..., 2:] / 2], dim=-1)

    # 计算交集的左上角和右下角
    inter_left_top = torch.max(box1[..., :2], box2[..., :2])
    inter_right_bottom = torch.min(box1[..., 2:], box2[..., 2:])

    # 计算交集面积
    inter_area = torch.clamp(inter_right_bottom - inter_left_top, min=0).prod(-1)
    # 计算并集面积
    box1_area = torch.clamp(box1[..., 2:] - box1[..., :2], min=0).prod(-1)
    box2_area = torch.clamp(box2[..., 2:] - box2[..., :2], min=0).prod(-1)
    union_area = box1_area + box2_area - inter_area + eps
    # 基础IOU
    iou = inter_area / union_area

    if not CIoU:
        return iou

    # 计算CIoU（包含长宽比和中心距离惩罚）
    # 计算最小外接矩形的对角线长度
    enclose_left_top = torch.min(box1[..., :2], box2[..., :2])
    enclose_right_bottom = torch.max(box1[..., 2:], box2[..., 2:])
    enclose_wh = torch.clamp(enclose_right_bottom - enclose_left_top, min=0)
    enclose_diag = torch.pow(enclose_wh[..., 0], 2) + torch.pow(enclose_wh[..., 1], 2) + eps

    # 计算中心距离的平方
    center_distance = torch.pow(box1[..., 0] - box2[..., 0], 2) + torch.pow(box1[..., 1] - box2[..., 1], 2)

    # 计算长宽比一致性
    v = (4 / (np.pi ** 2)) * torch.pow(torch.atan(box1[..., 2] / (box1[..., 3] + eps)) -
                                       torch.atan(box2[..., 2] / (box2[..., 3] + eps)), 2)
    alpha = v / (1 - iou + v + eps)

    # CIoU = IOU - (中心距离/外接矩形对角线) - 长宽比惩罚
    ciou = iou - (center_distance / enclose_diag) - alpha * v
    return ciou


def soft_nms(boxes, scores, iou_threshold=0.5, sigma=0.5, score_threshold=0.001):
    """
    Soft-NMS实现：衰减重叠框的置信度，而非直接删除
    :param boxes: 预测框坐标 [N,4] (xyxy格式)
    :param scores: 预测框置信度 [N]
    :param iou_threshold: IoU衰减阈值
    :param sigma: 高斯衰减系数（0.3~0.7为宜）
    :param score_threshold: 最终保留置信度阈值
    :return: 保留的框索引
    """
    # 复制scores避免修改原数据
    scores = scores.clone()
    keep = []
    # 初始化索引
    indices = torch.arange(0, scores.shape[0], dtype=torch.int32, device=boxes.device)

    while indices.numel() > 0:
        # 取当前置信度最高的框
        idx = indices[scores[indices].argmax()]
        keep.append(idx)
        if indices.numel() == 1:
            break

        # 计算最高分框与剩余框的IoU
        box = boxes[idx].unsqueeze(0)
        rest_boxes = boxes[indices[1:]]
        iou = bbox_iou(box, rest_boxes, xywh=False)

        # 高斯衰减更新剩余框的置信度
        decay = torch.exp(-(iou ** 2) / sigma)
        scores[indices[1:]] *= decay

        # 过滤低置信度框
        mask = scores[indices[1:]] > score_threshold
        indices = indices[1:][mask]

    return torch.tensor(keep, dtype=torch.int64, device=boxes.device)


class DecodeBox():
    def __init__(self, num_classes, input_shape):
        super(DecodeBox, self).__init__()
        self.num_classes = num_classes
        self.bbox_attrs = 4 + num_classes
        self.input_shape = input_shape

    def decode_box(self, inputs):
        # dbox  batch_size, 4, 8400
        # cls   batch_size, 20, 8400
        dbox, cls, origin_cls, anchors, strides = inputs
        # 获得中心宽高坐标
        dbox = dist2bbox(dbox, anchors.unsqueeze(0), xywh=True, dim=1) * strides
        y = torch.cat((dbox, cls.sigmoid()), 1).permute(0, 2, 1)
        # 进行归一化，到0~1之间
        scale_t = torch.tensor(
            [self.input_shape[1], self.input_shape[0], self.input_shape[1], self.input_shape[0]],
            device=y.device, dtype=y.dtype,
        )
        y[:, :, :4] = y[:, :, :4] / scale_t
        return y

    def yolo_correct_boxes(self, box_xy, box_wh, input_shape, image_shape, letterbox_image):
        # -----------------------------------------------------------------#
        #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
        # -----------------------------------------------------------------#
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        if letterbox_image:
            new_shape = np.round(image_shape * np.min(input_shape / image_shape))
            offset = (input_shape - new_shape) / 2. / input_shape
            scale = input_shape / new_shape
            box_yx = (box_yx - offset) * scale
            box_hw *= scale

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]],
                               axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes

    def yolo_correct_boxes_tensor(self, box_xy, box_wh, input_shape, image_shape, letterbox_image):
        """GPU 版：在 tensor 上完成坐标反变换，避免提前 .cpu().numpy() 造成同步。"""
        box_yx = box_xy.flip(-1)
        box_hw = box_wh.flip(-1)
        if not isinstance(input_shape, torch.Tensor):
            input_shape = torch.tensor(input_shape, dtype=box_xy.dtype, device=box_xy.device)
        if not isinstance(image_shape, torch.Tensor):
            image_shape = torch.tensor(image_shape, dtype=box_xy.dtype, device=box_xy.device)

        if letterbox_image:
            scale_ratio = torch.min(input_shape / image_shape)
            new_shape = torch.round(image_shape * scale_ratio)
            offset = (input_shape - new_shape) / 2.0 / input_shape
            scale = input_shape / new_shape
            box_yx = (box_yx - offset) * scale
            box_hw = box_hw * scale

        box_mins = box_yx - (box_hw / 2.0)
        box_maxes = box_yx + (box_hw / 2.0)
        boxes = torch.cat([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], dim=-1)
        boxes = boxes * torch.cat([image_shape, image_shape], dim=-1)
        return boxes

    def non_max_suppression(self, prediction, num_classes, input_shape, image_shape, letterbox_image, conf_thres=0.5,
                            nms_thres=0.4):
        # ----------------------------------------------------------#
        #   将预测结果的格式转换成左上角右下角的格式。
        #   prediction  [batch_size, num_anchors, 85]
        #   image_shape: single array (H,W) or list of arrays for per-image shapes
        # ----------------------------------------------------------#
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        # Support per-image shape for batch inference
        image_shapes = image_shape if isinstance(image_shape, (list, tuple)) else [image_shape] * len(prediction)
        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):
            img_shape = image_shapes[i] if i < len(image_shapes) else image_shapes[0]
            # ----------------------------------------------------------#
            #   对种类预测部分取max。
            #   class_conf  [num_anchors, 1]    种类置信度
            #   class_pred  [num_anchors, 1]    种类
            # ----------------------------------------------------------#
            class_conf, class_pred = torch.max(image_pred[:, 4:4 + num_classes], 1, keepdim=True)

            # ----------------------------------------------------------#
            #   利用置信度进行第一轮筛选
            # ----------------------------------------------------------#
            conf_mask = (class_conf[:, 0] >= conf_thres).squeeze()

            # ----------------------------------------------------------#
            #   根据置信度进行预测结果的筛选
            # ----------------------------------------------------------#
            image_pred = image_pred[conf_mask]
            class_conf = class_conf[conf_mask]
            class_pred = class_pred[conf_mask]
            if not image_pred.size(0):
                continue
            # -------------------------------------------------------------------------#
            #   detections  [num_anchors, 6]
            #   6的内容为：x1, y1, x2, y2, class_conf, class_pred
            # -------------------------------------------------------------------------#
            detections = torch.cat((image_pred[:, :4], class_conf.float(), class_pred.float()), 1)

            # ------------------------------------------#
            #   获得预测结果中包含的所有种类（保持 GPU，避免 CPU 往返）
            # ------------------------------------------#
            unique_labels = torch.unique(detections[:, -1])

            for c in unique_labels:
                # ------------------------------------------#
                #   获得某一类得分筛选后全部的预测结果
                # ------------------------------------------#
                detections_class = detections[detections[:, -1] == c]

                # ------------------------------------------#
                #   替换为Soft-NMS（核心修改）
                # ------------------------------------------#
                keep = soft_nms(
                    boxes=detections_class[:, :4],
                    scores=detections_class[:, 4],
                    iou_threshold=nms_thres,
                    sigma=0.5,  # 高斯衰减系数，可根据需求调整（0.3~0.7）
                    score_threshold=conf_thres
                )
                max_detections = detections_class[keep]

                # Add max detections to outputs
                output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))

            if output[i] is not None:
                out = output[i]
                box_xy = (out[:, 0:2] + out[:, 2:4]) / 2
                box_wh = out[:, 2:4] - out[:, 0:2]
                out[:, :4] = self.yolo_correct_boxes_tensor(
                    box_xy, box_wh, input_shape, img_shape, letterbox_image
                )
                output[i] = out.cpu().numpy()
        return output


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np


    # ---------------------------------------------------#
    #   将预测值的每个特征层调成真实值
    # ---------------------------------------------------#
    def get_anchors_and_decode(input, input_shape, anchors, anchors_mask, num_classes):
        # -----------------------------------------------#
        #   input   batch_size, 3 * (4 + 1 + num_classes), 20, 20
        # -----------------------------------------------#
        batch_size = input.size(0)
        input_height = input.size(2)
        input_width = input.size(3)

        # -----------------------------------------------#
        #   输入为640x640时 input_shape = [640, 640]  input_height = 20, input_width = 20
        #   640 / 20 = 32
        #   stride_h = stride_w = 32
        # -----------------------------------------------#
        stride_h = input_shape[0] / input_height
        stride_w = input_shape[1] / input_width
        # -------------------------------------------------#
        #   此时获得的scaled_anchors大小是相对于特征层的
        #   anchor_width, anchor_height / stride_h, stride_w
        # -------------------------------------------------#
        scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in
                          anchors[anchors_mask[2]]]

        # -----------------------------------------------#
        #   batch_size, 3 * (4 + 1 + num_classes), 20, 20 =>
        #   batch_size, 3, 5 + num_classes, 20, 20  =>
        #   batch_size, 3, 20, 20, 4 + 1 + num_classes
        # -----------------------------------------------#
        prediction = input.view(batch_size, len(anchors_mask[2]),
                                num_classes + 5, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()

        # -----------------------------------------------#
        #   先验框的中心位置的调整参数
        # -----------------------------------------------#
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        # -----------------------------------------------#
        #   先验框的宽高调整参数
        # -----------------------------------------------#
        w = torch.sigmoid(prediction[..., 2])
        h = torch.sigmoid(prediction[..., 3])
        # -----------------------------------------------#
        #   获得置信度，是否有物体 0 - 1
        # -----------------------------------------------#
        conf = torch.sigmoid(prediction[..., 4])
        # -----------------------------------------------#
        #   种类置信度 0 - 1
        # -----------------------------------------------#
        pred_cls = torch.sigmoid(prediction[..., 5:])

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        # ----------------------------------------------------------#
        #   生成网格，先验框中心，网格左上角
        #   batch_size,3,20,20
        #   range(20)
        #   [
        #       [0, 1, 2, 3 ……, 19],
        #       [0, 1, 2, 3 ……, 19],
        #       …… （20次）
        #       [0, 1, 2, 3 ……, 19]
        #   ] * (batch_size * 3)
        #   [batch_size, 3, 20, 20]
        #
        #   [
        #       [0, 1, 2, 3 ……, 19],
        #       [0, 1, 2, 3 ……, 19],
        #       …… （20次）
        #       [0, 1, 2, 3 ……, 19]
        #   ].T * (batch_size * 3)
        #   [batch_size, 3, 20, 20]
        # ----------------------------------------------------------#
        grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
            batch_size * len(anchors_mask[2]), 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
            batch_size * len(anchors_mask[2]), 1, 1).view(y.shape).type(FloatTensor)

        # ----------------------------------------------------------#
        #   按照网格格式生成先验框的宽高
        #   batch_size, 3, 20 * 20 => batch_size, 3, 20, 20
        #   batch_size, 3, 20 * 20 => batch_size, 3, 20, 20
        # ----------------------------------------------------------#
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
        anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)

        # ----------------------------------------------------------#
        #   利用预测结果对先验框进行调整
        #   首先调整先验框的中心，从先验框中心向右下角偏移
        #   再调整先验框的宽高。
        #   x  0 ~ 1 => 0 ~ 2 => -0.5 ~ 1.5 + grid_x
        #   y  0 ~ 1 => 0 ~ 2 => -0.5 ~ 1.5 + grid_y
        #   w  0 ~ 1 => 0 ~ 2 => 0 ~ 4 * anchor_w
        #   h  0 ~ 1 => 0 ~ 2 => 0 ~ 4 * anchor_h
        # ----------------------------------------------------------#
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data * 2. - 0.5 + grid_x
        pred_boxes[..., 1] = y.data * 2. - 0.5 + grid_y
        pred_boxes[..., 2] = (w.data * 2) ** 2 * anchor_w
        pred_boxes[..., 3] = (h.data * 2) ** 2 * anchor_h

        point_h = 5
        point_w = 5

        box_xy = pred_boxes[..., 0:2].cpu().numpy() * 32
        box_wh = pred_boxes[..., 2:4].cpu().numpy() * 32
        grid_x = grid_x.cpu().numpy() * 32
        grid_y = grid_y.cpu().numpy() * 32
        anchor_w = anchor_w.cpu().numpy() * 32
        anchor_h = anchor_h.cpu().numpy() * 32

        # 注释掉图片相关代码（避免缺少图片文件报错）
        # fig = plt.figure()
        # ax  = fig.add_subplot(121)
        # from PIL import Image
        # img = Image.open("img/street.jpg").resize([640, 640])
        # plt.imshow(img, alpha=0.5)
        # plt.ylim(-30, 650)
        # plt.xlim(-30, 650)
        # plt.scatter(grid_x, grid_y)
        # plt.scatter(point_h * 32, point_w * 32, c='black')
        # plt.gca().invert_yaxis()

        anchor_left = grid_x - anchor_w / 2
        anchor_top = grid_y - anchor_h / 2

        # rect1 = plt.Rectangle([anchor_left[0, 0, point_h, point_w],anchor_top[0, 0, point_h, point_w]], \
        #     anchor_w[0, 0, point_h, point_w],anchor_h[0, 0, point_h, point_w],color="r",fill=False)
        # rect2 = plt.Rectangle([anchor_left[0, 1, point_h, point_w],anchor_top[0, 1, point_h, point_w]], \
        #     anchor_w[0, 1, point_h, point_w],anchor_h[0, 1, point_h, point_w],color="r",fill=False)
        # rect3 = plt.Rectangle([anchor_left[0, 2, point_h, point_w],anchor_top[0, 2, point_h, point_w]], \
        #     anchor_w[0, 2, point_h, point_w],anchor_h[0, 2, point_h, point_w],color="r",fill=False)

        # ax.add_patch(rect1)
        # ax.add_patch(rect2)
        # ax.add_patch(rect3)

        # ax  = fig.add_subplot(122)
        # plt.imshow(img, alpha=0.5)
        # plt.ylim(-30, 650)
        # plt.xlim(-30, 650)
        # plt.scatter(grid_x, grid_y)
        # plt.scatter(point_h * 32, point_w * 32, c='black')
        # plt.scatter(box_xy[0, :, point_h, point_w, 0], box_xy[0, :, point_h, point_w, 1], c='r')
        # plt.gca().invert_yaxis()

        # pre_left    = box_xy[...,0] - box_wh[...,0] / 2
        # pre_top     = box_xy[...,1] - box_wh[...,1] / 2

        # rect1 = plt.Rectangle([pre_left[0, 0, point_h, point_w], pre_top[0, 0, point_h, point_w]],\
        #     box_wh[0, 0, point_h, point_w,0], box_wh[0, 0, point_h, point_w,1],color="r",fill=False)
        # rect2 = plt.Rectangle([pre_left[0, 1, point_h, point_w], pre_top[0, 1, point_h, point_w]],\
        #     box_wh[0, 1, point_h, point_w,0], box_wh[0, 1, point_h, point_w,1],color="r",fill=False)
        # rect3 = plt.Rectangle([pre_left[0, 2, point_h, point_w], pre_top[0, 2, point_h, point_w]],\
        #     box_wh[0, 2, point_h, point_w,0], box_wh[0, 2, point_h, point_w,1],color="r",fill=False)

        # ax.add_patch(rect1)
        # ax.add_patch(rect2)
        # ax.add_patch(rect3)

        # plt.show()


    feat = torch.from_numpy(np.random.normal(0.2, 0.5, [4, 255, 20, 20])).float()
    anchors = np.array([[116, 90], [156, 198], [373, 326], [30, 61], [62, 45], [59, 119], [10, 13], [16, 30], [33, 23]])
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    get_anchors_and_decode(feat, [640, 640], anchors, anchors_mask, 80)