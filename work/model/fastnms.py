#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-06-03 15:35:27
#   Description : keras_yolov4
#
# ================================================================
import paddle.fluid.layers as P


def _iou(box_a, box_b):
    '''
    :param box_a:    [c, A, 4]
    :param box_b:    [c, B, 4]
    :return:   [c, A, B]  两两之间的iou
    '''
    # 变成左上角坐标、右下角坐标
    boxes1 = P.concat([box_a[:, :, :2] - box_a[:, :, 2:] * 0.5,
                       box_a[:, :, :2] + box_a[:, :, 2:] * 0.5], axis=-1)
    boxes2 = P.concat([box_b[:, :, :2] - box_b[:, :, 2:] * 0.5,
                       box_b[:, :, :2] + box_b[:, :, 2:] * 0.5], axis=-1)

    c = P.shape(boxes1)[0]
    A = P.shape(boxes1)[1]
    B = P.shape(boxes2)[1]

    box_a = P.reshape(boxes1, (c, A, 1, 4))
    box_b = P.reshape(boxes2, (c, 1, B, 4))
    expand_box_a = P.expand(box_a, [1, 1, B, 1])
    expand_box_b = P.expand(box_b, [1, A, 1, 1])

    # 两个矩形的面积
    boxes1_area = (expand_box_a[:, :, :, 2] - expand_box_a[:, :, :, 0]) * \
                  (expand_box_a[:, :, :, 3] - expand_box_a[:, :, :, 1])
    boxes2_area = (expand_box_b[:, :, :, 2] - expand_box_b[:, :, :, 0]) * \
                  (expand_box_b[:, :, :, 3] - expand_box_b[:, :, :, 1])

    # 相交矩形的左上角坐标、右下角坐标
    left_up = P.elementwise_max(expand_box_a[:, :, :, :2], expand_box_b[:, :, :, :2])
    right_down = P.elementwise_min(expand_box_a[:, :, :, 2:], expand_box_b[:, :, :, 2:])

    # 相交矩形的面积inter_area。iou
    # inter_section = P.elementwise_max(right_down - left_up, 0.0)
    inter_section = P.relu(right_down - left_up)
    inter_area = inter_section[:, :, :, 0] * inter_section[:, :, :, 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / (union_area + 1e-9)
    return iou

def fast_nms(boxes, scores, conf_thresh, nms_thresh, keep_top_k, nms_top_k):
    '''
    :param boxes:    [?, 4]
    :param scores:   [80, ?]
    '''

    # 同类方框根据得分降序排列
    scores, idx = P.argsort(scores, axis=1, descending=True)

    idx = idx[:, :keep_top_k]
    scores = scores[:, :keep_top_k]

    num_classes, num_dets = P.shape(idx)[0], P.shape(idx)[1]

    idx = P.reshape(idx, (-1, ))
    boxes = P.gather(boxes, idx)
    boxes = P.reshape(boxes, (num_classes, num_dets, 4))

    # 计算一个c×n×n的IOU矩阵，其中每个n×n矩阵表示对该类n个候选框，两两之间的IOU
    iou = _iou(boxes, boxes)

    # 因为自己与自己的IOU=1，IOU(A,B)=IOU(B,A)，所以对上一步得到的IOU矩阵
    # 进行一次处理。具体做法是将每一个通道，的对角线元素和下三角部分置为0
    rows = P.range(0, num_dets, 1, 'int32')
    cols = P.range(0, num_dets, 1, 'int32')
    rows = P.expand(P.reshape(rows, (1, -1)), [num_dets, 1])
    cols = P.expand(P.reshape(cols, (-1, 1)), [1, num_dets])
    tri_mask = P.cast(rows > cols, 'float32')
    tri_mask = P.expand(P.reshape(tri_mask, (1, num_dets, num_dets)), [num_classes, 1, 1])
    iou = tri_mask * iou
    iou_max = P.reduce_max(iou, dim=1)

    # 同一类别，n个框与“分数比它高的框”的最高iou超过nms_thresh的话，就丢弃。下标是0的框肯定被保留。
    keep = P.where(iou_max <= nms_thresh)

    # Assign each kept detection to its corresponding class
    classes = P.range(0, num_classes, 1, 'int32')
    classes = P.expand(P.reshape(classes, (-1, 1)), [1, num_dets])
    classes = P.gather_nd(classes, keep)

    boxes = P.gather_nd(boxes, keep)
    scores = P.gather_nd(scores, keep)

    # Only keep the top cfg.max_num_detections highest scores across all classes
    scores, idx = P.argsort(scores, axis=0, descending=True)
    idx = idx[:nms_top_k]
    scores = scores[:nms_top_k]

    classes = P.gather(classes, idx)
    boxes = P.gather(boxes, idx)

    return boxes, scores, classes

def fastnms(all_pred_boxes, all_pred_scores, resize_shape,
            origin_shape, conf_thresh, nms_thresh, keep_top_k, nms_top_k, use_yolo_box):
    '''
    :param all_pred_boxes:      [batch_size, -1, 4]
    :param all_pred_scores:     [batch_size, -1, 80]
    :param resize_shape:        [batch_size, 2]
    :param origin_shape:        [batch_size, 2]
    '''
    conf_preds = P.transpose(all_pred_scores, perm=[0, 2, 1])  # [1, 80, -1]
    cur_scores = conf_preds[0]  # [80, -1]
    conf_scores = P.reduce_max(cur_scores, dim=0)  # [-1, ]
    # keep如果是[None]，并且在gather()里使用了keep，就会出现
    # cudaGetLastError  invalid configuration argument errno: 9   这个错误。
    # 为了避免上面的问题，只能让keep不是[None]，所以这里当keep是[None]时给keep赋予一个坐标[[0]]。
    keep = P.where(conf_scores > conf_thresh)
    def exist_objs_1(keep):
        return keep
    def no_objs_1():
        keep_extra = P.zeros((1, 1), 'int64')
        return keep_extra
    keep = P.cond(P.shape(keep)[0] == 0, no_objs_1, lambda: exist_objs_1(keep))
    scores = P.gather(all_pred_scores[0], keep)
    scores = P.transpose(scores, perm=[1, 0])
    boxes = P.gather(all_pred_boxes[0], keep)
    boxes, scores, classes = fast_nms(boxes, scores, conf_thresh, nms_thresh, keep_top_k, nms_top_k)


    # 再做一次分数过滤。前面提到，只要某个框最高分数>阈值就保留，
    # 然而计算上面那个矩阵时，这个框其实重复了80次，每一个分身代表是不同类的物品。
    # 非最高分数的其它类别，它的得分可能小于阈值，要过滤。
    # 所以fastnms存在这么一个现象：某个框它最高分数 > 阈值，它有一个非最高分数类的得分也超过了阈值，
    # 那么最后有可能两个框都保留，而且这两个框有相同的xywh
    keep = P.where(scores > conf_thresh)
    def exist_objs_2(keep, boxes, classes, scores):
        boxes = P.gather(boxes, keep)
        classes = P.gather(classes, keep)
        scores = P.gather(scores, keep)
        return boxes, classes, scores
    def no_objs_2(boxes, classes, scores):
        keep = P.zeros((1, 1), 'int64')
        boxes = P.gather(boxes, keep)
        classes = P.gather(classes, keep)
        scores = P.gather(scores, keep)
        scores -= 2.0  # 巧妙设置为负分数让python端过滤
        return boxes, classes, scores
    boxes, classes, scores = P.cond(P.shape(keep)[0] == 0,
                                           lambda: no_objs_2(boxes, classes, scores),
                                           lambda: exist_objs_2(keep, boxes, classes, scores))
    # 变成左上角坐标、右下角坐标
    boxes = P.concat([boxes[:, :2] - boxes[:, 2:] * 0.5,
                      boxes[:, :2] + boxes[:, 2:] * 0.5], axis=-1)

    # 缩放到原图大小
    if not use_yolo_box:
        resize_shape_f = P.cast(resize_shape, 'float32')
        origin_shape_f = P.cast(origin_shape, 'float32')
        scale = origin_shape_f / resize_shape_f
        scale = P.expand(scale, [1, 3])   # [[h, w, h, w, h, w]]
        boxes *= scale[:, 1:5]   # 批大小是1才支持这么做，因为scale第0维表示批大小，boxes第0维却表示这张图片预测出的物体数

    # 批大小在前
    boxes = P.reshape(boxes, (1, -1, 4), name='boxes')
    scores = P.reshape(scores, (1, -1), name='scores')
    classes = P.reshape(classes, (1, -1), name='classes')
    return [boxes, scores, classes]

