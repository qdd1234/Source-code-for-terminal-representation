#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-06-29 12:22:31
#   Description : numpy后处理
#
# ================================================================
import os
import copy
import numpy as np



# 对坐标解码
# def decode_np(conv_output, anchors, stride, num_class, conf_thresh):
#     conv_shape       = P.shape(conv_output)
#     batch_size       = conv_shape[0]
#     n_grid           = conv_shape[1]
#     anchor_per_scale = len(anchors)
#     conv_output = P.reshape(conv_output, (batch_size, n_grid, n_grid, anchor_per_scale, 5 + num_class))
#     conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
#     conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
#     conv_raw_conf = conv_output[:, :, :, :, 4:5]
#     conv_raw_prob = conv_output[:, :, :, :, 5:]
#
#     rows = P.range(0, n_grid, 1, 'float32')
#     cols = P.range(0, n_grid, 1, 'float32')
#     rows = P.expand(P.reshape(rows, (1, -1, 1)), [n_grid, 1, 1])
#     cols = P.expand(P.reshape(cols, (-1, 1, 1)), [1, n_grid, 1])
#     offset = P.concat([rows, cols], axis=-1)
#     offset = P.reshape(offset, (1, n_grid, n_grid, 1, 2))
#     offset = P.expand(offset, [batch_size, 1, 1, anchor_per_scale, 1])
#
#     pred_xy = (P.sigmoid(conv_raw_dxdy) + offset) * stride
#     pred_wh = (P.exp(conv_raw_dwdh) * P.assign(anchors))
#     pred_xywh = P.concat([pred_xy, pred_wh], axis=-1)
#     pred_conf = P.sigmoid(conv_raw_conf)
#     pred_prob = P.sigmoid(conv_raw_prob)
#
#     pred_xywh = P.reshape(pred_xywh, (batch_size, -1, 4))  # [-1, -1, 4]
#     pred_conf = P.reshape(pred_conf, (batch_size, -1, 1))  # [-1, -1, 1]
#     pred_prob = P.reshape(pred_prob, (batch_size, -1, num_class))  # [-1, -1, 80]
#     return pred_xywh, pred_conf, pred_prob



def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _process_feats(out, anchors, mask, input_shape):
    grid_h, grid_w, num_boxes = map(int, out.shape[1: 4])

    anchors = [anchors[i] for i in mask]
    anchors_tensor = np.array(anchors).reshape(1, 1, len(anchors), 2)

    # Reshape to batch, height, width, num_anchors, box_params.
    out = out[0]
    box_xy = _sigmoid(out[..., :2])
    box_wh = np.exp(out[..., 2:4])
    box_wh = box_wh * anchors_tensor

    box_confidence = _sigmoid(out[..., 4])
    box_confidence = np.expand_dims(box_confidence, axis=-1)
    box_class_probs = _sigmoid(out[..., 5:])

    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)

    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)

    box_xy += grid
    box_xy /= (grid_w, grid_h)
    box_wh /= input_shape
    box_xy -= (box_wh / 2.)  # 坐标格式是左上角xy加矩形宽高wh，xywh都除以图片边长归一化了。
    boxes = np.concatenate((box_xy, box_wh), axis=-1)

    return boxes, box_confidence, box_class_probs


def _filter_boxes(boxes, box_confidences, box_class_probs, _t1):
    box_scores = box_confidences * box_class_probs
    box_classes = np.argmax(box_scores, axis=-1)
    box_class_scores = np.max(box_scores, axis=-1)
    pos = np.where(box_class_scores >= _t1)

    boxes = boxes[pos]
    classes = box_classes[pos]
    scores = box_class_scores[pos]

    return boxes, classes, scores


def _nms_boxes(boxes, scores):
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 1)
        h1 = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        _t2 = 0.45
        inds = np.where(ovr <= _t2)[0]
        order = order[inds + 1]

    keep = np.array(keep)

    return keep


def _yolo_out(outs, shape, input_shape, _t1):
    masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    anchors = [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55],
               [72, 146], [142, 110], [192, 243], [459, 401]]

    boxes, classes, scores = [], [], []

    for out, mask in zip(outs, masks):
        b, c, s = _process_feats(out, anchors, mask, input_shape)
        b, c, s = _filter_boxes(b, c, s, _t1)
        boxes.append(b)
        classes.append(c)
        scores.append(s)

    boxes = np.concatenate(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)

    # boxes坐标格式是左上角xy加矩形宽高wh，xywh都除以图片边长归一化了。
    # Scale boxes back to original image shape.
    w, h = shape[1], shape[0]
    image_dims = [w, h, w, h]
    boxes = boxes * image_dims

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        keep = _nms_boxes(b, s)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])

    if not nclasses and not nscores:
        boxes = np.zeros((1, 4), 'float32')
        scores = np.zeros((1, ), 'float32') - 2.0
        classes = np.zeros((1, ), 'float32')
        # return None, None, None
        return boxes, scores, classes

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    # 换坐标
    boxes[:, [2, 3]] = boxes[:, [0, 1]] + boxes[:, [2, 3]]

    return boxes, scores, classes

