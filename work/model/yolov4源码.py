#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-06-10 10:20:27
#   Description : paddlepaddle_yolov4
#
# ================================================================
import paddle.fluid as fluid
import paddle.fluid.layers as P
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay
import numpy as np

from model.fastnms import fastnms


def _softplus(input):
    expf = fluid.layers.exp(fluid.layers.clip(input, -200, 50))
    return fluid.layers.log(1 + expf)


def _mish(input):
    return input * fluid.layers.tanh(_softplus(input))


def conv2d_unit(x, filters, kernels, stride=1, padding=0, bn=1, act='mish', name='', is_test=False, trainable=True):
    use_bias = (bn != 1)
    bias_attr = False
    if use_bias:
        bias_attr = ParamAttr(initializer=fluid.initializer.Constant(0.0), name=name + ".conv.bias", trainable=trainable)
    x = fluid.layers.conv2d(
        input=x,
        num_filters=filters,
        filter_size=kernels,
        stride=stride,
        padding=padding,
        act=None,
        param_attr=ParamAttr(initializer=fluid.initializer.Normal(0.0, 0.01), name=name + ".conv.weights", trainable=trainable),
        bias_attr=bias_attr)

    if bn:
        bn_name = name + ".bn"
        if not trainable:  # 冻结层时（即trainable=False），bn的均值、标准差也还是会变化，只有设置is_test=True才保证不变
            is_test = True
        x = fluid.layers.batch_norm(
            input=x,
            act=None,
            is_test=is_test,
            param_attr=ParamAttr(
                initializer=fluid.initializer.Constant(1.0),
                regularizer=L2Decay(0.),
                trainable=trainable,
                name=bn_name + '.scale'),
            bias_attr=ParamAttr(
                initializer=fluid.initializer.Constant(0.0),
                regularizer=L2Decay(0.),
                trainable=trainable,
                name=bn_name + '.offset'),
            moving_mean_name=bn_name + '.mean',
            moving_variance_name=bn_name + '.var')
    if act == 'leaky':
        x = fluid.layers.leaky_relu(x, alpha=0.1)
    elif act == 'mish':
        x = _mish(x)
    return x


def residual_block(inputs, filters_1, filters_2, conv_start_idx, is_test, trainable):
    x = conv2d_unit(inputs, filters_1, 1, stride=1, padding=0, name='conv%.3d' % conv_start_idx, is_test=is_test, trainable=trainable)
    x = conv2d_unit(x, filters_2, 3, stride=1, padding=1, name='conv%.3d' % (conv_start_idx + 1), is_test=is_test, trainable=trainable)
    x = fluid.layers.elementwise_add(x=inputs, y=x, act=None)
    return x


def stack_residual_block(inputs, filters_1, filters_2, n, conv_start_idx, is_test, trainable):
    x = residual_block(inputs, filters_1, filters_2, conv_start_idx, is_test, trainable)
    for i in range(n - 1):
        x = residual_block(x, filters_1, filters_2, conv_start_idx + 2 * (1 + i), is_test, trainable)
    return x


def _spp(x):
    x_1 = x
    x_2 = fluid.layers.pool2d(
        input=x,
        pool_size=5,
        pool_type='max',
        pool_stride=1,
        pool_padding=2,
        ceil_mode=True)
    x_3 = fluid.layers.pool2d(
        input=x,
        pool_size=9,
        pool_type='max',
        pool_stride=1,
        pool_padding=4,
        ceil_mode=True)
    x_4 = fluid.layers.pool2d(
        input=x,
        pool_size=13,
        pool_type='max',
        pool_stride=1,
        pool_padding=6,
        ceil_mode=True)
    out = fluid.layers.concat(input=[x_4, x_3, x_2, x_1], axis=1)
    return out


# 对坐标解码
def decode(conv_output, anchors, stride, num_class, conf_thresh):
    conv_shape       = P.shape(conv_output)
    batch_size       = conv_shape[0]
    n_grid           = conv_shape[1]
    anchor_per_scale = len(anchors)
    conv_output = P.reshape(conv_output, (batch_size, n_grid, n_grid, anchor_per_scale, 5 + num_class))
    conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
    conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
    conv_raw_conf = conv_output[:, :, :, :, 4:5]
    conv_raw_prob = conv_output[:, :, :, :, 5:]

    rows = P.range(0, n_grid, 1, 'float32')
    cols = P.range(0, n_grid, 1, 'float32')
    rows = P.expand(P.reshape(rows, (1, -1, 1)), [n_grid, 1, 1])
    cols = P.expand(P.reshape(cols, (-1, 1, 1)), [1, n_grid, 1])
    offset = P.concat([rows, cols], axis=-1)
    offset = P.reshape(offset, (1, n_grid, n_grid, 1, 2))
    offset = P.expand(offset, [batch_size, 1, 1, anchor_per_scale, 1])

    pred_xy = (P.sigmoid(conv_raw_dxdy) + offset) * stride
    pred_wh = (P.exp(conv_raw_dwdh) * P.assign(anchors))
    pred_xywh = P.concat([pred_xy, pred_wh], axis=-1)
    pred_conf = P.sigmoid(conv_raw_conf)
    pred_prob = P.sigmoid(conv_raw_prob)

    pred_xywh = P.reshape(pred_xywh, (batch_size, -1, 4))  # [-1, -1, 4]
    pred_conf = P.reshape(pred_conf, (batch_size, -1, 1))  # [-1, -1, 1]
    pred_prob = P.reshape(pred_prob, (batch_size, -1, num_class))  # [-1, -1, 80]
    return pred_xywh, pred_conf, pred_prob


def yolo_decode(output_l, output_m, output_s, postprocess=None, param=None):
    if postprocess == 'fastnms':
        resize_shape = param['resize_shape']
        origin_shape = param['origin_shape']
        anchors = param['anchors']
        conf_thresh = param['conf_thresh']
        nms_thresh = param['nms_thresh']
        keep_top_k = param['keep_top_k']
        nms_top_k = param['nms_top_k']
        num_classes = param['num_classes']
        num_anchors = param['num_anchors']

        use_yolo_box = False
        use_yolo_box = True

        # 先对坐标解码
        # 第一种方式。慢一点，但支持修改。
        if not use_yolo_box:
            # 相当于numpy的transpose()，交换下标
            output_l = fluid.layers.transpose(output_l, perm=[0, 2, 3, 1], name='output_l')
            output_m = fluid.layers.transpose(output_m, perm=[0, 2, 3, 1], name='output_m')
            output_s = fluid.layers.transpose(output_s, perm=[0, 2, 3, 1], name='output_s')
            pred_xywh_s, pred_conf_s, pred_prob_s = decode(output_s, anchors[0], 8, num_classes, conf_thresh)
            pred_xywh_m, pred_conf_m, pred_prob_m = decode(output_m, anchors[1], 16, num_classes, conf_thresh)
            pred_xywh_l, pred_conf_l, pred_prob_l = decode(output_l, anchors[2], 32, num_classes, conf_thresh)
            # 获取分数。可以不用将pred_conf_s第2维重复80次，paddle支持直接相乘。
            pred_score_s = pred_conf_s * pred_prob_s
            pred_score_m = pred_conf_m * pred_prob_m
            pred_score_l = pred_conf_l * pred_prob_l
            # 所有输出层的预测框集合后再执行nms
            all_pred_boxes = P.concat([pred_xywh_s, pred_xywh_m, pred_xywh_l], axis=1)  # [batch_size, -1, 4]
            all_pred_scores = P.concat([pred_score_s, pred_score_m, pred_score_l], axis=1)  # [batch_size, -1, 80]

        # 第二种方式。用官方yolo_box()函数快一点
        if use_yolo_box:
            anchors = anchors.astype(np.int32)
            anchors = np.reshape(anchors, (-1, num_anchors * 2))
            anchors = anchors.tolist()
            # [bz, ?1, 4]  [bz, ?1, 80]   注意，是过滤置信度位小于conf_thresh的，而不是过滤最终分数！
            bbox_l, prob_l = fluid.layers.yolo_box(
                x=output_l,
                img_size=origin_shape,
                anchors=anchors[2],
                class_num=num_classes,
                conf_thresh=conf_thresh,
                downsample_ratio=32,
                clip_bbox=False)
            bbox_m, prob_m = fluid.layers.yolo_box(
                x=output_m,
                img_size=origin_shape,
                anchors=anchors[1],
                class_num=num_classes,
                conf_thresh=conf_thresh,
                downsample_ratio=16,
                clip_bbox=False)
            bbox_s, prob_s = fluid.layers.yolo_box(
                x=output_s,
                img_size=origin_shape,
                anchors=anchors[0],
                class_num=num_classes,
                conf_thresh=conf_thresh,
                downsample_ratio=8,
                clip_bbox=False)
            boxes = []
            scores = []
            boxes.append(bbox_l)
            boxes.append(bbox_m)
            boxes.append(bbox_s)
            scores.append(prob_l)
            scores.append(prob_m)
            scores.append(prob_s)
            all_pred_boxes = fluid.layers.concat(boxes, axis=1)  # [batch_size, -1, 4]
            all_pred_scores = fluid.layers.concat(scores, axis=1)  # [batch_size, -1, 80]
            # 把x0y0x1y1格式转换成cx_cy_w_h格式
            all_pred_boxes = P.concat([(all_pred_boxes[:, :, :2] + all_pred_boxes[:, :, 2:]) * 0.5,
                                       all_pred_boxes[:, :, 2:] - all_pred_boxes[:, :, :2]], axis=-1)
        # 官方的multiclass_nms()也更快一点。但是为了之后的深度定制。
        # 用fastnms
        boxes, scores, classes = fastnms(all_pred_boxes, all_pred_scores, resize_shape, origin_shape, conf_thresh,
                                         nms_thresh, keep_top_k, nms_top_k, use_yolo_box)
        return boxes, scores, classes
    elif 'multiclass_nms':
        origin_shape = param['origin_shape']
        anchors = param['anchors']
        conf_thresh = param['conf_thresh']
        nms_thresh = param['nms_thresh']
        keep_top_k = param['keep_top_k']
        nms_top_k = param['nms_top_k']
        num_classes = param['num_classes']
        num_anchors = param['num_anchors']

        anchors = anchors.astype(np.int32)
        anchors = np.reshape(anchors, (-1, num_anchors * 2))
        anchors = anchors.tolist()
        # [bz, ?1, 4]  [bz, ?1, 80]   注意，是过滤置信度位小于conf_thresh的，而不是过滤最终分数！
        bbox_l, prob_l = fluid.layers.yolo_box(
            x=output_l,
            img_size=origin_shape,
            anchors=anchors[2],
            class_num=num_classes,
            conf_thresh=conf_thresh,
            downsample_ratio=32,
            clip_bbox=False)
        bbox_m, prob_m = fluid.layers.yolo_box(
            x=output_m,
            img_size=origin_shape,
            anchors=anchors[1],
            class_num=num_classes,
            conf_thresh=conf_thresh,
            downsample_ratio=16,
            clip_bbox=False)
        bbox_s, prob_s = fluid.layers.yolo_box(
            x=output_s,
            img_size=origin_shape,
            anchors=anchors[0],
            class_num=num_classes,
            conf_thresh=conf_thresh,
            downsample_ratio=8,
            clip_bbox=False)
        boxes = []
        scores = []
        boxes.append(bbox_l)
        boxes.append(bbox_m)
        boxes.append(bbox_s)
        scores.append(prob_l)
        scores.append(prob_m)
        scores.append(prob_s)
        all_pred_boxes = fluid.layers.concat(boxes, axis=1)  # [batch_size, -1, 4]
        all_pred_scores = fluid.layers.concat(scores, axis=1)  # [batch_size, -1, 80]
        all_pred_scores = fluid.layers.transpose(all_pred_scores, perm=[0, 2, 1])
        pred = fluid.layers.multiclass_nms(all_pred_boxes, all_pred_scores,
                                           score_threshold=conf_thresh,
                                           nms_top_k=nms_top_k,
                                           keep_top_k=keep_top_k,
                                           nms_threshold=nms_thresh,
                                           background_label=-1)  # 对于YOLO算法，一定要设置background_label=-1，否则检测不出人。
        return pred


def YOLOv4(inputs, num_classes, num_anchors, initial_filters=32, is_test=False, trainable=True,
           export=False, postprocess=None, param=None):
    i32 = initial_filters
    i64 = i32 * 2
    i128 = i32 * 4
    i256 = i32 * 8
    i512 = i32 * 16
    i1024 = i32 * 32

    # cspdarknet53部分
    x = conv2d_unit(inputs, i32, 3, stride=1, padding=1, name='conv001', is_test=is_test, trainable=trainable)

    # ============================= s2 =============================
    x = conv2d_unit(x, i64, 3, stride=2, padding=1, name='conv002', is_test=is_test, trainable=trainable)
    s2 = conv2d_unit(x, i64, 1, stride=1, name='conv003', is_test=is_test, trainable=trainable)
    x = conv2d_unit(x, i64, 1, stride=1, name='conv004', is_test=is_test, trainable=trainable)
    x = stack_residual_block(x, i32, i64, n=1, conv_start_idx=5, is_test=is_test, trainable=trainable)
    x = conv2d_unit(x, i64, 1, stride=1, name='conv007', is_test=is_test, trainable=trainable)
    x = fluid.layers.concat([x, s2], axis=1)
    s2 = conv2d_unit(x, i64, 1, stride=1, name='conv008', is_test=is_test, trainable=trainable)

    # ============================= s4 =============================
    x = conv2d_unit(s2, i128, 3, stride=2, padding=1, name='conv009', is_test=is_test, trainable=trainable)
    s4 = conv2d_unit(x, i64, 1, stride=1, name='conv010', is_test=is_test, trainable=trainable)
    x = conv2d_unit(x, i64, 1, stride=1, name='conv011', is_test=is_test, trainable=trainable)
    x = stack_residual_block(x, i64, i64, n=2, conv_start_idx=12, is_test=is_test, trainable=trainable)
    x = conv2d_unit(x, i64, 1, stride=1, name='conv016', is_test=is_test, trainable=trainable)
    x = fluid.layers.concat([x, s4], axis=1)
    s4 = conv2d_unit(x, i128, 1, stride=1, name='conv017', is_test=is_test, trainable=trainable)

    # ============================= s8 =============================
    x = conv2d_unit(s4, i256, 3, stride=2, padding=1, name='conv018', is_test=is_test, trainable=trainable)
    s8 = conv2d_unit(x, i128, 1, stride=1, name='conv019', is_test=is_test, trainable=trainable)
    x = conv2d_unit(x, i128, 1, stride=1, name='conv020', is_test=is_test, trainable=trainable)
    x = stack_residual_block(x, i128, i128, n=8, conv_start_idx=21, is_test=is_test, trainable=trainable)
    x = conv2d_unit(x, i128, 1, stride=1, name='conv037', is_test=is_test, trainable=trainable)
    x = fluid.layers.concat([x, s8], axis=1)
    s8 = conv2d_unit(x, i256, 1, stride=1, name='conv038', is_test=is_test, trainable=trainable)

    # ============================= s16 =============================
    x = conv2d_unit(s8, i512, 3, stride=2, padding=1, name='conv039', is_test=is_test, trainable=trainable)
    s16 = conv2d_unit(x, i256, 1, stride=1, name='conv040', is_test=is_test, trainable=trainable)
    x = conv2d_unit(x, i256, 1, stride=1, name='conv041', is_test=is_test, trainable=trainable)
    x = stack_residual_block(x, i256, i256, n=8, conv_start_idx=42, is_test=is_test, trainable=trainable)
    x = conv2d_unit(x, i256, 1, stride=1, name='conv058', is_test=is_test, trainable=trainable)
    x = fluid.layers.concat([x, s16], axis=1)
    s16 = conv2d_unit(x, i512, 1, stride=1, name='conv059', is_test=is_test, trainable=trainable)

    # ============================= s32 =============================
    x = conv2d_unit(s16, i1024, 3, stride=2, padding=1, name='conv060', is_test=is_test, trainable=trainable)
    s32 = conv2d_unit(x, i512, 1, stride=1, name='conv061', is_test=is_test, trainable=trainable)
    x = conv2d_unit(x, i512, 1, stride=1, name='conv062', is_test=is_test, trainable=trainable)
    x = stack_residual_block(x, i512, i512, n=4, conv_start_idx=63, is_test=is_test, trainable=trainable)
    x = conv2d_unit(x, i512, 1, stride=1, name='conv071', is_test=is_test, trainable=trainable)
    x = fluid.layers.concat([x, s32], axis=1)
    s32 = conv2d_unit(x, i1024, 1, stride=1, name='conv072', is_test=is_test, trainable=trainable)
    # cspdarknet53部分结束

    # fpn部分
    x = conv2d_unit(s32, i512, 1, stride=1, act='leaky', name='conv073', is_test=is_test, trainable=trainable)
    x = conv2d_unit(x, i1024, 3, stride=1, padding=1, act='leaky', name='conv074', is_test=is_test, trainable=trainable)
    x = conv2d_unit(x, i512, 1, stride=1, act='leaky', name='conv075', is_test=is_test, trainable=trainable)
    x = _spp(x)

    x = conv2d_unit(x, i512, 1, stride=1, act='leaky', name='conv076', is_test=is_test, trainable=trainable)
    x = conv2d_unit(x, i1024, 3, stride=1, padding=1, act='leaky', name='conv077', is_test=is_test, trainable=trainable)
    fpn_s32 = conv2d_unit(x, i512, 1, stride=1, act='leaky', name='conv078', is_test=is_test, trainable=trainable)

    # pan01
    x = conv2d_unit(fpn_s32, i256, 1, stride=1, act='leaky', name='conv079', is_test=is_test, trainable=trainable)
    x = fluid.layers.resize_nearest(x, scale=float(2))
    s16 = conv2d_unit(s16, i256, 1, stride=1, act='leaky', name='conv080', is_test=is_test, trainable=trainable)
    x = fluid.layers.concat([s16, x], axis=1)
    x = conv2d_unit(x, i256, 1, stride=1, act='leaky', name='conv081', is_test=is_test, trainable=trainable)
    x = conv2d_unit(x, i512, 3, stride=1, padding=1, act='leaky', name='conv082', is_test=is_test, trainable=trainable)
    x = conv2d_unit(x, i256, 1, stride=1, act='leaky', name='conv083', is_test=is_test, trainable=trainable)
    x = conv2d_unit(x, i512, 3, stride=1, padding=1, act='leaky', name='conv084', is_test=is_test, trainable=trainable)
    fpn_s16 = conv2d_unit(x, i256, 1, stride=1, act='leaky', name='conv085', is_test=is_test, trainable=trainable)
    # pan01结束

    # pan02
    x = conv2d_unit(fpn_s16, i128, 1, stride=1, act='leaky', name='conv086', is_test=is_test, trainable=trainable)
    x = fluid.layers.resize_nearest(x, scale=float(2))
    s8 = conv2d_unit(s8, i128, 1, stride=1, act='leaky', name='conv087', is_test=is_test, trainable=trainable)
    x = fluid.layers.concat([s8, x], axis=1)
    x = conv2d_unit(x, i128, 1, stride=1, act='leaky', name='conv088', is_test=is_test, trainable=trainable)
    x = conv2d_unit(x, i256, 3, stride=1, padding=1, act='leaky', name='conv089', is_test=is_test, trainable=trainable)
    x = conv2d_unit(x, i128, 1, stride=1, act='leaky', name='conv090', is_test=is_test, trainable=trainable)
    x = conv2d_unit(x, i256, 3, stride=1, padding=1, act='leaky', name='conv091', is_test=is_test, trainable=trainable)
    x = conv2d_unit(x, i128, 1, stride=1, act='leaky', name='conv092', is_test=is_test, trainable=trainable)
    # pan02结束

    # output_s, 不用concat()
    output_s = conv2d_unit(x, i256, 3, stride=1, padding=1, act='leaky', name='conv093', is_test=is_test,
                           trainable=trainable)
    output_s = conv2d_unit(output_s, num_anchors * (num_classes + 5), 1, stride=1, bn=0, act=None, name='conv094',
                           is_test=is_test, trainable=trainable)

    # output_m, 需要concat()
    x = conv2d_unit(x, i256, 3, stride=2, padding=1, act='leaky', name='conv095', is_test=is_test, trainable=trainable)
    x = fluid.layers.concat([x, fpn_s16], axis=1)
    x = conv2d_unit(x, i256, 1, stride=1, act='leaky', name='conv096', is_test=is_test, trainable=trainable)
    x = conv2d_unit(x, i512, 3, stride=1, padding=1, act='leaky', name='conv097', is_test=is_test, trainable=trainable)
    x = conv2d_unit(x, i256, 1, stride=1, act='leaky', name='conv098', is_test=is_test, trainable=trainable)
    x = conv2d_unit(x, i512, 3, stride=1, padding=1, act='leaky', name='conv099', is_test=is_test, trainable=trainable)
    x = conv2d_unit(x, i256, 1, stride=1, act='leaky', name='conv100', is_test=is_test, trainable=trainable)
    output_m = conv2d_unit(x, i512, 3, stride=1, padding=1, act='leaky', name='conv101', is_test=is_test,
                           trainable=trainable)
    output_m = conv2d_unit(output_m, num_anchors * (num_classes + 5), 1, stride=1, bn=0, act=None, name='conv102',
                           is_test=is_test, trainable=trainable)

    # output_l, 需要concat()
    x = conv2d_unit(x, i512, 3, stride=2, padding=1, act='leaky', name='conv103', is_test=is_test, trainable=trainable)
    x = fluid.layers.concat([x, fpn_s32], axis=1)
    x = conv2d_unit(x, i512, 1, stride=1, act='leaky', name='conv104', is_test=is_test, trainable=trainable)
    x = conv2d_unit(x, i1024, 3, stride=1, padding=1, act='leaky', name='conv105', is_test=is_test, trainable=trainable)
    x = conv2d_unit(x, i512, 1, stride=1, act='leaky', name='conv106', is_test=is_test, trainable=trainable)
    x = conv2d_unit(x, i1024, 3, stride=1, padding=1, act='leaky', name='conv107', is_test=is_test, trainable=trainable)
    x = conv2d_unit(x, i512, 1, stride=1, act='leaky', name='conv108', is_test=is_test, trainable=trainable)
    output_l = conv2d_unit(x, i1024, 3, stride=1, padding=1, act='leaky', name='conv109', is_test=is_test,
                           trainable=trainable)
    output_l = conv2d_unit(output_l, num_anchors * (num_classes + 5), 1, stride=1, bn=0, act=None, name='conv110',
                           is_test=is_test, trainable=trainable)

    if export:
        return yolo_decode(output_l, output_m, output_s, postprocess, param)

    # 相当于numpy的transpose()，交换下标
    output_l = fluid.layers.transpose(output_l, perm=[0, 2, 3, 1], name='output_l')
    output_m = fluid.layers.transpose(output_m, perm=[0, 2, 3, 1], name='output_m')
    output_s = fluid.layers.transpose(output_s, perm=[0, 2, 3, 1], name='output_s')
    return output_l, output_m, output_s



