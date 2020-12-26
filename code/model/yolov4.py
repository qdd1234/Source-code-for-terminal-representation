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
from model.acfpn import ACFPN

acfpn = ACFPN()

def _softplus(input):
    expf = fluid.layers.exp(fluid.layers.clip(input, -200, 50))
    return fluid.layers.log(1 + expf)


def _mish(input):
    return input * fluid.layers.tanh(_softplus(input))


def conv2d_unit(x, filters, kernels, stride=1, padding=0, bn=1, act='mish',dilation=1, name='', is_test=False, trainable=True):
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
        dilation = dilation,
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

def stack_residual_block(inputs,conv_start_idx, is_test, trainable):
    x =  conv2d_unit(inputs, inputs.shape[1], 3, stride=1, padding=1, name='conv%.3d' % (conv_start_idx), is_test=is_test, trainable=trainable)
    x1,x2 = fluid.layers.split(x, num_or_sections = 2, dim=1)
    _,c,h,w = x2.shape
    x3 = conv2d_unit(x2, c, 3, stride=1, padding=1, name='conv%.3d' % (conv_start_idx + 1), is_test=is_test, trainable=trainable)
    x4 = conv2d_unit(x3, c, 3, stride=1, padding=1, name='conv%.3d' % (conv_start_idx + 2), is_test=is_test, trainable=trainable)
    x5 = fluid.layers.concat(input = [x3,x4],axis=1)
    _,c,h,w = x5.shape
    x6 = conv2d_unit(x5, c, 1, stride=1, padding=0, name='conv%.3d' % (conv_start_idx+3), is_test=is_test, trainable=trainable)
    x = fluid.layers.concat(input = [x1,x2,x6],axis=1)
    return x

def aspp(x):
    x_1 = x
    x_2 = conv2d_unit(x, x.shape[1], 1, stride=0, padding=1, bn = 0,act = 'leaky',name='SPP1')
    x_3 = conv2d_unit(x, x.shape[1], 3, stride=1, padding=3, dilation=3,bn = 0,act = 'leaky',name='SPP2')
    x_4 = conv2d_unit(x, x.shape[1], 6, stride=1, padding=6, dilation=6,bn = 0,act = 'leaky',name='SPP3')
    out = fluid.layers.concat(input=[x_4, x_3, x_2, x_1], axis=1)
    return out


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

    x = conv2d_unit(inputs, i32, 3, stride=1, padding=1, name='conv001', is_test=is_test, trainable=trainable)
    x = conv2d_unit(x, i32, 3, stride=2, padding=1, name='conv002', is_test=is_test, trainable=trainable)
    x = conv2d_unit(x, i32, 1, stride=1, name='conv003', is_test=is_test, trainable=trainable)
    x = conv2d_unit(x, i64, 3, stride=2, padding=1, name='conv004', is_test=is_test, trainable=trainable)
    x = conv2d_unit(x, i64, 1, stride=1, name='conv005', is_test=is_test, trainable=trainable)
    ###第一个block，这时候的特征通道已经变成了2倍的x.shape[0]
    x = conv2d_unit(x, i64, 3, stride=2, padding=1, name='conv0046', is_test=is_test, trainable=trainable)
    x = conv2d_unit(x, i64, 1, stride=1, name='conv0051', is_test=is_test, trainable=trainable)
    x = stack_residual_block(x,conv_start_idx=100, is_test=is_test, trainable=trainable)
    s1 = conv2d_unit(x, i128, 1, stride=1, name='conv007', is_test=is_test, trainable=trainable)

    x = conv2d_unit(s1, i128, 3, stride=2, padding=1, name='conv008', is_test=is_test, trainable=trainable)
    x = conv2d_unit(x, i128, 1, stride=1, name='conv009', is_test=is_test, trainable=trainable)
    #第二个block
    x = stack_residual_block(x,conv_start_idx=111, is_test=is_test, trainable=trainable)
    s2 = conv2d_unit(x, i256, 1, stride=1, name='conv011', is_test=is_test, trainable=trainable)

    x = conv2d_unit(s2, i256, 3, stride=2, padding=1, name='conv012', is_test=is_test, trainable=trainable)
    x = conv2d_unit(x, i256, 1, stride=1, name='conv013', is_test=is_test, trainable=trainable)
    #第三个block

    x = stack_residual_block(x,conv_start_idx=122, is_test=is_test, trainable=trainable)
    s3 = conv2d_unit(x, i512, 1, stride=1, name='conv015', is_test=is_test, trainable=trainable)

    #最后一个卷积
    x = conv2d_unit(s3, i512, 1, stride=1, name='conv016', is_test=is_test, trainable=trainable)

    ######################################FPN结构###########################################################

    x,x1 = fluid.layers.split(x,num_or_sections = 2,dim = 1)
    x = conv2d_unit(x, i256, 3, act = 'leaky',bn = 0,stride=1, padding = 1,name='CSP-SPP_1', is_test=is_test, trainable=trainable)
    x = conv2d_unit(x, i256, 1, act = 'leaky',bn = 0,stride=1, name='CSP-SPP', is_test=is_test, trainable=trainable)
    x = _spp(x)  
    x = conv2d_unit(x, i256, 3, act = 'leaky',bn = 0,stride=1,padding = 1, name='CSP-SPP_2', is_test=is_test, trainable=trainable)
    x = fluid.layers.concat([x, x1], axis=1)
    ###########################以上是csp-spp结构###########################

    ###################这是acfpn结构#######################################
    x = conv2d_unit(x, i256*2, 1, act = 'leaky',stride=1, name='conv017', is_test=is_test, trainable=trainable)
    s = conv2d_unit(x, i512, 3, act = 'leaky',stride=1,padding = 1, name='conv018', is_test=is_test, trainable=trainable)

    s11 = acfpn.dense_aspp(s,name = 'dsaspp')
    s11 = fluid.layers.concat([s11,s])
    print('s1')
    s11 = conv2d_unit(x, i256*2, 1, act = 'leaky',stride=1, name='convcnam', is_test=is_test, trainable=trainable)
    s21 = acfpn.cxam(s11,name = 'cxam')
    s31 = acfpn.cnam(s11,s,name = 'cnam')
    s41 =s11+s21+s31
    s = conv2d_unit(s, i512, 3, act = 'leaky',stride=1, padding = 1,name='convcnam1', is_test=is_test, trainable=trainable)
    s = fluid.layers.elementwise_add(s,s41)
    print('use pac')
##################################################

    x = conv2d_unit(s, i512, 3, act = 'leaky',stride=1,padding = 1, name='conv0180', is_test=is_test, trainable=trainable)
    output_s = conv2d_unit(x, num_anchors * (num_classes + 5), 1, stride=1, bn=0, act=None, name='conv030',is_test=is_test, trainable=trainable)


    x = conv2d_unit(s, i128, 1, act = 'leaky',stride=1, name='conv020', is_test=is_test, trainable=trainable)
    x = fluid.layers.resize_nearest(x, scale=float(2))
    x = fluid.layers.concat([x, s2], axis=1)
    x = conv2d_unit(x, i256, 3, act = 'leaky',stride=1, padding=1, name='conv026', is_test=is_test, trainable=trainable)
    output_m = conv2d_unit(x, num_anchors * (num_classes + 5), 1, stride=1, bn=0, act=None, name='conv0300',is_test=is_test, trainable=trainable)

    

    x = conv2d_unit(x, i64, 1, act = 'leaky',stride=1, name='conv023', is_test=is_test, trainable=trainable)
    x = fluid.layers.resize_nearest(x, scale=float(2))
    x = fluid.layers.concat([x, s1], axis=1)
    l = conv2d_unit(x, i128, 3, act = 'leaky',stride=1,padding = 1, name='conv024', is_test=is_test, trainable=trainable)
    output_l = conv2d_unit(l, num_anchors * (num_classes + 5), 1, stride=1, bn=0, act=None, name='conv025',is_test=is_test, trainable=trainable)

    # x = conv2d_unit(l, i128, 3, act = 'leaky',stride=2, padding=1, name='conv031', is_test=is_test, trainable=trainable)
    # x = fluid.layers.concat([x, m], axis=1)
    # x = conv2d_unit(x, i256, 3, act = 'leaky',stride=1, padding=1, name='conv026', is_test=is_test, trainable=trainable)
    # output_m = conv2d_unit(x, num_anchors * (num_classes + 5), 1, stride=1, bn=0, act=None, name='conv027',is_test=is_test, trainable=trainable)

    # x = conv2d_unit(x, i256, 3, act = 'leaky',stride=2, padding=1, name='conv028', is_test=is_test, trainable=trainable)
    # x = fluid.layers.concat([x, s], axis=1)
    # x = conv2d_unit(x, i512, 3, act = 'leaky',stride=1,padding = 1, name='conv029', is_test=is_test, trainable=trainable)
    # output_s = conv2d_unit(x, num_anchors * (num_classes + 5), 1, stride=1, bn=0, act=None, name='conv030',is_test=is_test, trainable=trainable)

    if export:
        return yolo_decode(output_l, output_m, output_s, postprocess, param)

    # 相当于numpy的transpose()，交换下标
    output_l = fluid.layers.transpose(output_l, perm=[0, 2, 3, 1], name='output_l')
    output_m = fluid.layers.transpose(output_m, perm=[0, 2, 3, 1], name='output_m')
    output_s = fluid.layers.transpose(output_s, perm=[0, 2, 3, 1], name='output_s')
    return output_s, output_m, output_l
