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
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay

from model.custom_layers import DropBlock


class YOLOv3Head(object):
    def __init__(self,
                 norm_decay=0.,
                 num_classes=80,
                 anchors=[[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                          [59, 119], [116, 90], [156, 198], [373, 326]],
                 anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                 drop_block=True,
                 block_size=3,
                 keep_prob=0.9,
                 weight_prefix_name=''):
        self.norm_decay = norm_decay
        self.num_classes = num_classes
        self.anchor_masks = anchor_masks
        self._parse_anchors(anchors)
        self.prefix_name = weight_prefix_name
        self.drop_block = drop_block
        self.block_size = block_size
        self.keep_prob = keep_prob

    def _parse_anchors(self, anchors):
        """
        Check ANCHORS/ANCHOR_MASKS in config and parse mask_anchors

        """
        self.anchors = []
        self.mask_anchors = []

        assert len(anchors) > 0, "ANCHORS not set."
        assert len(self.anchor_masks) > 0, "ANCHOR_MASKS not set."

        for anchor in anchors:
            assert len(anchor) == 2, "anchor {} len should be 2".format(anchor)
            self.anchors.extend(anchor)

        anchor_num = len(anchors)
        for masks in self.anchor_masks:
            self.mask_anchors.append([])
            for mask in masks:
                assert mask < anchor_num, "anchor mask index overflow"
                self.mask_anchors[-1].extend(anchors[mask])

    def _conv_bn(self,
                 input,
                 ch_out,
                 filter_size,
                 stride,
                 padding,
                 act='leaky',
                 is_test=True,
                 name=None):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=ch_out,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            act=None,
            param_attr=ParamAttr(name=name + ".conv.weights"),
            bias_attr=False)

        bn_name = name + ".bn"
        bn_param_attr = ParamAttr(
            regularizer=L2Decay(self.norm_decay), name=bn_name + '.scale')
        bn_bias_attr = ParamAttr(
            regularizer=L2Decay(self.norm_decay), name=bn_name + '.offset')
        out = fluid.layers.batch_norm(
            input=conv,
            act=None,
            is_test=is_test,
            param_attr=bn_param_attr,
            bias_attr=bn_bias_attr,
            moving_mean_name=bn_name + '.mean',
            moving_variance_name=bn_name + '.var')

        if act == 'leaky':
            out = fluid.layers.leaky_relu(x=out, alpha=0.1)
        return out

    def _upsample(self, input, scale=2, name=None):
        out = fluid.layers.resize_nearest(
            input=input, scale=float(scale), name=name)
        return out



    def _detection_block(self, input, channel, is_test=True, name=None):
        assert channel % 2 == 0, \
            "channel {} cannot be divided by 2 in detection block {}" \
                .format(channel, name)

        conv = input
        for j in range(2):  # 1x1卷积、3x3卷积、DropBlock（符合条件的话）。重复2次。
            conv = self._conv_bn(
                conv,
                channel,
                filter_size=1,
                stride=1,
                padding=0,
                is_test=is_test,
                name='{}.{}.0'.format(name, j))
            conv = self._conv_bn(
                conv,
                channel * 2,
                filter_size=3,
                stride=1,
                padding=1,
                is_test=is_test,
                name='{}.{}.1'.format(name, j))
            if self.drop_block and j == 0 and channel != 512:
                conv = DropBlock(
                    conv,
                    block_size=self.block_size,
                    keep_prob=self.keep_prob,
                    is_test=is_test)

        # DropBlock()随机丢失信息。先黑盒处理，不看代码。
        if self.drop_block and channel == 512:
            conv = DropBlock(
                conv,
                block_size=self.block_size,
                keep_prob=self.keep_prob,
                is_test=is_test)

        # 先1x1卷积得到route。route再3x3卷积得到tip。
        # tip再接卷积进行预测。route用于上采样。
        route = self._conv_bn(
            conv,
            channel,
            filter_size=1,
            stride=1,
            padding=0,
            is_test=is_test,
            name='{}.2'.format(name))
        tip = self._conv_bn(
            route,
            channel * 2,
            filter_size=3,
            stride=1,
            padding=1,
            is_test=is_test,
            name='{}.tip'.format(name))
        return route, tip

    def _get_outputs(self, input, is_train=True):
        outputs = []

        # get last out_layer_num blocks in reverse order
        out_layer_num = len(self.anchor_masks)  # 输出层的数量，有3个输出层。

        # 倒序的写法。blocks是input的倒序。用于fpn的3个张量的倒序。
        # blocks = [backbone_s32, backbone_s16, backbone_s8]
        blocks = input[-1:-out_layer_num - 1:-1]

        route = None
        for i, block in enumerate(blocks):  # blocks = [backbone_s32, backbone_s16, backbone_s8]
            if i > 0:  # perform concat in first 2 detection_block
                block = fluid.layers.concat(input=[route, block], axis=1)

            # tip再接卷积进行预测。route用于上采样。
            route, tip = self._detection_block(
                block,
                channel=512 // (2 ** i),
                is_test=(not is_train),
                name=self.prefix_name + "yolo_block.{}".format(i))

            # out channel number = mask_num * (5 + class_num)
            num_filters = len(self.anchor_masks[i]) * (self.num_classes + 5)
            with fluid.name_scope('yolo_output'):
                # 输出卷积层的weights没加正则
                # 输出卷积层的bias加了L2正则
                block_out = fluid.layers.conv2d(
                    input=tip,
                    num_filters=num_filters,
                    filter_size=1,
                    stride=1,
                    padding=0,
                    act=None,
                    param_attr=ParamAttr(
                        name=self.prefix_name +
                             "yolo_output.{}.conv.weights".format(i)),
                    bias_attr=ParamAttr(
                        regularizer=L2Decay(0.),
                        name=self.prefix_name +
                             "yolo_output.{}.conv.bias".format(i)))
                outputs.append(block_out)

            # route用于上采样。
            if i < len(blocks) - 1:
                # do not perform upsample in the last detection_block
                route = self._conv_bn(
                    input=route,
                    ch_out=256 // (2 ** i),
                    filter_size=1,
                    stride=1,
                    padding=0,
                    is_test=(not is_train),
                    name=self.prefix_name + "yolo_transition.{}".format(i))
                # upsample
                route = self._upsample(route)
        return outputs

    def __call__(self, x):
        return self._get_outputs(x)





class YOLACTHead(object):
    def __init__(self,
                 norm_decay=0.,
                 num_classes=80,
                 mask_dim=32,
                 anchors=[[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                          [59, 119], [116, 90], [156, 198], [373, 326]],
                 anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                 drop_block=True,
                 block_size=3,
                 keep_prob=0.9,
                 weight_prefix_name=''):
        self.norm_decay = norm_decay
        self.num_classes = num_classes
        self.mask_dim = mask_dim
        self.anchor_masks = anchor_masks
        self._parse_anchors(anchors)
        self.prefix_name = weight_prefix_name
        self.drop_block = drop_block
        self.block_size = block_size
        self.keep_prob = keep_prob

    def _parse_anchors(self, anchors):
        """
        Check ANCHORS/ANCHOR_MASKS in config and parse mask_anchors

        """
        self.anchors = []
        self.mask_anchors = []

        assert len(anchors) > 0, "ANCHORS not set."
        assert len(self.anchor_masks) > 0, "ANCHOR_MASKS not set."

        for anchor in anchors:
            assert len(anchor) == 2, "anchor {} len should be 2".format(anchor)
            self.anchors.extend(anchor)

        anchor_num = len(anchors)
        for masks in self.anchor_masks:
            self.mask_anchors.append([])
            for mask in masks:
                assert mask < anchor_num, "anchor mask index overflow"
                self.mask_anchors[-1].extend(anchors[mask])

    def _conv_bn(self,
                 input,
                 ch_out,
                 filter_size,
                 stride,
                 padding,
                 act='leaky',
                 is_test=True,
                 name=None):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=ch_out,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            act=None,
            param_attr=ParamAttr(name=name + ".conv.weights"),
            bias_attr=False)

        bn_name = name + ".bn"
        bn_param_attr = ParamAttr(
            regularizer=L2Decay(self.norm_decay), name=bn_name + '.scale')
        bn_bias_attr = ParamAttr(
            regularizer=L2Decay(self.norm_decay), name=bn_name + '.offset')
        out = fluid.layers.batch_norm(
            input=conv,
            act=None,
            is_test=is_test,
            param_attr=bn_param_attr,
            bias_attr=bn_bias_attr,
            moving_mean_name=bn_name + '.mean',
            moving_variance_name=bn_name + '.var')

        if act == 'leaky':
            out = fluid.layers.leaky_relu(x=out, alpha=0.1)
        return out

    def _conv(self,
                 input,
                 ch_out,
                 filter_size,
                 stride,
                 padding,
                 act='leaky',
                 is_test=True,
                 name=None):
        out = fluid.layers.conv2d(
            input=input,
            num_filters=ch_out,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            act=None,
            param_attr=ParamAttr(name=name + ".conv.weights"),
            bias_attr=False)

        bn_name = name + ".bn"
        bn_param_attr = ParamAttr(
            regularizer=L2Decay(self.norm_decay), name=bn_name + '.scale')
        bn_bias_attr = ParamAttr(
            regularizer=L2Decay(self.norm_decay), name=bn_name + '.offset')
        out = fluid.layers.batch_norm(
            input=out,
            act=None,
            is_test=is_test,
            param_attr=bn_param_attr,
            bias_attr=bn_bias_attr,
            moving_mean_name=bn_name + '.mean',
            moving_variance_name=bn_name + '.var')

        if act == 'leaky':
            out = fluid.layers.leaky_relu(x=out, alpha=0.1)
        return out

    def _upsample(self, input, scale=2, name=None):
        out = fluid.layers.resize_nearest(
            input=input, scale=float(scale), name=name)
        return out



    def _detection_block(self, input, channel, is_test=True, name=None):
        assert channel % 2 == 0, \
            "channel {} cannot be divided by 2 in detection block {}" \
                .format(channel, name)

        conv = input
        for j in range(2):  # 1x1卷积、3x3卷积、DropBlock（符合条件的话）。重复2次。
            conv = self._conv_bn(
                conv,
                channel,
                filter_size=1,
                stride=1,
                padding=0,
                is_test=is_test,
                name='{}.{}.0'.format(name, j))
            conv = self._conv_bn(
                conv,
                channel * 2,
                filter_size=3,
                stride=1,
                padding=1,
                is_test=is_test,
                name='{}.{}.1'.format(name, j))
            if self.drop_block and j == 0 and channel != 512:
                conv = DropBlock(
                    conv,
                    block_size=self.block_size,
                    keep_prob=self.keep_prob,
                    is_test=is_test)

        # DropBlock()随机丢失信息。先黑盒处理，不看代码。
        if self.drop_block and channel == 512:
            conv = DropBlock(
                conv,
                block_size=self.block_size,
                keep_prob=self.keep_prob,
                is_test=is_test)

        # 先1x1卷积得到route。route再3x3卷积得到tip。
        # tip再接卷积进行预测。route用于上采样。
        route = self._conv_bn(
            conv,
            channel,
            filter_size=1,
            stride=1,
            padding=0,
            is_test=is_test,
            name='{}.2'.format(name))
        tip = self._conv_bn(
            route,
            channel * 2,
            filter_size=3,
            stride=1,
            padding=1,
            is_test=is_test,
            name='{}.tip'.format(name))
        return route, tip


    def _spp(self, x):
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
        out = fluid.layers.concat(input=[x_1, x_2, x_3, x_4], axis=1)
        return out

    def _upsample_bilinear(self, input, scale=2, name=None):
        out = fluid.layers.resize_bilinear(input, scale=float(scale), name=name)
        out = fluid.layers.leaky_relu(x=out, alpha=0.1)
        return out

    def proto_net(self, x, is_test):
        # x = self._spp(x)
        x = self._conv(x, 256, filter_size=3, stride=1, padding=1, is_test=is_test, name='proto_net.0')
        x = self._conv(x, 256, filter_size=3, stride=1, padding=1, is_test=is_test, name='proto_net.1')
        x = self._conv(x, 256, filter_size=3, stride=1, padding=1, is_test=is_test, name='proto_net.2')
        # x = self._upsample_bilinear(x, name='proto_net.3')
        x = self._upsample(x, name='proto_net.3')
        x = self._conv(x, 256, filter_size=3, stride=1, padding=1, is_test=is_test, name='proto_net.4')
        x = self._conv(x, self.mask_dim, filter_size=1, stride=1, padding=0, is_test=is_test, name='proto_net.5')
        x = fluid.layers.transpose(x, perm=[0, 2, 3, 1])
        return x

    def _get_outputs(self, input, is_train=True):
        outputs = []
        mcf_outputs = []   # 掩码系数

        # get last out_layer_num blocks in reverse order
        out_layer_num = len(self.anchor_masks)  # 输出层的数量，有3个输出层。

        # 倒序的写法。blocks是input的倒序。用于fpn的3个张量的倒序。
        # blocks = [backbone_s32, backbone_s16, backbone_s8]
        blocks = input[-1:-out_layer_num - 1:-1]

        route = None
        for i, block in enumerate(blocks):  # blocks = [backbone_s32, backbone_s16, backbone_s8]
            if i > 0:  # perform concat in first 2 detection_block
                block = fluid.layers.concat(input=[route, block], axis=1)

            # tip再接卷积进行预测。route用于上采样。
            route, tip = self._detection_block(
                block,
                channel=512 // (2 ** i),
                is_test=(not is_train),
                name=self.prefix_name + "yolo_block.{}".format(i))

            # out channel number = mask_num * (5 + class_num)
            num_filters = len(self.anchor_masks[i]) * (self.num_classes + 5)
            num_filters2 = len(self.anchor_masks[i]) * self.mask_dim
            with fluid.name_scope('yolo_output'):
                # 输出卷积层的weights没加正则
                # 输出卷积层的bias加了L2正则
                block_out = fluid.layers.conv2d(
                    input=tip,
                    num_filters=num_filters,
                    filter_size=1,
                    stride=1,
                    padding=0,
                    act=None,
                    param_attr=ParamAttr(
                        name=self.prefix_name +
                             "yolo_output.{}.conv.weights".format(i)),
                    bias_attr=ParamAttr(
                        regularizer=L2Decay(0.),
                        name=self.prefix_name +
                             "yolo_output.{}.conv.bias".format(i)))
                # block_out.stop_gradient = True
                outputs.append(block_out)

                mcf_out = fluid.layers.conv2d(
                    input=tip,
                    num_filters=num_filters2,
                    filter_size=1,
                    stride=1,
                    padding=0,
                    act=None,
                    param_attr=ParamAttr(
                        name=self.prefix_name +
                             "yolo_mcf_output.{}.conv.weights".format(i)),
                    bias_attr=ParamAttr(
                        regularizer=L2Decay(0.),
                        name=self.prefix_name +
                             "yolo_mcf_output.{}.conv.bias".format(i)))
                mcf_outputs.append(mcf_out)

            # route用于上采样。
            if i < len(blocks) - 1:
                # do not perform upsample in the last detection_block
                route = self._conv_bn(
                    input=route,
                    ch_out=256 // (2 ** i),
                    filter_size=1,
                    stride=1,
                    padding=0,
                    is_test=(not is_train),
                    name=self.prefix_name + "yolo_transition.{}".format(i))
                # upsample
                route = self._upsample(route)
            else:  # 最后一个tip继续卷积，得到掩膜原型
                proto_out = self.proto_net(tip, is_test=(not is_train))
                # 用于语义分割损失
                segm = fluid.layers.conv2d(tip, self.num_classes, filter_size=(1, 1),
                                           param_attr=ParamAttr(name="semantic_seg_conv.weights"),
                                           bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0),
                                                               name="semantic_seg_conv.bias"))
        return outputs, mcf_outputs, proto_out, segm

    def __call__(self, x):
        return self._get_outputs(x)











