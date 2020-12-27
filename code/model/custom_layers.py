#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : zhoukunyang
#   Created date: 2020-12-08
#  
#
# ================================================================
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay
from paddle.fluid.initializer import Constant


class Conv2dUnit(object):
    def __init__(self,
                 filters,
                 filter_size,
                 stride=1,
                 padding=0,
                 bias_attr=False,
                 bn=0,
                 act=None,
                 name='',
                 freeze_norm=False,
                 is_test=False,
                 norm_decay=0.,
                 use_dcn=False):
        super(Conv2dUnit, self).__init__()
        self.filters = filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.bias_attr = bias_attr
        self.bn = bn
        self.act = act
        self.name = name
        self.freeze_norm = freeze_norm
        self.is_test = is_test
        self.norm_decay = norm_decay
        self.use_dcn = use_dcn

    def __call__(self, x):
        conv_name = self.name
        if self.use_dcn:
            offset_mask = fluid.layers.conv2d(
                input=x,
                num_filters=self.filter_size * self.filter_size * 3,
                filter_size=self.filter_size,
                stride=self.stride,
                padding=self.padding,
                act=None,
                param_attr=ParamAttr(initializer=Constant(0.0), name=conv_name + "_conv_offset.w_0"),
                bias_attr=ParamAttr(initializer=Constant(0.0), name=conv_name + "_conv_offset.b_0"),
                name=conv_name + "_conv_offset")
            offset = offset_mask[:, :self.filter_size**2 * 2, :, :]
            mask = offset_mask[:, self.filter_size**2 * 2:, :, :]
            mask = fluid.layers.sigmoid(mask)
            x = fluid.layers.deformable_conv(input=x, offset=offset, mask=mask,
                                             num_filters=self.filters,
                                             filter_size=self.filter_size,
                                             stride=self.stride,
                                             padding=self.padding,
                                             groups=1,
                                             deformable_groups=1,
                                             im2col_step=1,
                                             param_attr=ParamAttr(name=conv_name + "_weights"),
                                             bias_attr=False,
                                             name=conv_name + ".conv2d.output.1")
        else:
            battr = None
            if self.bias_attr:
                battr = ParamAttr(name=conv_name + "_bias")
            x = fluid.layers.conv2d(
                input=x,
                num_filters=self.filters,
                filter_size=self.filter_size,
                stride=self.stride,
                padding=self.padding,
                act=None,
                param_attr=ParamAttr(name=conv_name + "_weights"),
                bias_attr=battr,
                name=conv_name + '.conv2d.output.1')
        if self.bn:
            if conv_name == "conv1":
                bn_name = "bn_" + conv_name
            else:
                bn_name = "bn" + conv_name[3:]
            norm_lr = 0. if self.freeze_norm else 1.   # 归一化层学习率
            norm_decay = self.norm_decay   # 衰减
            pattr = ParamAttr(
                name=bn_name + '_scale',
                learning_rate=norm_lr,
                regularizer=L2Decay(norm_decay))   # L2权重衰减正则化
            battr = ParamAttr(
                name=bn_name + '_offset',
                learning_rate=norm_lr,
                regularizer=L2Decay(norm_decay))   # L2权重衰减正则化
            global_stats = True if self.freeze_norm else False
            x = fluid.layers.batch_norm(
                input=x,
                name=bn_name + '.output.1',
                param_attr=pattr,
                bias_attr=battr,
                moving_mean_name=bn_name + '_mean',
                moving_variance_name=bn_name + '_variance',
                use_global_stats=global_stats)
            scale = fluid.framework._get_var(pattr.name)
            bias = fluid.framework._get_var(battr.name)
            if self.freeze_norm:
                scale.stop_gradient = True
                bias.stop_gradient = True
        if self.act == 'leaky':
            x = fluid.layers.leaky_relu(x, alpha=0.1)
        elif self.act == 'relu':
            x = fluid.layers.relu(x)
        return x


def DropBlock(input, block_size, keep_prob, is_test):
    if is_test:
        return input

    def CalculateGamma(input, block_size, keep_prob):
        input_shape = fluid.layers.shape(input)
        feat_shape_tmp = fluid.layers.slice(input_shape, [0], [3], [4])
        feat_shape_tmp = fluid.layers.cast(feat_shape_tmp, dtype="float32")
        feat_shape_t = fluid.layers.reshape(feat_shape_tmp, [1, 1, 1, 1])
        feat_area = fluid.layers.pow(feat_shape_t, factor=2)

        block_shape_t = fluid.layers.fill_constant(
            shape=[1, 1, 1, 1], value=block_size, dtype='float32')
        block_area = fluid.layers.pow(block_shape_t, factor=2)

        useful_shape_t = feat_shape_t - block_shape_t + 1
        useful_area = fluid.layers.pow(useful_shape_t, factor=2)

        upper_t = feat_area * (1 - keep_prob)
        bottom_t = block_area * useful_area
        output = upper_t / bottom_t
        return output

    gamma = CalculateGamma(input, block_size=block_size, keep_prob=keep_prob)
    input_shape = fluid.layers.shape(input)
    p = fluid.layers.expand_as(gamma, input)

    input_shape_tmp = fluid.layers.cast(input_shape, dtype="int64")
    random_matrix = fluid.layers.uniform_random(
        input_shape_tmp, dtype='float32', min=0.0, max=1.0)
    one_zero_m = fluid.layers.less_than(random_matrix, p)
    one_zero_m.stop_gradient = True
    one_zero_m = fluid.layers.cast(one_zero_m, dtype="float32")

    mask_flag = fluid.layers.pool2d(
        one_zero_m,
        pool_size=block_size,
        pool_type='max',
        pool_stride=1,
        pool_padding=block_size // 2)
    mask = 1.0 - mask_flag

    elem_numel = fluid.layers.reduce_prod(input_shape)
    elem_numel_m = fluid.layers.cast(elem_numel, dtype="float32")
    elem_numel_m.stop_gradient = True

    elem_sum = fluid.layers.reduce_sum(mask)
    elem_sum_m = fluid.layers.cast(elem_sum, dtype="float32")
    elem_sum_m.stop_gradient = True

    output = input * mask * elem_numel_m / elem_sum_m
    return output


