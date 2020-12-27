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
import paddle.fluid.layers as P
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay
from paddle.fluid.initializer import Constant
from model.custom_layers import Conv2dUnit
from model.head import YOLOv3Head


class ConvBlock(object):
    def __init__(self, filters, use_dcn=False, stride=2, block_name='', is_first=False):
        super(ConvBlock, self).__init__()
        filters1, filters2, filters3 = filters
        self.block_name = block_name
        self.is_first = is_first

        self.conv1 = Conv2dUnit(filters1, 1, stride=1, padding=0, bias_attr=False, bn=1, act='relu', name=block_name+'_branch2a')
        self.conv2 = Conv2dUnit(filters2, 3, stride=stride, padding=1, bias_attr=False, bn=1, act='relu', name=block_name+'_branch2b', use_dcn=use_dcn)
        self.conv3 = Conv2dUnit(filters3, 1, stride=1, padding=0, bias_attr=False, bn=1, act=None, name=block_name+'_branch2c')

        if not self.is_first:
            self.conv4 = Conv2dUnit(filters3, 1, stride=1, padding=0, bias_attr=False, bn=1, act=None, name=block_name+'_branch1')
        else:
            self.conv4 = Conv2dUnit(filters3, 1, stride=stride, padding=0, bias_attr=False, bn=1, act=None, name=block_name+'_branch1')

    def __call__(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.conv2(x)
        x = self.conv3(x)
        if not self.is_first:
            input_tensor = fluid.layers.pool2d(
                input=input_tensor,
                pool_size=2,
                pool_stride=2,
                pool_padding=0,
                ceil_mode=True,
                pool_type='avg')
        shortcut = self.conv4(input_tensor)
        x = P.elementwise_add(x=x, y=shortcut, act='relu', name=self.block_name + ".add.output.5")
        return x


class IdentityBlock(object):
    def __init__(self, filters, use_dcn=False, block_name=''):
        super(IdentityBlock, self).__init__()
        filters1, filters2, filters3 = filters
        self.block_name = block_name

        self.conv1 = Conv2dUnit(filters1, 1, stride=1, padding=0, bias_attr=False, bn=1, act='relu', name=block_name+'_branch2a')
        self.conv2 = Conv2dUnit(filters2, 3, stride=1, padding=1, bias_attr=False, bn=1, act='relu', name=block_name+'_branch2b', use_dcn=use_dcn)
        self.conv3 = Conv2dUnit(filters3, 1, stride=1, padding=0, bias_attr=False, bn=1, act=None, name=block_name+'_branch2c')

    def __call__(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.conv2(x)
        x = self.conv3(x)
        x = P.elementwise_add(x=x, y=input_tensor, act='relu', name=self.block_name + ".add.output.5")
        return x

class Resnet50Vd(object):
    def __init__(self, dcn_v2_stages=[5]):
        super(Resnet50Vd, self).__init__()
        self.stage1_conv1_1 = Conv2dUnit(32, 3, stride=2, padding=1, bias_attr=False, bn=1, act='relu', name='conv1_1')
        self.stage1_conv1_2 = Conv2dUnit(32, 3, stride=1, padding=1, bias_attr=False, bn=1, act='relu', name='conv1_2')
        self.stage1_conv1_3 = Conv2dUnit(64, 3, stride=1, padding=1, bias_attr=False, bn=1, act='relu', name='conv1_3')

        # stage2
        self.stage2_0 = ConvBlock([64, 64, 256], stride=1, block_name='res2a', is_first=True)
        self.stage2_1 = IdentityBlock([64, 64, 256], block_name='res2b')
        self.stage2_2 = IdentityBlock([64, 64, 256], block_name='res2c')

        # stage3
        use_dcn = 3 in dcn_v2_stages
        self.stage3_0 = ConvBlock([128, 128, 512], block_name='res3a', use_dcn=use_dcn)
        self.stage3_1 = IdentityBlock([128, 128, 512], block_name='res3b', use_dcn=use_dcn)
        self.stage3_2 = IdentityBlock([128, 128, 512], block_name='res3c', use_dcn=use_dcn)
        self.stage3_3 = IdentityBlock([128, 128, 512], block_name='res3d', use_dcn=use_dcn)

        # stage4
        use_dcn = 4 in dcn_v2_stages
        self.stage4_0 = ConvBlock([256, 256, 1024], block_name='res4a', use_dcn=use_dcn)
        self.stage4_1 = IdentityBlock([256, 256, 1024], block_name='res4b', use_dcn=use_dcn)
        self.stage4_2 = IdentityBlock([256, 256, 1024], block_name='res4c', use_dcn=use_dcn)
        self.stage4_3 = IdentityBlock([256, 256, 1024], block_name='res4d', use_dcn=use_dcn)
        self.stage4_4 = IdentityBlock([256, 256, 1024], block_name='res4e', use_dcn=use_dcn)
        self.stage4_5 = IdentityBlock([256, 256, 1024], block_name='res4f', use_dcn=use_dcn)

        # stage5
        use_dcn = 5 in dcn_v2_stages
        self.stage5_0 = ConvBlock([512, 512, 2048], block_name='res5a', use_dcn=use_dcn)
        self.stage5_1 = IdentityBlock([512, 512, 2048], block_name='res5b', use_dcn=use_dcn)
        self.stage5_2 = IdentityBlock([512, 512, 2048], block_name='res5c', use_dcn=use_dcn)

    def __call__(self, input_tensor):
        x = self.stage1_conv1_1(input_tensor)
        x = self.stage1_conv1_2(x)
        x = self.stage1_conv1_3(x)
        x = fluid.layers.pool2d(
            input=x,
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max')

        # stage2
        x = self.stage2_0(x)
        x = self.stage2_1(x)
        s4 = self.stage2_2(x)
        # stage3
        x = self.stage3_0(s4)
        x = self.stage3_1(x)
        x = self.stage3_2(x)
        s8 = self.stage3_3(x)
        # stage4
        x = self.stage4_0(s8)
        x = self.stage4_1(x)
        x = self.stage4_2(x)
        x = self.stage4_3(x)
        x = self.stage4_4(x)
        s16 = self.stage4_5(x)
        # stage5
        x = self.stage5_0(s16)
        x = self.stage5_1(x)
        s32 = self.stage5_2(x)
        return [s8, s16, s32]






