# coding=utf-8

# Author: zhoukunyang


import paddle
from paddle import fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay
from paddle.fluid.initializer import Xavier


def ConvNorm(input,
             num_filters,
             filter_size,
             stride=1,
             groups=1,
             norm_decay=0.,
             norm_type='affine_channel',
             norm_groups=16,
             dilation=1,
             lr_scale=1,
             freeze_norm=False,
             act=None,
             norm_name=None,
             initializer=None,
             bias_attr=False,
             name=None):
    if bias_attr:
        bias_para = ParamAttr(
            name=name + "_bias",
            initializer=fluid.initializer.Constant(value=0),
            learning_rate=lr_scale * 2)
    else:
        bias_para = False
    conv = fluid.layers.conv2d(
        input=input,
        num_filters=num_filters,
        filter_size=filter_size,
        stride=stride,
        padding=((filter_size - 1) // 2) * dilation,
        dilation=dilation,
        groups=groups,
        act=None,
        param_attr=ParamAttr(
            name=name + "_weights",
            initializer=initializer,
            learning_rate=lr_scale),
        bias_attr=bias_para,
        name=name + '.conv2d.output.1')

    norm_lr = 0. if freeze_norm else 1.
    pattr = ParamAttr(
        name=norm_name + '_scale',
        learning_rate=norm_lr * lr_scale,
        regularizer=L2Decay(norm_decay))
    battr = ParamAttr(
        name=norm_name + '_offset',
        learning_rate=norm_lr * lr_scale,
        regularizer=L2Decay(norm_decay))

    if norm_type in ['bn', 'sync_bn']:
        global_stats = True if freeze_norm else False
        out = fluid.layers.batch_norm(
            input=conv,
            act=act,
            name=norm_name + '.output.1',
            param_attr=pattr,
            bias_attr=battr,
            moving_mean_name=norm_name + '_mean',
            moving_variance_name=norm_name + '_variance',
            use_global_stats=global_stats)
        scale = fluid.framework._get_var(pattr.name)
        bias = fluid.framework._get_var(battr.name)
    elif norm_type == 'gn':
        out = fluid.layers.group_norm(
            input=conv,
            act=act,
            name=norm_name + '.output.1',
            groups=norm_groups,
            param_attr=pattr,
            bias_attr=battr)
        scale = fluid.framework._get_var(pattr.name)
        bias = fluid.framework._get_var(battr.name)
    elif norm_type == 'affine_channel':
        scale = fluid.layers.create_parameter(
            shape=[conv.shape[1]],
            dtype=conv.dtype,
            attr=pattr,
            default_initializer=fluid.initializer.Constant(1.))
        bias = fluid.layers.create_parameter(
            shape=[conv.shape[1]],
            dtype=conv.dtype,
            attr=battr,
            default_initializer=fluid.initializer.Constant(0.))
        out = fluid.layers.affine_channel(
            x=conv, scale=scale, bias=bias, act=act)
    if freeze_norm:
        scale.stop_gradient = True
        bias.stop_gradient = True
    return out

class ACFPN():
    def __init__(self,
                 num_chan=512,
                 min_level=2,
                 max_level=6,
                 spatial_scale=[1. / 32., 1. / 16., 1. / 8., 1. / 4.],
                 has_extra_convs=False,
                 norm_type=None,
                 freeze_norm=False,
                 use_c5=True,
                 norm_groups=32):
        self.freeze_norm = freeze_norm
        self.num_chan = num_chan
        self.min_level = min_level
        self.max_level = max_level
        self.spatial_scale = spatial_scale
        self.has_extra_convs = has_extra_convs
        self.norm_type = norm_type
        self.use_c5 = use_c5
        self.norm_groups = norm_groups

    def dense_aspp_block(self, input, num_filters1, num_filters2, dilation_rate,
                         dropout_prob, name):

        conv = ConvNorm(
            input,
            num_filters=num_filters1,
            filter_size=1,
            stride=1,
            groups=1,
            norm_decay=0.,
            norm_type='gn',
            norm_groups=self.norm_groups,
            dilation=dilation_rate,
            lr_scale=1,
            freeze_norm=False,
            act="relu",
            norm_name=name + "_gn",
            initializer=None,
            bias_attr=False,
            name=name + "_gn")

        conv = fluid.layers.conv2d(
            conv,
            num_filters2,
            filter_size=3,
            padding=dilation_rate,
            dilation=dilation_rate,
            act="relu",
            param_attr=ParamAttr(name=name + "_conv_w"),
            bias_attr=ParamAttr(name=name + "_conv_b"), )

        if dropout_prob > 0:
            conv = fluid.layers.dropout(conv, dropout_prob=dropout_prob)

        return conv

    def dense_aspp(self, input, name=None):
        dropout0 = 0.1
        d_feature0 = 512
        d_feature1 = 512

        aspp3 = self.dense_aspp_block(
            input,
            num_filters1=d_feature0,
            num_filters2=d_feature1,
            dropout_prob=dropout0,
            name=name + '_aspp3',
            dilation_rate=3)
        conv = fluid.layers.concat([aspp3, input], axis=1)

        aspp6 = self.dense_aspp_block(
            conv,
            num_filters1=d_feature0,
            num_filters2=d_feature1,
            dropout_prob=dropout0,
            name=name + '_aspp6',
            dilation_rate=6)
        conv = fluid.layers.concat([aspp6, conv], axis=1)

        aspp12 = self.dense_aspp_block(
            conv,
            num_filters1=d_feature0,
            num_filters2=d_feature1,
            dropout_prob=dropout0,
            name=name + '_aspp12',
            dilation_rate=12)
        conv = fluid.layers.concat([aspp12, conv], axis=1)

        aspp18 = self.dense_aspp_block(
            conv,
            num_filters1=d_feature0,
            num_filters2=d_feature1,
            dropout_prob=dropout0,
            name=name + '_aspp18',
            dilation_rate=18)
        conv = fluid.layers.concat([aspp18, conv], axis=1)

        aspp24 = self.dense_aspp_block(
            conv,
            num_filters1=d_feature0,
            num_filters2=d_feature1,
            dropout_prob=dropout0,
            name=name + '_aspp24',
            dilation_rate=24)

        conv = fluid.layers.concat(
            [aspp3, aspp6, aspp12, aspp18, aspp24], axis=1)

        conv1 = ConvNorm(
            conv,
            num_filters=self.num_chan,
            filter_size=1,
            stride=1,
            groups=1,
            norm_decay=0.,
            norm_type='gn',
            norm_groups=self.norm_groups,
            dilation=1,
            lr_scale=1,
            freeze_norm=False,
            act="relu",
            norm_name=name + "_dense_aspp_reduce_gn",
            initializer=None,
            bias_attr=False,
            name=name + "_dense_aspp_reduce_gn")

        return conv1

    # def cnam_block(self,inputs,name = ''):
    #     _,c,w,h = inputs.shape
    #     x1 = fluid.layers.conv2d(inputs,c,1,act = 'relu',name = 'cnam11'+name)
    #     y1 = fluid.layers.conv2d(inputs,c,1,act = 'relu',name = 'cnam12'+name)
    #     z1 = fluid.layers.conv2d(inputs,c,1,act = 'relu',name = 'cnam13'+name)
    #     x = fluid.layers.reshape(x = x1,shape=[c,_*w*h])
    #     y = fluid.layers.reshape(x = y1,shape=[_,*w*h,c])
    #     # x = fluid.layers.transpose(x,perm = [0,2,1])
    #     s = fluid.layers.matmul(x,y)
    #     s = fluid.layers.softmax(s)
    #     _,c,N = s.shape
    #     s = fluid.layers.reshape(x = s,shape=[s.shape[0],s.shape[1],w,h])
    #     s = fluid.layers.pool2d(s,pool_size=3,pool_type = 'avg',pool_stride = 1,pool_padding = 1)
    #     s = fluid.layers.sigmoid(s)
    #     s = fluid.layers.conv2d(s,1,1,act = 'relu',name='out_cnamm'+name)
    #     return s

    def cnam_block(self,input,name = ''):
        _,c,_,_ = input.shape
        x = fluid.layers.adaptive_pool2d(input=input,pool_size=[1, 1],pool_type='avg')
        x = fluid.layers.conv2d(x,c//2,1,act = None,name = 'cnam11'+name)
        param_attr = fluid.ParamAttr(name='batch_norm_w'+name+'cnam1', initializer=fluid.initializer.Constant(value=1.0))
        bias_attr = fluid.ParamAttr(name='batch_norm_b'+name+'cnam1', initializer=fluid.initializer.Constant(value=0.0))
        x = fluid.layers.batch_norm(x,param_attr = param_attr, bias_attr = bias_attr)
        x = fluid.layers.relu(x)
        x=  fluid.layers.conv2d(x,c,1,act =None,name = 'cnam12'+name)
        param_attr = fluid.ParamAttr(name='batch_norm_w'+name+'cnam2', initializer=fluid.initializer.Constant(value=1.0))
        bias_attr = fluid.ParamAttr(name='batch_norm_b'+name+'cnam2', initializer=fluid.initializer.Constant(value=0.0))
        x = fluid.layers.batch_norm(x,param_attr = param_attr, bias_attr = bias_attr)
        y = fluid.layers.conv2d(input,c//2,1,act =None,name = 'cnam13'+name)
        param_attr = fluid.ParamAttr(name='batch_norm_w'+name+'cnam3', initializer=fluid.initializer.Constant(value=1.0))
        bias_attr = fluid.ParamAttr(name='batch_norm_b'+name+'cnam3', initializer=fluid.initializer.Constant(value=0.0))
        y = fluid.layers.batch_norm(y,param_attr = param_attr, bias_attr = bias_attr)
        y = fluid.layers.relu(y)
        y = fluid.layers.conv2d(y,c,1,act =None,name = 'cnam14'+name)
        param_attr = fluid.ParamAttr(name='batch_norm_w'+name+'cnam4', initializer=fluid.initializer.Constant(value=1.0))
        bias_attr = fluid.ParamAttr(name='batch_norm_b'+name+'cnam4', initializer=fluid.initializer.Constant(value=0.0))
        y = fluid.layers.batch_norm(y,param_attr = param_attr, bias_attr = bias_attr)
        out = fluid.layers.elementwise_mul(x,y)
        out = fluid.layers.conv2d(out,1,1,act =None,name = 'cnam15'+name)
        return out
    
    def cnam(self,input,input1,name = 'cnam'):
        out_weight = self.cnam_block(input,name = name)
        _,c,_,_ = input1.shape
        x = fluid.layers.conv2d(input1,c,1,name = 'cnam'+name)
        out = fluid.layers.elementwise_mul(x,out_weight)
        return out
    
    def cxam(self,input,name ='cxam'):
        out_weight = self.cnam_block(input,name = name)
        _,c,_,_ = input.shape
        x = fluid.layers.conv2d(input,c,1,name = 'cxam'+name)
        out = fluid.layers.elementwise_mul(x,out_weight)
        return out




