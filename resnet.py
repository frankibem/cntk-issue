import numpy as np
from cntk.ops import relu, plus
from cntk.initializer import he_normal
from cntk.layers import BatchNormalization, Convolution, MaxPooling, Dropout


def conv_bn(layer_input, filter_size, num_filters, strides, init=he_normal(), name=''):
    """
    Returns a convolutional layer followed by a batch normalization layer
    """
    r = Convolution(filter_size, num_filters, activation=None, init=init, pad=True, strides=strides, bias=True, name=name)(layer_input)
    r = BatchNormalization(map_rank=1, normalization_time_constant=4096, name='{}_bn'.format(name))(r)
    return r


def conv_bn_relu(layer_input, filter_size, num_filters, strides, init=he_normal(), name=''):
    """
    Returns a convolutional layer followed by a batch normalization layer and then ReLU activation
    """
    r = conv_bn(layer_input, filter_size, num_filters, strides, init, name=name)
    return relu(r, name='{}_relu'.format(name))


def resnet_basic(layer_input, filter_size, num_filters, strides, prefix):
    """
    Returns a resnet basic building block
    """
    c1 = conv_bn_relu(layer_input, filter_size, num_filters, strides, name='{}_1'.format(prefix))
    c2 = conv_bn(c1, filter_size, num_filters, strides, name='{}_2'.format(prefix))
    p = plus(c2, layer_input, name='{}_res'.format(prefix))
    return relu(p, name='{}_relu'.format(prefix))


def resnet_basic_inc(layer_input, filter_size, num_filters, strides, prefix):
    """
    Returns a ResNet basic bulding block with projection
    Use when there is a change in layer_input/output channels
    """
    ones = np.ones_like(strides)
    c1 = conv_bn_relu(layer_input, filter_size, num_filters, strides, name='{}_1'.format(prefix))
    c2 = conv_bn(c1, filter_size, num_filters, ones, name='{}_2'.format(prefix))
    s = conv_bn(layer_input, ones, num_filters, strides, name='{}_3'.format(prefix))
    p = plus(c2, s, name='{}_res'.format(prefix))
    return relu(p, name='{}_relu'.format(prefix))


def resnet_basic_stack(layer_input, num_stack_layers, filter_size, num_filters, strides, prefix):
    """
    Returns a stack of ResNet basic building blocks
    """
    assert (num_stack_layers >= 0)
    l_in = layer_input
    for i in range(num_stack_layers):
        l_in = resnet_basic(l_in, filter_size, num_filters, strides, prefix='{}_{}'.format(prefix, i))
    return l_in


def resnet_model(layer_input):
    layer1 = resnet_basic_stack(layer_input, 1, (3, 3), 6, (1, 1), prefix='conv1')
    layer1 = MaxPooling((3, 3), (2, 2), name='pool1')(layer1)
    layer1 = Dropout(0.3, name='drop1')(layer1)

    layer2 = resnet_basic_inc(layer1, (3, 3), 8, (2, 2), prefix='conv21')
    layer2 = resnet_basic_stack(layer2, 1, (3, 3), 8, (1, 1), prefix='conv22')
    layer2 = Dropout(0.3, name='drop2')(layer2)

    layer3 = resnet_basic_inc(layer2, (3, 3), 10, (2, 2), prefix='conv31')
    layer3 = resnet_basic_stack(layer3, 1, (3, 3), 10, (1, 1), prefix='conv32')
    layer3 = Dropout(0.3, name='drop3')(layer3)

    layer4 = resnet_basic_inc(layer3, (3, 3), 10, (2, 2), prefix='conv41')
    layer4 = resnet_basic_stack(layer4, 1, (3, 3), 10, (1, 1), prefix='conv42')
    layer4 = Dropout(0.3, name='drop4')(layer4)

    return layer4
