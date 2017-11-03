from cntk.ops import relu, plus
from cntk.initializer import he_normal
from cntk.layers import BatchNormalization, Convolution, Dropout


def conv_bn(layer_input, filter_size, num_filters, strides=(1, 1, 1), init=he_normal(), name=''):
    """
    Returns a convolutional layer followed by a batch normalization layer
    """
    r = Convolution(filter_size, num_filters, activation=None, init=init, pad=True, strides=strides, bias=False, name=name)(layer_input)
    r = BatchNormalization(map_rank=1, normalization_time_constant=4096, name='{}_bn'.format(name))(r)
    return r


def conv_bn_relu(layer_input, filter_size, num_filters, strides=(1, 1, 1), init=he_normal(), name=''):
    """
    Returns a convolutional layer followed by a batch normalization layer and then ReLU activation
    """
    r = conv_bn(layer_input, filter_size, num_filters, strides, init, name=name)
    return relu(r, name='{}_relu'.format(name))


def resnet_basic(layer_input, filter_size, num_filters, prefix):
    """
    Returns a resnet basic building block
    """
    c1 = conv_bn_relu(layer_input, filter_size, num_filters, name='{}_1'.format(prefix))
    c2 = conv_bn(c1, filter_size, num_filters, name='{}_2'.format(prefix))
    p = plus(c2, layer_input, name='{}_res'.format(prefix))
    return relu(p, name='{}_relu'.format(prefix))


def resnet_basic_inc(layer_input, filter_size, num_filters, strides, prefix):
    """
    Returns a ResNet basic bulding block with projection
    Use when there is a change in layer_input/output channels
    """
    c1 = conv_bn_relu(layer_input, filter_size, num_filters, strides, name='{}_1'.format(prefix))
    c2 = conv_bn(c1, filter_size, num_filters, name='{}_2'.format(prefix))
    s = conv_bn(layer_input, (1, 1, 1), num_filters, strides, name='{}_3'.format(prefix))
    p = plus(c2, s, name='{}_res'.format(prefix))
    return relu(p, name='{}_relu'.format(prefix))


def resnet_basic_stack(layer_input, num_stack_layers, filter_size, num_filters, prefix):
    """
    Returns a stack of ResNet basic building blocks
    """
    assert (num_stack_layers >= 0)
    l_in = layer_input
    for i in range(num_stack_layers):
        l_in = resnet_basic(l_in, filter_size, num_filters, prefix='{}_{}'.format(prefix, i))
    return l_in


def resnet_model(layer_input):
    # Frame-level convolutional layers
    fconv1_1 = resnet_basic_stack(layer_input, 2, (1, 3, 3), 32, prefix='fconv1')

    fconv2_1 = resnet_basic_inc(fconv1_1, (1, 3, 3), 64, (1, 2, 2), prefix='fconv21')
    fconv2_2 = resnet_basic_stack(fconv2_1, 2, (1, 3, 3), 64, prefix='fconv22')

    fconv3_1 = resnet_basic_inc(fconv2_2, (1, 3, 3), 128, (1, 2, 2), prefix='fconv31')
    fconv3_2 = resnet_basic_stack(fconv3_1, 2, (1, 3, 3), 128, prefix='fconv32')
    fconv3_2 = Dropout(0.2, name='dropf3')(fconv3_2)

    # Depth-level convolutional layers
    dconv1 = resnet_basic_stack(fconv3_2, 2, (3, 1, 1), 128, prefix='dconv1')

    dconv2_1 = resnet_basic_inc(dconv1, (3, 1, 1), 64, (2, 1, 1), prefix='dconv21')
    dconv2_2 = resnet_basic_stack(dconv2_1, 2, (3, 1, 1), 64, prefix='dconv22')

    dconv3_1 = resnet_basic_inc(dconv2_2, (3, 1, 1), 8, (2, 1, 1), prefix='dconv31')
    dconv3_2 = resnet_basic_stack(dconv3_1, 2, (3, 1, 1), 8, prefix='dconv32')
    dconv3_2 = Dropout(0.2, name='dropd3')(dconv3_2)

    return dconv3_2
