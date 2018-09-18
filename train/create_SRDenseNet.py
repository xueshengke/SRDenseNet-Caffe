# This file creates SRDenseNet prototxt files: 'train_val' for training and 'deploy' for test
from __future__ import print_function
from __future__ import print_function
import sys
# caffe path
sys.path.append('/home/xueshengke/caffe-1.0/python')
from caffe import layers as L, params as P, to_proto
# from caffe.proto import caffe_pb2
import caffe

################################################################################
# change filename here
train_net_path = 'train_net.prototxt'
test_net_path = 'test_net.prototxt'
train_data_path = 'examples/SRDenseNet/train.txt'
test_data_path = 'examples/SRDenseNet/test.txt'

# parameters of the network
scale = 4
batch_size_train = 32
batch_size_test = 2
first_output = 8
block = 8
depth = 8
grow_rate = 16
bottleneck = 256
dropout = 0.0
################################################################################

# def bn_relu_conv(bottom, nout, ks, stride, pad, dropout):
#     batch_norm = L.BatchNorm(bottom, in_place=False, param=[dict(lr_mult=0, decay_mult=0),
#                                                             dict(lr_mult=0, decay_mult=0),
#                                                             dict(lr_mult=0, decay_mult=0)])
#     scale = L.Scale(batch_norm, bias_term=True, in_place=True, filler=dict(value=1), bias_filler=dict(value=0))
#     relu = L.ReLU(scale, in_place=True)
#     conv = L.Convolution(relu, num_output=nout, kernel_size=ks, stride=stride, pad=pad,
#                          bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
#     if dropout>0:
#         conv = L.Dropout(conv, dropout_ratio=dropout)
#     return conv

def conv_relu(bottom, nout, ks, stride, pad, dropout):
    conv = L.Convolution(bottom, num_output=nout, kernel_size=ks, stride=stride, pad=pad,
                         bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
    relu = L.ReLU(conv, in_place=True)
    if dropout>0:
        relu = L.Dropout(relu, dropout_ratio=dropout)
    return relu

def add_layer(bottom, num_filter, dropout):
    conv = conv_relu(bottom, nout=num_filter, ks=3, stride=1, pad=1, dropout=dropout)
    concate = L.Concat(bottom, conv, axis=1)
    return concate

# def transition(bottom, num_filter, dropout):
#     conv = bn_relu_conv(bottom, nout=num_filter, ks=1, stride=1, pad=0, dropout=dropout)
#     pooling = L.Pooling(conv, pool=P.Pooling.AVE, kernel_size=2, stride=2)
#     return pooling

################################################################################
# define the network for training and validation
def train_SRDenseNet(train_data=train_data_path, test_data=test_data_path,
             batch_size_train=batch_size_train, batch_size_test=batch_size_test,
             first_output=first_output, block=block, depth=depth, grow_rate=grow_rate,
             bottleneck=bottleneck, dropout=dropout):
    net = caffe.NetSpec()
    net.data, net.label = L.HDF5Data(hdf5_data_param={
        'source': train_data, 'batch_size': batch_size_train}, include={'phase': caffe.TRAIN}, ntop=2)
    train_data_layer = str(net.to_proto())
    net.data, net.label = L.HDF5Data(hdf5_data_param={
        'source': test_data, 'batch_size': batch_size_test}, include={'phase': caffe.TEST}, ntop=2)

    nchannels = first_output
    net.model = conv_relu(net.data, nout=nchannels, ks=3, stride=1, pad=1, dropout=dropout)

    for i in range(block):
        net.dense = conv_relu(net.model, nout=grow_rate, ks=3, stride=1, pad=1, dropout=dropout)
        for j in range(depth-1):
            net.dense = add_layer(net.dense, grow_rate, dropout)
        nchannels += grow_rate * depth
        net.model = L.Concat(net.model, net.dense, axis=1)

    net.bottleneck = conv_relu(net.model, nout=bottleneck, ks=1, stride=1, pad=0, dropout=dropout)
    net.deconv1 = L.Deconvolution(net.bottleneck, convolution_param=dict(num_output=bottleneck,
                                                  kernel_size=4, stride=2, pad=1,
                                                  bias_term=False,
                                                  weight_filler=dict(type='msra'),
                                                  bias_filler=dict(type='constant')))
    net.deconv1 = L.ReLU(net.deconv1, in_place=True)

    net.deconv2 = L.Deconvolution(net.deconv1, convolution_param=dict(num_output=bottleneck,
                                               kernel_size=4, stride=2, pad=1,
                                               bias_term=False,
                                               weight_filler=dict(type='msra'),
                                               bias_filler=dict(type='constant')))
    net.deconv2 = L.ReLU(net.deconv2, in_place=True)

    net.reconstruct = L.Convolution(net.deconv2, num_output=1, kernel_size=3, stride=1,
                                    pad=1, bias_term=False, weight_filler=dict(type='msra'),
                                    bias_filler=dict(type='constant'))

    net.loss = L.EuclideanLoss(net.reconstruct, net.label)

    return train_data_layer + str(net.to_proto())

################################################################################
# deploy the network for test; no data, label, loss layers
def test_SRDenseNet(first_output=first_output, block=block, depth=depth,
                    grow_rate=grow_rate, bottleneck=bottleneck, dropout=dropout):
    net = caffe.NetSpec()

    nchannels = first_output
    # net.model = conv_relu(bottom='data', nout=nchannels, ks=3, stride=1, pad=1, dropout=dropout)
    net.model = L.Convolution(bottom='data', num_output=nchannels, kernel_size=3, stride=1, pad=1,
                         bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
    net.model = L.ReLU(net.model, in_place=True)

    for i in range(block):
        net.dense = conv_relu(net.model, nout=grow_rate, ks=3, stride=1, pad=1, dropout=dropout)
        for j in range(depth-1):
            net.dense = add_layer(net.dense, grow_rate, dropout)
        nchannels += grow_rate * depth
        net.model = L.Concat(net.model, net.dense, axis=1)

    net.bottleneck = conv_relu(net.model, nout=bottleneck, ks=1, stride=1, pad=0, dropout=dropout)
    net.deconv1 = L.Deconvolution(net.bottleneck, convolution_param=dict(num_output=bottleneck,
                                                  kernel_size=4, stride=2, pad=1,
                                                  bias_term=False,
                                                  weight_filler=dict(type='msra'),
                                                  bias_filler=dict(type='constant')))
    net.deconv1 = L.ReLU(net.deconv1, in_place=True)

    net.deconv2 = L.Deconvolution(net.deconv1, convolution_param=dict(num_output=bottleneck,
                                               kernel_size=4, stride=2, pad=1,
                                               bias_term=False,
                                               weight_filler=dict(type='msra'),
                                               bias_filler=dict(type='constant')))
    net.deconv2 = L.ReLU(net.deconv2, in_place=True)

    net.reconstruct = L.Convolution(net.deconv2, num_output=1, kernel_size=3, stride=1,
                                    pad=1, bias_term=False, weight_filler=dict(type='msra'),
                                    bias_filler=dict(type='constant'))

    return net.to_proto()

################################################################################
if __name__ == '__main__':
    # write train_val network
    with open(train_net_path, 'w') as f:
        print(str(train_SRDenseNet()), file=f)
    print('create ' + train_net_path)

    # write test network
    with open(test_net_path, 'w') as f:
        f.write('name: "SRDenseNet_x'+str(scale)+'_block'+str(block)+'_depth'+str(depth)+'_grow'+str(grow_rate)+'"\n')
        f.write('input: "data"\n')
        f.write('input_dim: 1\n')
        f.write('input_dim: 1\n')
        f.write('input_dim: width\n')
        f.write('input_dim: height\n')
        print(str(test_SRDenseNet()), file=f)
    print('create ' + test_net_path)
