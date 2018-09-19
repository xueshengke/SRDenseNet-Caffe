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
first_channel = 8
block = 8
depth = 8
grow_rate = 16
bottleneck = 256
dropout = 0.0
################################################################################

def conv_relu(bottom, channel, kernel, stride, pad, dropout):
    conv = L.Convolution(bottom, num_output=channel, kernel_size=kernel, stride=stride, pad=pad,
                         bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
    relu = L.ReLU(conv, in_place=True)
    if dropout>0:
        relu = L.Dropout(relu, dropout_ratio=dropout)
    return relu

def add_layer(bottom, channel, dropout):
    conv = conv_relu(bottom, channel=channel, kernel=3, stride=1, pad=1, dropout=dropout)
    concate = L.Concat(bottom, conv, axis=1)
    return concate

################################################################################
# define the network for training and validation
def train_SRDenseNet(train_data=train_data_path, test_data=test_data_path,
                     batch_size_train=batch_size_train, batch_size_test=batch_size_test,
                     first_channel=first_channel, block=block, depth=depth, grow_rate=grow_rate,
                     bottleneck=bottleneck, dropout=dropout):
    net = caffe.NetSpec()
    net.data, net.label = L.HDF5Data(hdf5_data_param={
        'source': train_data, 'batch_size': batch_size_train}, include={'phase': caffe.TRAIN}, ntop=2)
    train_data_layer = str(net.to_proto())
    net.data, net.label = L.HDF5Data(hdf5_data_param={
        'source': test_data, 'batch_size': batch_size_test}, include={'phase': caffe.TEST}, ntop=2)

    net.model = conv_relu(net.data, channel=first_channel, kernel=3, stride=1, pad=1, dropout=dropout)

    num_channels = first_channel
    for i in range(block):
        net.dense = conv_relu(net.model, channel=grow_rate, kernel=3, stride=1, pad=1, dropout=dropout)
        for j in range(depth-1):
            net.dense = add_layer(net.dense, grow_rate, dropout)
        num_channels += grow_rate * depth
        net.model = L.Concat(net.model, net.dense, axis=1)

    net.bottleneck = conv_relu(net.model, channel=bottleneck, kernel=1, stride=1, pad=0, dropout=dropout)
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
def test_SRDenseNet(first_channel=first_channel, block=block, depth=depth,
                    grow_rate=grow_rate, bottleneck=bottleneck, dropout=dropout):
    net = caffe.NetSpec()

    net.data = L.Input(shape=dict(dim=[1,3,24,24]))

    # net.model = L.Convolution(bottom='data', num_output=nchannels, kernel_size=3, stride=1, pad=1,
    #                      bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
    # net.model = L.ReLU(net.model, in_place=True)
    net.model = conv_relu(net.data, channel=first_channel, kernel=3, stride=1, pad=1, dropout=dropout)

    num_channels = first_channel
    for i in range(block):
        net.dense = conv_relu(net.model, channel=grow_rate, kernel=3, stride=1, pad=1, dropout=dropout)
        for j in range(depth-1):
            net.dense = add_layer(net.dense, grow_rate, dropout)
        num_channels += grow_rate * depth
        net.model = L.Concat(net.model, net.dense, axis=1)

    net.bottleneck = conv_relu(net.model, channel=bottleneck, kernel=1, stride=1, pad=0, dropout=dropout)
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

    # net.loss = L.EuclideanLoss(net.reconstruct, net.label)

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
        print(str(test_SRDenseNet()), file=f)
    print('create ' + test_net_path)
