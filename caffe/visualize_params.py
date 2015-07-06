import caffe

def get_layer_params(layers, net):
    return {layer: (net.params[layer][0].data, net.params[layer][1].data) for layer in layers}

def print_param_dimensions(params):
    msg = "{} weights are {} dimensional and biases are {} dimensional"
    for param in params:
        print msg.format(param, params[param][0].shape, params[param][1].shape)


deploy_net = caffe.Net('./fish_net_deploy.prototxt', './fish_net_iter_10000.caffemodel', caffe.TEST)
ip_layers = ['ip1', 'ip2']
ip_params = get_layer_params(ip_layers, deploy_net)
print_param_dimensions(ip_params)

conv_deploy_net = caffe.Net('./fish_net_conv_deploy.prototxt', './train_test_iter_10000.caffemodel', caffe.TEST)
ip_conv_layers = ['ip_conv1', 'ip_conv2']
ip_conv_params = get_layer_params(ip_conv_layers)
print_param_dimensions(ip_conv_params)
