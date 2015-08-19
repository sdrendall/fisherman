import caffe
from os.path import expanduser

def main():
    from sys import argv

    # Check params and load paths
    if len(argv) < 5:
        print "Insufficient Arguments!"
        print "Proper Usage: %s [input_net] [input_net_params] [output_net] [output_net_params]"
        return

    input_net_path, input_net_params_path, output_net_path, output_net_params_path = map(expanduser, argv[1:])

    # Load nets and input params
    input_net = caffe.Net(input_net_path, input_net_params_path, caffe.TEST)
    output_net = caffe.Net(output_net_path, caffe.TEST)
    
    input_params = input_net.params
    output_params = output_net.params

    # Flatten input params into output params
    for input_param, output_param in zip(input_params.itervalues(), output_params.itervalues()):
        output_param[0].data.flat = input_param[0].data.flat
        output_param[1].data.flat = input_param[1].data.flat

    output_net.save(output_net_params_path)

main()
