function generate_label_image(hdf5_path, output_path)
%% generate_label_image(hdf5_path, output_path)

%% \beta = 3 params
%    thresh = 0.8;
%    sigma = 6;
%    open_rad = 1;

%% \beta = 1 params
    thresh = 0.9;
    sigma = 7;
    open_rad = 3;

    net_out_data = h5read(hdf5_path, '/output');
    cell_signal = net_out_data(:,:,2)';

    labels = segment_net_output(cell_signal, thresh, sigma, open_rad);

    imwrite(labels, output_path);

