function segmented_mask = segment_net_output(net_output, threshold, sigma, opening_radius)
    %tic
    filter_size = ceil(sigma) * 3;
    gauss_filter = fspecial('gaussian', [filter_size, filter_size], sigma);
    filtered_output = imfilter(net_output, gauss_filter, 'conv');
    %disp(['Filtering Time: ', num2str(toc)])

    %tic
    mask = filtered_output > threshold;
    mask = imopen(mask, strel('disk', opening_radius));
    %disp(['Thresholding and Opening Time: ', num2str(toc)])

    %tic
    ws_in = -filtered_output;
    ws_in(~mask) = -Inf;

    segmented_mask = watershed(ws_in);
    %disp(['Watersheding Time: ', num2str(toc)])
