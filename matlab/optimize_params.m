fisherman_root = '/home/sam/code/fisherman';
data_root = fullfile(fisherman_root, 'data/dense_labelling/');
test_index = json.read(fullfile(data_root, 'test_index.json'));

keys = fieldnames(test_index);
thresh = 0.4:0.1:0.99;
open_rad = 1:10;
sigma = 1:0.5:8;
beta = 1;

F_scores = zeros(length(test_index), length(thresh), length(sigma), length(open_rad));
Precision = zeros(length(test_index), length(thresh), length(sigma), length(open_rad));
Recall = zeros(length(test_index), length(thresh), length(sigma), length(open_rad));
N_true_positive = zeros(length(test_index), length(thresh), length(sigma), length(open_rad));
N_true_labels = zeros(length(test_index), length(thresh), length(sigma), length(open_rad));
N_predicted_labels = zeros(length(test_index), length(thresh), length(sigma), length(open_rad));

for iImage = 1:length(keys)
    disp(['Analyzing image ', num2str(iImage)])
    curr_elem = test_index.(keys{iImage});
    ground_truth_mask = imread(fullfile(data_root, curr_elem.label));

    net_out_name = strrep(curr_elem.label, 'fov_mask.tiff', 'fish_net_output.hdf5');
    net_out_path = fullfile(fisherman_root, 'data/dense_labelling_k161/net_out/', net_out_name)
    net_out_data = h5read(net_out_path, '/output');

    cell_signal = net_out_data(:,:,2)';

    for iRad = 1:length(open_rad)
        for iSigma = 1:length(sigma)
            parfor iThresh = 1:length(thresh)
                labels = segment_net_output(cell_signal, thresh(iThresh), sigma(iSigma), open_rad(iRad));
                labels(labels == 1) = 0;
                [ ...
                    F_scores(iImage, iThresh, iSigma, iRad), ...
                    Precision(iImage, iThresh, iSigma, iRad), ...
                    Recall(iImage, iThresh, iSigma, iRad), ...
                    N_true_positive(iImage, iThresh, iSigma, iRad), ...
                    N_true_labels(iImage, iThresh, iSigma, iRad), ...
                    N_predicted_labels(iImage, iThresh, iSigma, iRad) ...
                ] = calculate_f_score(ground_truth_mask, labels, beta);
            end
        end
    end
end

Precision_overall = squeeze(sum(N_true_positive, 1)./sum(N_predicted_labels, 1));
Recall_overall = squeeze(sum(N_true_positive, 1)./sum(N_true_labels, 1));
F_overall = (1 + beta^2)*(Precision_overall .* Recall_overall)./(Precision_overall*beta^2 + Recall_overall);

[f_max, ind_max] = max(F_overall(:));
[opt_thresh_ind, opt_sigma_ind, opt_rad_ind] = ind2sub(size(F_overall), ind_max);
optimal_thresh = thresh(opt_thresh_ind);
optimal_sigma = sigma(opt_sigma_ind);
optimal_open_rad = open_rad(opt_rad_ind);

disp('Optimization Complete:')
disp(['F max: ', num2str(f_max)])
disp(['Optimal Threshold: ', num2str(optimal_thresh)])
disp(['Optimal Sigma: ', num2str(optimal_sigma)])
disp(['Optimal Opening Radius: ', num2str(optimal_open_rad)])
disp('-----------------------------------------')
