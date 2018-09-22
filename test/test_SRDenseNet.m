%% --------------------------
% This is the implementation of paper 'T. Tong, G. Li, X. Liu, et al., 2017. 
% Image super-resolution using dense skip connections. ICCV, p.4809-4817.'
% Code written by 
% Shengke Xue, Ph.D., Zhejiang University, email: xueshengke@zju.edu.cn
%% test SRDenseNet
close all; clear; clc;

setenv('LC_ALL','C')    % remove all local configurations
setenv('GLOG_minloglevel','2')  % remove any log when loading caffe modules
addpath '/home/xueshengke/caffe-1.0/matlab'; % change to your caffe path
addpath(genpath('util'));

%% parameters, change settings if necessary
gpu_id = 0;
% caffe.set_mode_cpu(); % for CPU
caffe.set_mode_gpu(); % for GPU
caffe.set_device(gpu_id);

weights = 'model/snapshot_iter_100000.caffemodel';
model = 'test_net.prototxt';
scale = 4;
data_set_id = 1;     % index to select one dataset for test
data_set = {
    'Set5',         '*.bmp';
    'Set14',        '*.bmp';
    'BSDS100',      '*.png';
    'Urban100',     '*.png';
};

path_folder = 'data';
save_path = 'result';
patch = 100;

% choose dataset
test_set = data_set{data_set_id, 1};
path = fullfile(path_folder, test_set);
dic = dir(fullfile(path, data_set{data_set_id, 2}));

num_file = length(dic);
    
folder_result = fullfile(save_path, test_set, ['x' num2str(scale)]);

if ~exist(folder_result,'file')
    mkdir(folder_result);
end

%% process image files with Bicubic interplotation and SRDenseNet
bicubic_set = zeros(num_file, 3);
srdensenet_set = zeros(num_file, 3);
im_h_set = cell(num_file, 1);
im_bic_set = cell(num_file, 1);
im_gnd_set = cell(num_file, 1);

for i = 1 : length(dic)   
    %% read image
    disp([num2str(i) '/' num2str(num_file) ', testing the image: ' dic(i).name]);
    file_name = dic(i).name;
    img  = imread(fullfile(path, file_name));
    image_name = file_name(1:end-4);

    %% work on luminance only
    im_ycbcr= img;
    if size(img, 3) > 1
        im_ycbcr = rgb2ycbcr(img);
        im_cb = im2double(im_ycbcr(:, :, 2));
        im_cr = im2double(im_ycbcr(:, :, 3));
        im_cb_gnd = modcrop(im_cb, scale);
        im_cr_gnd = modcrop(im_cr, scale);
        im_cb_low = imresize(im_cb_gnd, 1/scale, 'bicubic');
        im_cb_b = imresize(im_cb_low, scale, 'bicubic');
        im_cb_b = single(im_cb_b);
        im_cr_low = imresize(im_cr_gnd, 1/scale, 'bicubic');
        im_cr_b = imresize(im_cr_low, scale, 'bicubic');
        im_cr_b = single(im_cr_b);
    end
    im_y = im2double(im_ycbcr(:, :, 1));
    im_y_gnd = modcrop(im_y, scale);

    %% bicubic interpolation
    [hei,wid] = size(im_y_gnd);
    im_y_low = imresize(im_y_gnd, 1/scale, 'bicubic');
    im_y_b = imresize(im_y_low, scale, 'bicubic');
    im_y_b = single(im_y_b);
    
    %% run pretrained network on subimages, to avoid blob size exceeding
    [hei_low, wid_low] = size(im_y_low);
    im_y_h = zeros(hei, wid);
    for x = 1 : patch : hei_low
        for y = 1 : patch : wid_low
            subimage = im_y_low(max(1, x-1) : min(x+patch, hei_low), ...
                                max(1, y-1) : min(y+patch, wid_low));
            output = SRDenseNet(model, weights, subimage);
            [sh, sw] = size(output);
            sx = max(scale * (x-1) + 1, scale + 1);
            sy = max(scale * (y-1) + 1, scale + 1);
            im_y_h(sx : sx+sh-2*scale-1, sy : sy+sw-2*scale-1) = ...
                shave(output, [scale, scale]);
        end
    end
    
    %% remove outside border
    im_y_h1 = shave(uint8(single(im_y_h) * 255), [scale, scale]);
    im_y_b1 = shave(uint8(single(im_y_b) * 255), [scale, scale]);
    im_y_gnd1 = shave(uint8(single(im_y_gnd) * 255), [scale, scale]);

    if size(img, 3) > 1
        im_cb_b1 = shave(uint8(single(im_cb_b) * 255), [scale, scale]);
        im_cr_b1 = shave(uint8(single(im_cr_b) * 255), [scale, scale]);
        ycbcr_h1 = cat(3, (im_y_h1), (im_cb_b1), (im_cr_b1));
        im_h1 = ycbcr2rgb(ycbcr_h1);
        ycbcr_b1 = cat(3, (im_y_b1), (im_cb_b1), (im_cr_b1));
        im_b1 = ycbcr2rgb(ycbcr_b1);
        
        im_cb_gnd1 = shave(uint8(single(im_cb_gnd) * 255), [scale, scale]);
        im_cr_gnd1 = shave(uint8(single(im_cr_gnd) * 255), [scale, scale]);
        ycbcr_gnd1 = cat(3, (im_y_gnd1), (im_cb_gnd1), (im_cr_gnd1));
        im_gnd1 = ycbcr2rgb(ycbcr_gnd1);
    else
        im_h1 = im_y_h1;
        im_b1 = im_y_b1;
        im_gnd1 = im_y_gnd1;
    end
    im_h_set{i} = im_h1;
    im_bic_set{i} = im_b1;
    im_gnd_set{i} = im_gnd1;

    %% save image files
    imwrite(im_h1, fullfile(folder_result, [image_name '_x' num2str(scale) '.png']));

    %% compute PSNR, SSIM, and IFC
    bicubic_set(i, 1)    = compute_psnr(im_y_gnd1, im_y_b1);
    srdensenet_set(i, 1) = compute_psnr(im_y_gnd1, im_y_h1);
    bicubic_set(i, 2)    = ssim_index(im_y_gnd1, im_y_b1);
    srdensenet_set(i, 2) = ssim_index(im_y_gnd1, im_y_h1);
    bicubic_set(i, 3)    = ifcvec(double(im_y_gnd1), double(im_y_b1));
    srdensenet_set(i, 3) = ifcvec(double(im_y_gnd1), double(im_y_h1));
end

%% save PSNR and SSIM metrics
avg_Bicubic = mean(bicubic_set);
avg_SRDenseNet = mean(srdensenet_set);
PSNR_set = srdensenet_set(:,1);
SSIM_set = srdensenet_set(:,2);
IFC_set  = srdensenet_set(:,3);
save(fullfile(folder_result, ['PSNR_' test_set '_x' num2str(scale) '.mat']), 'PSNR_set');
save(fullfile(folder_result, ['SSIM_' test_set '_x' num2str(scale) '.mat']), 'SSIM_set');
save(fullfile(folder_result, ['IFC_' test_set '_x' num2str(scale) '.mat']), 'IFC_set');

%% display results of Bicubic and SRDenseNet
disp('--- Bicubic');
disp('--- PSNR ----- SSIM ---- IFC ---');
bicubic_set
disp('--- SRDenseNet');
disp('--- PSNR ----- SSIM ---- IFC ---');
srdensenet_set
disp(['--- average Bicubic    = ' num2str(avg_Bicubic)]);
disp(['--- average SRDenseNet = ' num2str(avg_SRDenseNet)]);
