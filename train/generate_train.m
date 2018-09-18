clear; close all;
addpath util;
%% settings
folder = {
    'data/BSDS200';
%     'data/BSDS200-aug';
%     'data/General-100-aug'; 
%     'data/T91-aug';
%     'data/Train_291';
%     '/ext/xueshengke/DIV2K_train_HR';
%     '/ext/xueshengke/DIV2K_valid_HR';
};
savepath = 'train.h5';%
scale = 4;  % x2, x2
size_input = 25; % 
size_label = scale * size_input; % output = (input - 1) * stride + kernel - 2 * padding
stride = size_input;    % stride = input means no overlap
batch_size = 32;

%% initialization
data = zeros(size_input, size_input, 1, 1);
label = zeros(size_label, size_label, 1, 1);
padding = abs(size_input - size_label)/2;
count = 0;

%% generate data
for t = 1 : length(folder)
    
file_bmp = dir(fullfile(folder{t},'*.bmp'));
file_png = dir(fullfile(folder{t},'*.png'));
filepaths = [file_bmp; file_png];

for i = 1 : length(filepaths)
    
    image = imread(fullfile(folder{t}, filepaths(i).name));
    image = rgb2ycbcr(image);
    image = im2double(image(:, :, 1));
    
    im_label = modcrop(image, scale);
    im_input = imresize(im_label, 1/scale, 'bicubic');
    [hei,wid] = size(im_input);
    
    for x = 1 : stride : hei - size_input + 1
        for y = 1 : stride : wid - size_input + 1

            locx = scale * (x - 1) + 1;
            locy = scale * (y - 1) + 1;
%             locx = floor(scale * (x + (size_input - 1)/2) - (size_label + scale)/2 + 1);
%             locy = floor(scale * (y + (size_input - 1)/2) - (size_label + scale)/2 + 1);
            
            subim_input = im_input(x : x + size_input - 1, y : y + size_input - 1);
            subim_label = im_label(locx : locx + size_label - 1, locy : locy + size_label - 1);
            
            count = count + 1;
            data(:, :, 1, count) = subim_input;
            label(:, :, 1, count) = subim_label;
        end
    end
end

end
order = randperm(count);
data = data(:, :, 1, order);
label = label(:, :, 1, order); 

%% writing to HDF5
chunksz = batch_size;
created_flag = false;
totalct = 0;

for batchno = 1:floor(count/chunksz)
    last_read = (batchno-1)*chunksz;
    batchdata = data(:,:,1,last_read+1:last_read+chunksz); 
    batchlabs = label(:,:,1,last_read+1:last_read+chunksz);

    startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end
h5disp(savepath);

