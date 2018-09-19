function [ outputdata ] = SRDenseNet( model, weights, inputdata )
%% test on sub-images, since too large dimension leads to blob size exceeding
[hei, wid, channel] = size(inputdata);
%% update data dimensions in the first lines of the prototxt
fidin = fopen(model, 'r+');
i = 0;
while ~feof(fidin)
    tline = fgetl(fidin);
    i = i + 1;
    newtline{i} = tline;
    if i == 9
        newtline{i} = [tline(1:11) num2str(channel)];
    elseif i == 10
        newtline{i} = [tline(1:11) num2str(wid)];
    elseif i == 11
        newtline{i} = [tline(1:11) num2str(hei)];
    end
end
fclose(fidin);
%% write the new dimension in to the prototxt
fidin = fopen(model, 'w+');
for j = 1 : i
    fprintf(fidin, '%s\n', newtline{j});
end
fclose(fidin);
%% create net and load weights
net = caffe.Net(model, weights, 'test'); 
%% feedforward computation
result = net.forward({inputdata});
outputdata = result{1};
% outputdata = outputdata';
caffe.reset_all();
end