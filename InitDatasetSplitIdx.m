image_dir = 'D:\matlabcode\DCA\UCMERCED\';
ratio_train_test = 0.90;
fnames = dir(image_dir);
num_files = size(fnames,1);
num_class = num_files-2;
num_img_per_class = zeros(num_class,1);
label = [];
train_label = [];
test_label = [];
R_train_all = [];
R_test_all = [];
R_all = [];
for img = 1:num_files
    if( (strcmp(fnames(img).name , '.')==1) || (strcmp(fnames(img).name , '..')==1))
    continue;
    end
    subfoldername = fnames(img).name;
    filename_tif = dir(fullfile(strcat(image_dir,subfoldername),'*.tif '));
    num_img_per_class(img-2) = length(filename_tif);
    label = [label; (img-2)*ones(num_img_per_class(img-2),1)];
    img
end
for ic = 1:num_class
            R = randperm(num_img_per_class(ic));
            num_train = fix(num_img_per_class(ic)*ratio_train_test);
            R_train = R(1:num_train);
            R_test = R(num_train+1:end);
            R_train_all=[R_train_all,R_train+sum(num_img_per_class(1:ic-1,1))];
            R_test_all=[R_test_all,R_test+sum(num_img_per_class(1:ic-1,1))];
            R_all=[R_all,R];
end
train_label = label(R_train_all);
test_label = label(R_test_all);
LabeledIndex=R_train_all';
UnlabeledIndex=R_test_all';
save '21class\DatasetSplitIdx9.mat' LabeledIndex UnlabeledIndex
%DatasetSplitIdx.mat  LabeledIndex UnlabeledIndex

