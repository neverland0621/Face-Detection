% Starter code prepared by James Hays for CS 4476, Georgia Tech
% This function should return all positive training examples (faces) from
% 36x36 images in 'train_path_pos'. Each face should be converted into a
% HoG template according to 'feature_params'. For improved performance, try
% mirroring or warping the positive training examples to augment your
% training data.

function features_pos = get_positive_features_extrafeatures(train_path_pos, feature_params)
% 'train_path_pos' is a string. This directory contains 36x36 images of
%   faces
% 'feature_params' is a struct, with fields
%   feature_params.template_size (default 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.
%      (although you don't have to make the detector step size equal a
%      single HoG cell).


% 'features_pos' is N by D matrix where N is the number of faces and D
% is the template dimensionality, which would be
%   (feature_params.template_size / feature_params.hog_cell_size)^2 * 31
% if you're using the default vl_hog parameters

% Useful functions:
% vl_hog, HOG = VL_HOG(IM, CELLSIZE)
%  http://www.vlfeat.org/matlab/vl_hog.html  (API)
%  http://www.vlfeat.org/overview/hog.html   (Tutorial)
% rgb2gray

image_files = dir( fullfile( train_path_pos, '*.jpg') ); %Caltech Faces stored as .jpg
num_images = length(image_files);
%disp(image_files)
%disp(num_images)
% placeholder to be deleted. 100 random features.

for i =  1:num_images
    consider_image = single(imread(fullfile(train_path_pos,image_files(i).name)));
    features_hog= reshape(vl_hog(consider_image,(feature_params.template_size / feature_params.hog_cell_size)),1,((feature_params.template_size / feature_params.hog_cell_size)^2 * 31));    
    [location,features_sift] = vl_dsift(single(consider_image),'Size', 16,'Step', 10,'fast');  
    [frames,features_phow] = vl_phow(single(consider_image));
    features_sift_append = reshape(features_sift,1,[]);
    features_phow_append = reshape(features_phow,1,[]);  
    features_pos(i,:) = horzcat(features_hog, features_sift_append, features_phow_append);  
    
end   

%features_pos = rand(100, (feature_params.template_size / feature_params.hog_cell_size)^2 * 31);