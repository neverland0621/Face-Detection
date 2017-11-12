% Sliding window face detection with linear SVM. 
% All code by James Hays, except for pieces of evaluation code from Pascal
% VOC toolkit. Images from CMU+MIT face database, CalTech Web Face
% Database, and SUN scene database.

% Code structure:
% proj5.m <--- You code parts of this
%  + get_positive_features.m  <--- You code this
%  + get_random_negative_features.m  <--- You code this
%   [classifier training]   <--- You code this
%  + report_accuracy.m
%  + run_detector.m  <--- You code this
%    + non_max_supr_bbox.m
%  + evaluate_all_detections.m
%    + VOCap.m
%  + visualize_detections_by_image.m
%  + visualize_detections_by_image_no_gt.m
%  + visualize_detections_by_confidence.m

% Other functions. You don't need to use any of these unless you're trying
% to modify or build a test set:
%  Training and Testing data related functions:
%   test_scenes/visualize_cmumit_database_landmarks.m
%   test_scenes/visualize_cmumit_database_bboxes.m
%   test_scenes/cmumit_database_points_to_bboxes.m %This function converts
%    from the original MIT+CMU test set landmark points to Pascal VOC
%    annotation format (bounding boxes).

%   caltech_faces/caltech_database_points_to_crops.m %This function extracts
%    training crops from the Caltech Web Face Database. The crops are
%    intentionally large to contain most of the head, not just the face.
%    The test_scene annotations are likewise scaled to contain most of the
%    head.

% set up paths to VLFeat functions. 
% See http://www.vlfeat.org/matlab/matlab.html for VLFeat Matlab documentation
% This should work on 32 and 64 bit versions of Windows, MacOS, and Linux
close all
clear
run('../vlfeat-0.9.20/toolbox/vl_setup')

[~,~,~] = mkdir('visualizations');

data_path = '../data/'; 
train_path_pos = fullfile(data_path, 'caltech_faces/Caltech_CropFaces'); %Positive training examples. 36x36 head crops
non_face_scn_path = fullfile(data_path, 'train_non_face_scenes'); %We can mine random or hard negatives from here
test_scn_path = fullfile(data_path,'test_scenes/test_jpg'); %CMU+MIT test scenes
% test_scn_path = fullfile(data_path,'extra_test_scenes'); %Bonus scenes
label_path = fullfile(data_path,'test_scenes/ground_truth_bboxes.txt'); %the ground truth face locations in the test set

%The faces are 36x36 pixels, which works fine as a template size. You could
%add other fields to this struct if you want to modify HoG default
%parameters such as the number of orientations, but that does not help
%performance in our limited test.
feature_params = struct('template_size', 36, 'hog_cell_size', 6);


%% Step 1. Load positive training crops and random negative examples
%YOU CODE 'get_positive_features' and 'get_random_negative_features'

features_pos = get_positive_features( train_path_pos, feature_params );
%%Extra features positive-sift, hog and phow
%features_pos = get_positive_features_extrafeatures( train_path_pos, feature_params );
%disp(features_pos) 
num_negative_examples = 10000; %Higher will work strictly better, but you should start with 10000 for debugging

features_neg = get_random_negative_features( non_face_scn_path, feature_params, num_negative_examples);
%%Extra features negative-sift, hog and phow
%features_neg = get_random_negative_extrafeatures( non_face_scn_path, feature_params, num_negative_examples);

    
%% step 2. Train Classifier
% Use vl_svmtrain on your training features to get a linear classifier
% specified by 'w' and 'b'
% [w b] = vl_svmtrain(X, Y, lambda) 
% http://www.vlfeat.org/sandbox/matlab/vl_svmtrain.html
% 'lambda' is an important parameter, try many values. Small values seem to
% work best e.g. 0.0001, but you can try other values

%YOU CODE classifier training. Make sure the outputs are 'w' and 'b'.
%w = rand((feature_params.template_size / feature_params.hog_cell_size)^2 * 31,1); %placeholder, delete
%b = rand(1); %placeholder, delete
lambda = 0.0001;
[pos_height,pos_width] = size(features_pos);
[neg_height,neg_width] = size(features_neg);

features_all = [features_pos;features_neg];
%disp(size(features_all))
feature_labels_pos = ones(pos_height,1);
%disp(size(feature_labels_pos))
feature_labels_neg = ones(neg_height,1).*-1;
%disp(size(feature_labels_neg))
labels_all = [feature_labels_pos;feature_labels_neg];

%SVM
[w,b] = vl_svmtrain(features_all',labels_all,lambda);

%SVM for extra features

%[w,b] = vl_svmtrain(double(features_all'),labels_all,lambda);
%Decision Tree
%tree = fitctree(features_all,labels_all);
%Ensemble
%temp1 = templateTree('MaxNumSplits',1,'minleaf',100,'MinParent',80);
%ens = fitensemble(features_all,labels_all,'AdaBoostM1',3,temp1);

%view(ens.Trained{1},'Mode','graph')


%% step 3. Examine learned classifier
% You don't need to modify anything in this section. The section first
% evaluates _training_ error, which isn't ultimately what we care about,
% but it is a good sanity check. Your training error should be very low.

fprintf('Initial classifier performance on train data:\n')
%confidences = [features_pos; features_neg]*w + b;
confidences = double([features_pos; features_neg])*w + b;
%confidences = [features_pos; features_neg].*w + b;
label_vector = [ones(size(features_pos,1),1); -1*ones(size(features_neg,1),1)];
[tp_rate, fp_rate, tn_rate, fn_rate] =  report_accuracy( confidences, label_vector );

% Visualize how well separated the positive and negative examples are at
% training time. Sometimes this can idenfity odd biases in your training
% data, especially if you're trying hard negative mining. This
% visualization won't be very meaningful with the placeholder starter code.
non_face_confs = confidences( label_vector < 0);
face_confs     = confidences( label_vector > 0);
figure(2); 
plot(sort(face_confs), 'g'); hold on
plot(sort(non_face_confs),'r'); 
plot([0 size(non_face_confs,1)], [0 0], 'b');
hold off;

% Visualize the learned detector. This would be a good thing to include in
% your writeup!
n_hog_cells = sqrt(length(w) / 31); %specific to default HoG parameters
%n_hog_cells = ceil(sqrt(length(w) / 31)); 

imhog = vl_hog('render', single(reshape(w, [n_hog_cells n_hog_cells 31])), 'verbose') ;


figure(3); imagesc(imhog) ; colormap gray; set(3, 'Color', [.988, .988, .988])

pause(0.1) %let's ui rendering catch up
hog_template_image = frame2im(getframe(3));
% getframe() is unreliable. Depending on the rendering settings, it will
% grab foreground windows instead of the figure in question. It could also
% return a partial image.
imwrite(hog_template_image, 'visualizations/hog_template.png')
    
 
%% step 4. (optional extra credit) Mine hard negatives
% Mining hard negatives is graduate credit / extra credit. You can get very
% good performance by using random negatives, so hard negative mining is
% somewhat unnecessary for face detection. If you implement hard negative
% mining, you probably want to modify 'run_detector', run the detector on
% the images in 'non_face_scn_path', and keep all of the features above
% some confidence level. Hard negative mining would probably be more
% important if you had a strict budget of negative training examples or a
% more expressive, non-linear classifier that can benefit from more
% trianing data.
[bboxes, confidences, image_ids_new] = run_detector(non_face_scn_path, w, b, feature_params);
feature_size = feature_params.template_size;
features_hard_neg = zeros(size(bboxes,1), (feature_params.template_size / feature_params.hog_cell_size)^2 * 31);
k = 1;
for i = 1:size(bboxes,1)
      
    img = im2single(rgb2gray(imread(strcat(non_face_scn_path, '/', image_ids_new{i}))));
    img = single(img)/255;
    if(size(img,3) > 1)
        img = rgb2gray(img);
    end
    min_x = bboxes(i,1);
    min_y = bboxes(i,2);
    X_len = bboxes(i,3)-min_x; 
    Y_len = bboxes(i,4)-min_y;    
    cropped_image = imcrop(img,[min_x min_y X_len Y_len]);       
    [y, x] = size(cropped_image);
    %size(cropped_image)
    some_value = 36;    
    for j = 1:5
        image_final = cropped_image(randi(y-some_value+1)+(0:some_value-1),randi(x-some_value+1)+(0:some_value-1));                
        features_hard_neg(k,:) = reshape(vl_hog(image_final, feature_params.hog_cell_size), 1, ((feature_params.template_size / feature_params.hog_cell_size)^2 * 31));
        k = k+1;
    end    

end
features_neg_all = [features_neg ;features_hard_neg];
%features_neg_all = features_neg_all(1:5000,:);
lambda = 0.0001;
X_new  = [features_pos; features_neg_all];
feature_labels_pos = ones(size(features_pos,1),1);
feature_labels_neg = ones(size(features_neg_all,1),1).*-1;
labels_all = [feature_labels_pos;feature_labels_neg];

[w_new, b_new] = vl_svmtrain(transpose(X_new), labels_all, lambda);




%% Step 5. Run detector on test set.
% YOU CODE 'run_detector'. Make sure the outputs are properly structured!
% They will be interpreted in Step 6 to evaluate and visualize your
% results. See run_detector.m for more details.
%%SVM
[bboxes, confidences, image_ids] = run_detector(test_scn_path, w, b, feature_params);
%%SVM with hard mining
%[bboxes, confidences, image_ids] = run_detector(test_scn_path, w_new, b_new, feature_params);
%%SVM with extra features - sift , hog and phow
%[bboxes, confidences, image_ids] = run_detector_extrafeatures(test_scn_path, w, b, feature_params);
%%Decision Tree
%[bboxes, confidences, image_ids] = run_detector_tree(test_scn_path, ens, feature_params);
%For HOG
%[bboxes, confidences, image_ids] = run_detector_hog(test_scn_path, w,b, feature_params);
% run_detector will have (at least) two parameters which can heavily
% influence performance -- how much to rescale each step of your multiscale
% detector, and the threshold for a detection . If your recall rate is low
% and your detector still has high precision at its highest recall point,
% you can improve your average precision by reducing the threshold for a
% positive detection.
%final values
%bboxes = [bboxes1;bboxes2];
%confidences = [confidences1;confidences2];
%image_ids = [image_ids1;image_ids2];

%% Step 6. Evaluate and Visualize detections
% These functions require ground truth annotations, and thus can only be
% run on the CMU+MIT face test set. Use visualize_detectoins_by_image_no_gt
% for testing on extra images (it is commented out below).

% Don't modify anything in 'evaluate_detections'!
[gt_ids, gt_bboxes, gt_isclaimed, tp, fp, duplicate_detections] = ...
    evaluate_detections(bboxes, confidences, image_ids, label_path);

visualize_detections_by_image(bboxes, confidences, image_ids, tp, fp, test_scn_path, label_path)
% visualize_detections_by_image_no_gt(bboxes, confidences, image_ids, test_scn_path)

% visualize_detections_by_confidence(bboxes, confidences, image_ids, test_scn_path, label_path);

% performance to aim for
% random (stater code) 0.001 AP
% single scale ~ 0.2 to 0.4 AP
% multiscale, 6 pixel cell size and detector step ~ 0.83 AP
% multiscale, 4 pixel cell size and detector step ~ 0.89 AP
% multiscale, 3 pixel cell size and detector step ~ 0.92 AP