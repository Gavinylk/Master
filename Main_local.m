%% Texture Classification with SVM
addpath(genpath('home/saraei/Dokumente/TextureClassification/'));
run vlfeat-0.9.20/toolbox/vl_setup
clc, clear;
delete(gcp('nocreate'));
c = parcluster('local');
c.NumWorkers = 30;
parpool(c, c.NumWorkers);
%% Load Kylber Mat Texture Dataset

load('Kylberg.mat');
load('imdb.mat');
%% Prepare Classes for labeling and seperation into train and test groups of Kylberg
[M N] = size(Kylberg_Dataset);
temp = cell(M/12, 1);
cnt = 1;

for i = 1:12:M
        
    temp(cnt) = Kylberg_Dataset(i, 3);
    cnt = cnt + 1;
 
end

O = size(temp);
cnt = 1;

for j = 1:10:O
   
        train_Label(cnt) = temp(j,1);
        cnt = cnt + 1;
    
end


cnt = 1;

for k = 2:5:O
     
     test_Label(cnt) = temp(k, 1);
     cnt = cnt + 1;
     
end
    
train_Label = train_Label';
test_Label = test_Label';


% Labeling classes
cnt = 1;

for i = 1:size(test_Label, 1)-1
    
    next = strtok(test_Label(i+1), '-');
    actual = strtok(test_Label(i), '-');
    
    if(strcmpi(next, actual)) 
        
        test(i) = cnt;
        test_name(i) = actual;
    else
        test(i) = cnt;
        test_name(i) = actual;
        cnt = cnt + 1;
        
        
    end
    
end

cnt = 1;

for i = 1:size(train_Label, 1)-1
    
    next = strtok(train_Label(i+1), '-');
    actual = strtok(train_Label(i), '-');
    
    if(strcmpi(next, actual)) 
     
        train(i) = cnt;
        train_name(i) = actual;
    else
        train(i) = cnt;
        train_name(i) = actual;
        cnt = cnt + 1;
        
        
    end
    
end

test = test';
train = train';
test_name = test_name';
train_name = train_name';

%% DTD Preparation
cnt_train = 1;
cnt_test = 1;
train_name = cell(1880, 1);
test_name = cell(1880, 1);

for i = 1:size(images.set, 2)
    
    if(images.set(i) == 1)
    
        train_name{cnt_train} = images.name{i};
        train_label(cnt_train) = images.class(i);
        cnt_train = cnt_train + 1;
    end
    
    if(images.set(i) == 3)
    
        test_name{cnt_test} = images.name{i};
        test_label(cnt_test) = images.class(i);
        cnt_test = cnt_test + 1;
    end
end
train_label = train_label';
test_label = test_label';
%% Generate Feature Vectors 
%% LBP

% Kylberg
% LBP_r,p_u_2,8

% Cell Array for Feature Vectors
% LBPF_train = cell(size(train, 1), 1);
% LBPF_test = cell(size(test, 1), 1);
% 
% parfor n = 1:size(test, 1) 
%     img = LBP(imread(strcat(pwd, '/Kylberg', '/', test_name{n}, '/',test_Label{n})), 2, 8);
%     %figure(1), imshow(img, [0 255]);
%     
% 
%     % Normalized Histogram & Mapping
%     LBP_Pre = imhist(uint8(img)); 
%     LBP_Pre(LBP_Pre==0) = []; % Delete Entries with 0 as element & the first entry with GV 0
%     LBP_Pre(1) = [];
%     LBP_Pre = LBP_Pre/sum(LBP_Pre); % Feature Vector
%     LBP_Pre = LBP_Pre';
%     LBPF_test{n} = LBP_Pre;
%     X = sprintf('%d of %d completed', n, size(test, 1));
%     disp(X);
% end

% DTD
 
% LBPF_train = cell(size(train_label, 1), 1);
% LBPF_test = cell(size(test_label, 1), 1);
% 
% parfor_progress(size(train_label, 1));
% 
% 
% parfor n = 1:size(train_label, 1) 
%     
%     LBPF_train{n} = LBP(imresize(rgb2gray(imread(strcat(pwd, '/DTD/images', '/', train_name{n}))), [512 512]), 2, 8, 1);
%     LBPF_test{n} = LBP(imresize(rgb2gray(imread(strcat(pwd, '/DTD/images', '/', test_name{n}))), [512 512]), 2, 8, 1);
%      
%    parfor_progress;
%    
% end
% 
% parfor_progress(0);
% % 
% % % Save trained Data to mat
% save('LBPF_train_8_2_DTD_noHist.mat','LBPF_train', '-v7.3'); 
% save('LBPF_test_8_2_DTD_noHist.mat','LBPF_test', '-v7.3'); 

%% Gabor Filter Bank
% % 5 Scales & 8 Orientations
 % FV_train = cell(size(train_name, 1), 1);
 % FV_test = cell(size(test_name, 1), 1);
 % gaborArray = gaborFilterBank(5,8,39,39);
 
 % parfor_progress(size(train_name, 1));

% parfor n = 1:size(train_name, 1) 
    
    % %FV_train{n} = gaborFeatures(imresize(rgb2gray(imread(strcat(pwd, '/DTD/images', '/', train_name{n}))), [512 512]),gaborArray,4,4);  
	% FV_train{n} = gaborFeatures(imread(strcat(pwd, '/Kylberg', '/', train_name{n}, '/',train_Label{n})),gaborArray);
	% parfor_progress;
    
% end

% parfor_progress(0);

% parfor_progress(size(test_name, 1));

% parfor n = 1:size(test_name, 1) 
    
    % %FV_test{n} = gaborFeatures(imresize(rgb2gray(imread(strcat(pwd, '/DTD/images', '/', test_name{n}))), [512 512]),gaborArray,4,4);  
    % FV_test{n} = gaborFeatures(imread(strcat(pwd, '/Kylberg', '/', test_name{n}, '/',test_Label{n})),gaborArray);
	% parfor_progress;
    
% end

% parfor_progress(0);

%% GLCM Feature Extraction

% 3 steps & 4 orientations --> 12 GLCM
% Offset = [0 1; -1 1; -1 0; -1 -1;0 2; -2 2; -2 0; -2 -2;0 3; -3 3; -3 0; -3 -3];

% FV_train = cell(size(train_name, 1), 1);
% FV_test = cell(size(test_name, 1), 1);


% MultiGLCM = cell(12, 1);

% parfor_progress(size(train_name, 1));

% parfor n = 1:size(train_name, 1)
    
    % MultiGLCM = graycomatrix(imread(strcat(pwd, '/Kylberg', '/', train_name{n}, '/',train_Label{n})), 'NumLevels', 8,'Offset', Offset,'Symmetric', true);
    % [a b c] = size(MultiGLCM);
    % MultiGLCM = mat2cell(MultiGLCM, a, b, ones(1,c));
   
        % for i = 1:size(Offset, 1)
            
            % MultiGLCM{i} = reshape(MultiGLCM{i}./sum(MultiGLCM{i}(:)).',1,[]);
        
        % end
   
   % % Create Feature Vector
   
    % MultiGLCM = cell2mat(squeeze(MultiGLCM));   
    % FV_train{n} = sum(MultiGLCM)./size(MultiGLCM, 1);
    % parfor_progress;
       
% end

% parfor_progress(0);

% parfor_progress(size(test_name, 1));

% parfor n = 1:size(test_name, 1)
    
    % MultiGLCM = graycomatrix(imread(strcat(pwd, '/Kylberg', '/', test_name{n}, '/',test_Label{n})), 'NumLevels', 8,'Offset', Offset,'Symmetric', true);
    % [a b c] = size(MultiGLCM);
    % MultiGLCM = mat2cell(Mul[cv bestc bestg] = Classify(train_label, trainData, 3, [5:1:5], [1:1:1], 1);tiGLCM, a, b, ones(1,c));
   
        % for i = 1:size(Offset, 1)
            
            % MultiGLCM{i} = reshape(MultiGLCM{i}./sum(MultiGLCM{i}(:)).',1,[]);
        
        % end
   
   % % Create Feature Vector
   
    % MultiGLCM = cell2mat(squeeze(MultiGLCM));   
    % FV_test{n} = sum(MultiGLCM)./size(MultiGLCM, 1);
    % parfor_progress;
       
% end

% parfor_progress(0);

%% HOG Feature Extraction
% FV_train = cell(size(train_name, 1), 1);
% FV_test = cell(size(test_name, 1), 1);
 
% parfor_progress(size(train_name, 1));

% parfor n = 1:size(train_name, 1)

	% %FV_train{n} = HOG(imresize(rgb2gray(imread(strcat(pwd, '/DTD/images', '/', train_name{n}))), [512 512]), 2, 8, 9, [-1 0 1], [-1 0 1]');
	% FV_train{n} = extractHOGFeatures(imread(strcat(pwd, '/Kylberg', '/', train_name{n}, '/',train_Label{n})),'CellSize',[6 6]);
	% parfor_progress;
	
% end

% parfor_progress(0);

% parfor_progress(size(test_name, 1));

% parfor n = 1:size(test_name, 1)

	% %FV_test{n} = HOG(imresize(rgb2gray(imread(strcat(pwd, '/DTD/images', '/', test_name{n}))), [512 512]), 2, 8, 9, [-1 0 1], [-1 0 1]');
	% FV_test{n} = extractHOGFeatures(imread(strcat(pwd, '/Kylberg', '/', test_name{n}, '/',test_Label{n})),'CellSize',[6 6]);
	% parfor_progress;
	
% end

% parfor_progress(0);

%% SFTA Extraction
% FV_train = cell(size(train_name, 1), 1);
% FV_test = cell(size(test_name, 1), 1);
%  
% parfor_progress(size(train_name, 1));
% 
% parfor n = 1:size(train_name, 1)
% 
% 	FV_train{n} = SFTA(imresize(rgb2gray(imread(strcat(pwd, '/DTD/images', '/', train_name{n}))), [512 512]), 8);
% 	%FV_train{n} = SFTA(imread(strcat(pwd, '/Kylberg', '/', train_name{n}, '/',train_Label{n})),8);
%     parfor_progress;
% 	
% end
% 
% parfor_progress(0);
% 
% parfor_progress(size(test_name, 1));
% 
% parfor n = 1:size(test_name, 1)
% 
% 	FV_test{n} = SFTA(imresize(rgb2gray(imread(strcat(pwd, '/DTD/images', '/', test_name{n}))), [512 512]), 8);
% 	%FV_test{n} = SFTA(imread(strcat(pwd, '/Kylberg', '/', test_name{n}, '/',test_Label{n})),8);
%     parfor_progress;
% 	
% end
% 
% parfor_progress(0);

%% Cross Feature Extraction

%Cell Array for Feature Vectors
% FV_train = cell(size(train_name, 1), 1);
% FV_test = cell(size(test_name, 1), 1);
% 
% parfor_progress(size(train_name, 1));
% 
% parfor n = 1:size(train_name, 1) 
%    
%         %FV_train{n} = CrossFeature_sparse(imresize(rgb2gray(imread(strcat(pwd, '/DTD/images', '/', train_name{n}))), [512 512]), 9, 3);
%         FV_train{n} = CrossFeature_sparse(imread(strcat(pwd, '/Kylberg', '/', train_name{n}, '/',train_Label{n})), 9, 3);
% 		parfor_progress;
%     
%         
%     
%     
%     
%     
% end
% 
% parfor_progress(0);
% 
% parfor_progress(size(test_name, 1));
% 
% parfor n = 1:size(test_name, 1)
%     
%         %FV_test{n} = CrossFeature_sparse(imresize(rgb2gray(imread(strcat(pwd, '/DTD/images', '/', test_name{n}))), [512 512]), 9, 3);
%         FV_test{n} = CrossFeature_sparse(imread(strcat(pwd, '/Kylberg', '/', test_name{n}, '/',test_Label{n})), 9, 3);
% 		parfor_progress;
%     
% end
% parfor_progress(0);
 
% Save trained Data to mat
% save('CF_sparse_train_Stride3_ROI9_L2.mat','CF_train', '-v7.3'); 
% save('CF_sparse_test_Stride3_ROI9_L2.mat','CF_test', '-v7.3'); 

%% Local Fourier Descriptor

% FV_train = cell(size(train_name, 1), 1);
% FV_test = cell(size(test_name, 1), 1);
% 
% parfor_progress(size(train_name, 1));
% 
% parfor n = 1:size(train_name, 1) 
%     
%     FV_train{n} = localFourier(double(imresize(rgb2gray(imread(strcat(pwd, '/DTD/images', '/', train_name{n}))), [512 512])), 3, 3);
%     parfor_progress;
%     
% end
% 
% parfor_progress(0);
% 
% parfor_progress(size(test_name, 1));
% 
% parfor n = 1:size(test_name, 1)
%     
%     FV_test{n} = localFourier(double(imresize(rgb2gray(imread(strcat(pwd, '/DTD/images', '/', test_name{n}))), [512 512])), 3, 3);
%     parfor_progress;
% end
% 
% parfor_progress(0);



%% Local STD Descriptor

% FV_train = cell(size(train_name, 1), 1);
% FV_test = cell(size(test_name, 1), 1);
% 
% parfor_progress(size(train_name, 1));
% 
% parfor n = 1:size(train_name, 1) 
%     
%     FV_train{n} = stdClassify(double(imresize(rgb2gray(imread(strcat(pwd, '/DTD/images', '/', train_name{n}))), [512 512])), 3, 3);
%     parfor_progress;
%     
% end
% 
% parfor_progress(0);
% 
% parfor_progress(size(test_name, 1));
% 
% parfor n = 1:size(test_name, 1)
%     
%     FV_test{n} = stdClassify(double(imresize(rgb2gray(imread(strcat(pwd, '/DTD/images', '/', test_name{n}))), [512 512])), 3, 3);
%     parfor_progress;
% end
% 
% parfor_progress(0);


%% Multi Feature Descriptor (GradientMag, GradientPhas, Fourier, SD)

% FV_train = cell(size(train_name, 1), 1);
% FV_test = cell(size(test_name, 1), 1);
% 
% parfor_progress(size(train_name, 1));
% 
% parfor n = 1:size(train_name, 1) 
%     
%     FV_train{n} = MultiFeature(double(imresize(rgb2gray(imread(strcat(pwd, '/DTD/images', '/', train_name{n}))), [512 512])), 3, 1, 8, [0.1:0.01:1]);
%     parfor_progress;
%     
% end
% 
% parfor_progress(0);
% 
% parfor_progress(size(test_name, 1));
% 
% parfor n = 1:size(test_name, 1)
%     
%     FV_test{n} = MultiFeature(double(imresize(rgb2gray(imread(strcat(pwd, '/DTD/images', '/', test_name{n}))), [512 512])), 3, 1, 8, [0.1:0.01:1]);
%     parfor_progress;
% end
% 
% parfor_progress(0);

%% SIFT Feature Extraction

% FV_train = cell(size(train_name, 1), 1);
% FV_test = cell(size(test_name, 1), 1);
% 
% parfor_progress(size(train_name, 1));
% 
% parfor n = 1:size(train_name, 1) 
%     
%     FV_train{n} = Fisher(double(SIFT(single(histeq(imresize(rgb2gray(imread(strcat(pwd, '/DTD/images', '/', train_name{n}))), [512 512]))))), 256);
%     FV_test{n} = Fisher(double(SIFT(single(histeq(imresize(rgb2gray(imread(strcat(pwd, '/DTD/images', '/', test_name{n}))), [512 512]))))), 256);
%     parfor_progress;
%     
% end
% 
% %save('Fischer_SIFT_train_[4 6 8 10 12 14 16]_Step_5.mat','FV_train', '-v7.3');
% %save('Fisher_SIFT_test_[4 6 8 10 12 14 16]_Step_5.mat','FV_test', '-v7.3');
% parfor_progress(0);

%% MSER Feature Extraction

% FV_train = cell(size(train_name, 1), 1);
% FV_test = cell(size(test_name, 1), 1);
% 
% parfor_progress(size(train_name, 1));
% 
% parfor n = 1:size(train_name, 1) 
%     
%     FV_train{n} = mser(histeq(imresize(rgb2gray(imread(strcat(pwd, '/DTD/images', '/', train_name{n}))), [512 512])));
%     FV_test{n} = mser(histeq(imresize(rgb2gray(imread(strcat(pwd, '/DTD/images', '/', test_name{n}))), [512 512])));
%     
%     parfor_progress;
%     
% end
% 
% parfor_progress(0);

%% Local Fisher Feature Extraction
FV_train = cell(size(train_name, 1), 1);
FV_test = cell(size(test_name, 1), 1);

parfor_progress(size(train_name, 1));

parfor n = 1:size(train_name, 1) 
    
    FV_train{n} = FisherPatches(histeq(imresize(rgb2gray(imread(strcat(pwd, '/DTD/images', '/', train_name{n}))), [512 512])), 49,21,3,49, 10);
    FV_test{n} = FisherPatches(histeq(imresize(rgb2gray(imread(strcat(pwd, '/DTD/images', '/', test_name{n}))), [512 512])), 49,21,3,49, 10);
    
    parfor_progress;
    
end

parfor_progress(0);

%% Fisher GMM Feature Vector Extraction

% Cell Array for Feature Vectors
% FV_train_gmm = cell(size(train_name, 1), 3);
% FV_train = cell(size(train_name, 1), 1);
% FV_test = cell(size(test_name, 1), 1);
% % % load('SFTA_train_DTD.mat');
% % % load('SFTA_test_DTD.mat');
% 
% 
% 
% parfor_progress(size(FV_train, 1));
%  
% for n = 1:size(FV_train, 1) 
%     img = double(imresize(rgb2gray(imread(strcat(pwd, '/DTD/images', '/', train_name{n}))), [512 512]));  
%     [FV_train_gmm{n, :}] = vl_gmm(img, 256);
%     FV_train{n} = vl_fisher(img, FV_train_gmm{n, 1}, FV_train_gmm{n, 2}, FV_train_gmm{n, 3}, 'Improved', 'Fast')';
%     %FV_train{n} = Fisher(double(imread(strcat(pwd, '/Kylberg', '/', train_name{n}, '/',train_Label{n}))),47);
%     parfor_progress;
%      
% end
% 
% parfor_progress(0);
% 
% parfor_progress(size(test_name, 1));
% 
% parfor n = 1:size(test_name, 1)
%     
%        img = double(imresize(rgb2gray(imread(strcat(pwd, '/DTD/images', '/', test_name{n}))), [512 512]));
%        FV_test{n} = vl_fisher(img, FV_train_gmm{n, 1}, FV_train_gmm{n, 2}, FV_train_gmm{n, 3}, 'Improved', 'Fast')';
% 
%        
%        parfor_progress;
%     
% end
% parfor_progress(0);

% Save trained Data to mat
% save('train_1.mat','FV_train'); 
% save('CF_test_Stride3_ROI9_L2.mat','CF_test'); 
%% Creating Training Matrix for SVM
%[FV_train FV_test] = adaptFV(FV_train, FV_test);

trainData = cell2mat(FV_train);
testData = cell2mat(FV_test); 

%% Feature Vector dimension reduction with PCA

% [eigenvectors, projected_data, eigenvalues] = pcaecon(trainData, size(trainData, 1));
% [foo, feature_idx] = sort(eigenvalues, 'descend');
% trainData = projected_data(:, feature_idx(1:1880));
% 
% [eigenvectors, projected_data, eigenvalues] = pcaecon(testData, size(testData, 1));
% [foo, feature_idx] = sort(eigenvalues, 'descend');
% testData = projected_data(:, feature_idx(1:1880));

%Norm data to [0 1]
trainData = abs(trainData)./max(abs(trainData(:)));
testData = abs(testData)./max(abs(testData(:)));
trainData = double(trainData);
testData = double(testData);
%% Classifying

[cv bestc bestg] = Classify(train_label, trainData, 3, [5:1:5], [-3:1:-3], 1);

%[bestc, bestg, bestcv] = automaticParameterSelection(double(train_label), double(trainData), 3);
% %% Train the SVM in one-vs-rest (OVR) mode
% % #######################
bestParam = ['-q -c ', num2str(bestc), ' -g ', num2str(bestg)];
model = ovrtrain(train_label, trainData, bestParam);
% bestParam = ['-q -c ', num2str(bestc), ' -t 0',' '];
% model = ovrtrain(train_label, [(1:size(train_label, 1))' trainData*trainData'], bestParam);


% %% #######################
% % Classify samples using OVR model
% % #######################
% [predict_label, accuracy, prob_values] = ovrpredict(test_label, [(1:size(test_label, 1))' testData*trainData'], model);
[predict_label, accuracy, prob_values] = ovrpredict(test_label, testData, model);
fprintf('Accuracy = %g%%\n', accuracy * 100);