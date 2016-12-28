clc, clear;

%% Read in Texture Datasets
List = ReadFileNames('Kylberg');

parfor i=1:numel(List)
    
    List2(i) = regexp(List(i), '/', 'split');
  
    
end

List2 = List2';

Kylberg_Dataset = vertcat(List2{:});

%% Load Kylber Mat Texture Dataset

load('Kylberg.mat');

%% LBP
% LBP_r,p_u_2,8

% Cell Array for Feature Vectors
LBPF = cell(10, 1);

for n = 1:1 % index exceeds matrix dim
    img = LBP(imread(strcat(pwd, '/', Kylberg_Dataset{n, 1}, '/', Kylberg_Dataset{n, 2}, '/',Kylberg_Dataset{n, 3})), 2, 8);
    %figure(1), imshow(img, [0 255]);
    

    % Normalized Histogram & Mapping
    LBP_Pre = imhist(uint8(img)); 
    LBP_Pre(LBP_Pre==0) = []; % Delete Entries with 0 as element & the first entry with GV 0
    LBP_Pre(1) = [];
    LBP_Pre = LBP_Pre/sum(LBP_Pre); % Feature Vector
    LBP_Pre = LBP_Pre';
    %bar(LBP_Pre, 0.1);
    LBPF{n} = LBP_Pre;
end
%% Gabor Filter Bank
% 5 Scales & 8 Orientations
h = 1;
scales = [3, 7, 9, 21, 37];
orientations = [0, 30, 45, 60, 90, 130, 160];
GaborFilterBank = cell(35, 1);
FV = cell(10, 1);

    for i=1:size(scales, 2)
        for j = 1:size(orientations, 2)
        
            subplot(7, 5, h);
            [Real, Imag] = Gabor(scales(i), 1/10, orientations(j), 20, 0);
            GaborFilterBank{h} = Real;
            imshow(Real);
            h = h+1;
            
        end
    end
% Gabor Filter Bank Convolution  

filteredImg = cell(35, 1);


parfor n = 1:10
    img = imread(strcat(pwd, '/', Kylberg_Dataset{n, 1}, '/', Kylberg_Dataset{n, 2}, '/',Kylberg_Dataset{n, 3}));
    
    for h = 1:35
        
        subplot(7, 5, h);
        filteredImg{h} = conv2(double(img), GaborFilterBank{h}, 'same');
        imshow(filteredImg{h});
        h = h+1;
    end


% Gabor Feature Extraction
GabFeatureVec = [];
Energy = zeros(1, 35);
Mean = zeros(1, 35);

% Calculate Local Energy & Mean Amplitude
parfor i = 1:35

    Energy(i) = sum(filteredImg{i}(:).^2);
    Mean(i) = sum(abs(filteredImg{i}(:)));
end

% Normalize Feature Vector
GabFeatureVec = horzcat(Energy, Mean);
GabFeatureVec = GabFeatureVec/(sum(GabFeatureVec));
FV{n} = GabFeatureVec;
end
%% GLCM Feature Extraction

% 3 steps & 4 orientations --> 12 GLCM
delta = [1, 2, 3];
orientations = [0, 45, 90, 135];

MultiGLCM = cell(12, 1);
h = 1;

for n = 1:10
   for i=1:size(delta, 2)
        for j = 1:size(orientations, 2)
            
            img = imread(strcat(pwd, '/', Kylberg_Dataset{n, 1}, '/', Kylberg_Dataset{n, 2}, '/',Kylberg_Dataset{n, 3}));
            MultiGLCM{h} = GLCM(img, 3, i, j);
            h = h + 1;
            
        end
   end
   
   % Create Feature Vector
   FeatureGLCM = sum([MultiGLCM{:,1}], 1)/size(MultiGLCM, 1);
       
end

%% SFTA Features

% Load in images
parfor n = 1:10

    img = imread(strcat(pwd, '/', Kylberg_Dataset{n, 1}, '/', Kylberg_Dataset{n, 2}, '/',Kylberg_Dataset{n, 3}));
    SFTA_F = SFTA(img, 8);
end


%% HOG Features

% Load in images
parfor n = 1:10

    img = imread(strcat(pwd, '/', Kylberg_Dataset{n, 1}, '/', Kylberg_Dataset{n, 2}, '/',Kylberg_Dataset{n, 3}));
    F = HOG(img, 2, 8, 9, [-1 0 1], [-1 0 1]');
end