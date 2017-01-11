% parfor_progress(1880);
% 
% parfor i = 1:1880
%    
%    LBPF_train{i} = LBP(logsample(double(imresize(rgb2gray(imread(strcat('/home/saraei/Documents/MATLAB/TextureClassification/DTD/images', '/', train_name{i}))), [512 512])), 5, 256, 256, 256, [], 300), 2, 8, 0);
%    LBPF_test{i} = LBP(logsample(double(imresize(rgb2gray(imread(strcat('/home/saraei/Documents/MATLAB/TextureClassification/DTD/images', '/', test_name{i}))), [512 512])), 5, 256, 256, 256, [], 300), 2, 8, 0);
%    parfor_progress;
% end
% 
% parfor_progress(0);
% 
% %%
% LBPF_train_Cat = cat(3, LBPF_train{:});
% 
% 
% %%
% [m_new c_new p_new] = vl_gmm(imgCat, 256);

%%
% numPoints = 1000 ;
% dimension = 2 ;
% data = rand(dimension,numPoints) ;
%img121 = LBP(logsample(double(imresize(rgb2gray(imread(strcat('/home/saraei/Documents/MATLAB/TextureClassification/DTD/images', '/', train_name{121}))), [512 512])), 5, 256, 256, 256, [], 300), 1, 8, 1);
%img1 = LBP(logsample(double(imresize(rgb2gray(imread(strcat('/home/saraei/Documents/MATLAB/TextureClassification/DTD/images', '/', train_name{1}))), [512 512])), 5, 256, 256, 256, [], 300), 1, 8, 1);
% for i = 1:47
%    
%     img = LBP(logsample(double(imresize(rgb2gray(imread(strcat('/home/saraei/Documents/MATLAB/TextureClassification/DTD/images', '/', train_name{G(i)}))), [512 512])), 5, 256, 256, 256, [], 300), 2, 8, 1);
%    data(i,:) = img(:);
% end

%%
% maxSize = max(size(img1, 2), size(img121, 2));
% img1(:,maxSize) = 0;
% img121(:,maxSize) = 0;


% numClusters = 47;


% Run EM starting from the given parameters
% [means,covariances,priors,ll,posteriors] = vl_gmm(data, numClusters, 'NumRepetitions', 1);

% Create Image Patches
% 
% blockStride = 21;
% blockSize = 21;
% 
% 
% % Get ImageSize
% [X Y] = size(img);
% eps = 0.01;
% k = 1;
% 
% for i = 1:blockStride:X-blockSize
%     for j = 1:blockStride:Y-blockSize
%         
%         % Zero Mean & Illumination Invariance
%         imgBlock = img(i:i+blockSize-1, j:j+blockSize-1);
%         imgBlock = double(imgBlock) - double(mean(imgBlock(:)));
%         
%         % Avoid division through 0
%         block{k} = double(abs(imgBlock)) / (double(std(double(abs(imgBlock(:)))))+eps);
%         
%         
%         k = k + 1;
%     end
% end
% cnt = zeros(47,47);
% 
% parfor_progress(47*47);
% 
% for i = 1:1880
%     for j = 2:1880
%     
%       
%        
%           
%           cnt(i,j) = abs(ssim(LBP(logsample(double(imresize(rgb2gray(imread(strcat('/home/saraei/Documents/MATLAB/TextureClassification/DTD/images', '/', train_name{i}))), [64 64])), 5, 32, 32, 32, [], 50), 1, 8, 1), LBP(logsample(double(imresize(rgb2gray(imread(strcat('/home/saraei/Documents/MATLAB/TextureClassification/DTD/images', '/', train_name{j}))), [64 64])), 5, 32, 32, 32, [], 50), 1, 8, 1)));
%          
%        
%           %FV_test{n} = FisherPatches(histeq(imresize(rgb2gray(imread(strcat(pwd, '/DTD/images', '/', test_name{n}))), [512 512])), 49,21,11,49, 5);
%           
%       
%     end
%     parfor_progress;
% end
% 
% parfor_progress(0);


%%



% parfor i = 1:size(block, 2)
%    for j = 2:size(block, 2)
%       
%       if(abs(ssim(block{i}, block{j})) > 0.9)
%          
%          cnt(i) = cnt(i) + 1;
%           
%       end
%       
%    end
% end

%% Mean of all Images in train data

parfor_progress(1880);
sumImage = 0.0;
for i = 1:1880 
   
   sumImage = sumImage + LBP(logsample(double(imresize(rgb2gray(imread(strcat('/home/saraei/Documents/MATLAB/TextureClassification/DTD/images', '/', train_name{i}))), [512 512])), 5, 256, 256, 256, [], 300), 2, 8, 1);

   parfor_progress;
   
end

train = sumImage/1880;
parfor_progress(0);


