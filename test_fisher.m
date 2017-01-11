% disp('Fisher Encoding of training and test data with precalc GMM');
% k = 1;
% l = 1;
% m = localGMM(1);
% 
% 
% parfor_progress(0);
% iter_s = 1:40:size(test_name, 1);
% iter_e = 40:40:size(test_name, 1);
% 
% parfor_progress(size(test_name, 1));
% 
% for n = 1:size(test_name, 1)
% 
%     
%          FV_train{n} = Fisher(LBP(logsample(double(imresize(rgb2gray(imread(strcat('/home/saraei/Documents/MATLAB/TextureClassification/DTD/images', '/', train_name{n}))), [512 512])),  5, 200, 256, 256, [], 256), 2, 8, 1), FV_train_gmm{m, 1}, FV_train_gmm{m, 2}, FV_train_gmm{m, 3});
%          FV_test{n} = Fisher(LBP(logsample(double(imresize(rgb2gray(imread(strcat('/home/saraei/Documents/MATLAB/TextureClassification/DTD/images', '/', test_name{n}))), [512 512])),  5, 200, 256, 256, [], 256), 2, 8, 1), FV_train_gmm{m, 1}, FV_train_gmm{m, 2}, FV_train_gmm{m, 3});
% 
%          if(n == iter_e(k) && k ~= 47) k = k + 1; l = l + 1; m = localGMM(l); end 
%          
%            
%         
%        parfor_progress;
%     
% end
% parfor_progress(0);


% k = 1;
% iter_s = 1:40:size(test_name, 1);
% iter_e = 40:40:size(test_name, 1);
% new = cell(size(train_name, 1), 1);
% 
% 
% 
% for n = 1:size(test_name, 1)
% 
%     
%          new{n} = FV_train_gmm{iter_s(k), 3};
%          
% 
%          if(n == iter_s(k) && k ~= 47) k = k + 1; end
%         
%        
%     
% end
for i = 1:40 
   for j = 1:47
      val(i,j) = classes{j}{i}(1,1); 
   end
end


