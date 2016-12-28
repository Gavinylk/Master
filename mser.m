function F = mser(img)

% f = cell(delta, 1);
% 
% parfor i = 1:delta
%     
%     [~,f{i}] = vl_mser(img,'MinDiversity',0.7,'MaxVariation',0.9,'Delta',i) ;
%     
% end
%F = detectMSERFeatures(img, 'MaxAreaVariation', 3, 'ThresholdDelta', 10);
%[F, ~] = extractFeatures(img, F);
%max_size = max(size(f, 2));
binary = im2bw(img, double(multithresh(img))/255.0);
% Trainset

%F = vl_ertr(f{max_size}) ;
%vl_plotframe(F) ;
%F = F(:)';

end
