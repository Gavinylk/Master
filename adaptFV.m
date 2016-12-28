% Adapt Feature Vectors to the largest one

function [out_train out_test] = adaptFV(FV_train, FV_test)

% find largest array
lArr_train = max(cell2mat(cellfun(@length, FV_train, 'UniformOutput', 0)));
lArr_test = max(cell2mat(cellfun(@length, FV_test, 'UniformOutput', 0)));

maxSize = max(lArr_train, lArr_test);

for i = 1:size(FV_train, 1)

   FV_train{i}(:,maxSize) = 0;
   FV_test{i}(:,maxSize) = 0;
%      FV_train{i}(lArr_train) = 0;
%      FV_test{i}(lArr_test) = 0;

end
out_train = FV_train;
out_test = FV_test;
end