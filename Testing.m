%% Check size of FV
for i = 1:size(FV_train, 1)
    
        
        
        if(size(FV_train{i}, 2) ~= 30721) disp(i); end
 
end

%% Recursive CF
img = imread('blanket.png');
img = CrossFeature(img, 49, 1);
    
    

