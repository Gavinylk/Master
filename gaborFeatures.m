function GabFeatureVec = gaborFeatures(img,gaborArray)


img = double(img);


%% Filter the image using the Gabor filter bank

% Filter input image by each Gabor filter
[u,v] = size(gaborArray);
gaborResult = cell(u,v);
for i = 1:u
    for j = 1:v
        gaborResult{i,j} = imfilter(img, gaborArray{i,j});
    end
end


%% Create feature vector

% Extract feature vector from input image
GabFeatureVec = [];
Energy = zeros(1, 35);
Mean = zeros(1, 35);

% Calculate Local Energy & Mean Amplitude
for i = 1:35

    Energy(i) = sum(abs(gaborResult{i}(:).^2));
    Mean(i) = sum(abs(gaborResult{i}(:)));
end

% Normalize Feature Vector
GabFeatureVec = horzcat(Energy, Mean);
GabFeatureVec = GabFeatureVec/(sum(GabFeatureVec));

        
    
end
