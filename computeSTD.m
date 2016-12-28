FV_train = cell(40, 47);
FV_test = cell(40, 47);

    parfor n = 1:size(train_name, 1) 
        for i = 1:47
            if(train_label(n) == i)
                
                FV_train{n} = double(imresize(rgb2gray(imread(strcat(pwd, '/DTD/images', '/', train_name{n}))), [512 512]));
                FV_test{n} = double(imresize(rgb2gray(imread(strcat(pwd, '/DTD/images', '/', test_name{n}))), [512 512]));
            end
        end
    end


%% Variation betweeen the train and test data & classes
 parfor i = 1:47   
     
   temp_train = mat2gray(std(cat(3, FV_train{:,i}), [], 3));
   temp_test = mat2gray(std(cat(3, FV_test{:,i}), [], 3));
   
   globalSTD_train(i) = std(temp_train(:)); 
   globalSTD_test(i) = std(temp_test(:)); 

 end

 %Difference in Variation of train and test
 variation = double(globalSTD_train)./double(globalSTD_test);
    
 %% Variation betweeen the train and test data & within the classes
 temp_train = zeros(40, 47);
 temp_test = zeros(40, 47);
 
 for i = 1:47
     for j = 1:40
     
   temp_train(j, i) = std(mat2gray(FV_train{j,i}(:)));
   temp_test(j, i) = std(mat2gray(FV_test{j,i}(:)));
   
   
     end
 end