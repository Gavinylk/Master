%# read dataset
[dataClass, data] = libsvmread('/home/saraei/Documents/MATLAB/TextureClassification/Code/libsvm-3.21-local/heart_scale');

%# split into train/test datasets
trainData = data(1:150,:);
testData = data(151:270,:);
trainClass = dataClass(1:150,:);
testClass = dataClass(151:270,:);
numTrain = size(trainData,1);
numTest = size(testData,1);

%# radial basis function: exp(-gamma*|u-v|^2)
sigma = 2e-3;
rbfKernel = @(X,Y) exp(-sigma .* pdist2(X,Y,'euclidean').^2);

%# compute kernel matrices between every pairs of (train,train) and
%# (test,train) instances and include sample serial number as first column
K =  [ (1:numTrain)' , rbfKernel(trainData,trainData) ];
KK = [ (1:numTest)'  , rbfKernel(testData,trainData)  ];

%# train and test
model = svmtrain(trainClass, K, '-t 4');
[predClass, acc, decVals] = svmpredict(testClass, KK, model);