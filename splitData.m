function [classes] = splitData(gmm, numClasses, sizeData)


iter_s = 1:40:sizeData;
iter_e = 40:40:sizeData;


   % split gmm into classes
   k = 1;
   
   for i = 1:numClasses     
      for j = iter_s(i):iter_e(i)
      
         classes{i}{k} = gmm{j, 3};
         k = k + 1;
      end
      k = 1;
   end
   
   classes = classes';

end