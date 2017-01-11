function [trueL predL] = confusionPlotSVM(trueLabel, predictLabel, numImg, numClass, sizeDataset)

iter_s = 1:numImg:sizeDataset;

   for j = 1:numClass
      for k = 1:numImg
         
         if(k~=1 && k~=40)
            trueL(j,k) = trueLabel(iter_s(j)+k);
            predL(j,k) = predictLabel(iter_s(j)+k);
         else
            trueL(j,k) = trueLabel(iter_s(j));
            predL(j,k) = predictLabel(iter_s(j));
         end
   
      end
   end
end