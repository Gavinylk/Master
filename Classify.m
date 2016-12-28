function [cv bestc bestg] = Classify(train_label, trainData, numFold, cost, gamma, kernel)

% #######################
% Parameter selection using 3-fold cross validation
% #######################
bestcv = 0;
bestg = 0;
bestc = 0;

if (kernel == 0)
% linear Kernel    
for log2c = cost,
  
    cmd = ['-q -c ', num2str(2^log2c), ' -t 0',' '];
    cv = get_cv_ac(train_label, [(1:size(train_label, 1))' trainData*trainData'], cmd, numFold);
    
    if (cv >= bestcv),
      bestcv = cv; bestc = 2^log2c;
    end
    fprintf('%g %g (best c=%g, rate=%g)\n', log2c, cv, bestc, bestcv);
  
end

end
if(kernel ~= 0)
for log2c = cost,
  for log2g = gamma,
    cmd = ['-q -c ', num2str(2^log2c), ' -g ', num2str(2^log2g),' '];
    cv = get_cv_ac(train_label, double(trainData), cmd, numFold);
    
    if (cv >= bestcv),
      bestcv = cv; bestc = 2^log2c; bestg = 2^log2g;
    end
    fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', log2c, log2g, cv, bestc, bestg, bestcv);
  end
end
end


end