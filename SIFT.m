function [FV] = SIFT(img)

 [~,FV] = vl_phow(img, 'Fast', true) ;

% [eigenvectors, projected_data, eigenvalues] = pcaecon(double(FV), size(FV, 1));
% [foo, feature_idx] = sort(eigenvalues, 'descend');
% FV = projected_data(:, feature_idx(1:128));
% FV = reshape(FV, 1, []);

end