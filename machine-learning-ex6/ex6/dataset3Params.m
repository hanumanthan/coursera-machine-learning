function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
values_to_try = [ 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
vec_size = length(values_to_try);
error = zeros(length(vec_size), length(vec_size));

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


for i = 1: vec_size
	for j = 1: vec_size
		model = svmTrain(X, y, values_to_try(i), @(x1, x2) gaussianKernel(x1, x2, values_to_try(j))); 		
		predictions = svmPredict(model, Xval);
		error(i, j) = mean(double(predictions ~= yval));
	end
end
error
[minval, col] = min(min(error,[],1));
[minval, row] = min(min(error,[],2));


C = values_to_try(row);
sigma = values_to_try(col);


% =========================================================================

end
