function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       

% Going to do this exercise using iterative programming instead of vectorization
l = size(X, 1);

% First, calculating iteratively the median mu
for i = 1:size(X, 1)
  for j = 1:size(X, 2)
    mu(1, j) = mu(1, j) + X(i,j);
  end
end
% We now, divide the computation by l
mu = mu ./ l;

% Second, Calculating the standard deviation
% In this case, the first step is to calculate the sample deviation upper half
%
% Standard Deviation  = sqrt(Variance(X_i))
% where Variance(X_i) = sum_{j=1}^{j=m}(X_ij - mu_ij)^2 * 1 / m - 1
for i = 1:size(X, 1)
  for j = 1:size(X, 2)
    upper = (X(i, j) - mu(1, j)) ^ 2;
    sigma(1, j) = sigma(1, j) + upper;
  end
end

% We now apply divide sigma by m - 1
sigma = sigma ./ (l - 1);

% Now take the sqrt
sigma = sqrt(sigma);

% Third part, subtract the encountered values from X
% where X_norm = (X - mu) / sigma
for i = 1:size(X, 1)
  for j = 1:size(X, 2)
    X_norm(i, j) = (X_norm(i, j) - mu(1, j)) / sigma(1, j);
  end
end

% Vectorized implementation
% mu = mean(X);
% sigma = std(X);
% X_norm = (X - mu) ./ sigma;

% ============================================================


end
