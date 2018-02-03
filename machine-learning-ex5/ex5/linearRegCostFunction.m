function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% COST %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
HIPO = X * theta;
COST = HIPO - y;

SQCOST = COST .^ 2;
JCOST  = (1 / (2 * m)) * sum(SQCOST);

REGTHETA    = theta;
REGTHETA(1) = 0;
SQREGTHETA  = REGTHETA .^ 2;
REGCOST     = (lambda / (2 * m)) * sum(SQREGTHETA);

J = JCOST + REGCOST;


% GRAD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
HIPO = X * theta;
COST = HIPO - y;

THETAGRAD    = X.' * COST;
SUMTHETAGRAD = (1 / m) .* THETAGRAD;

REGTHETA    = theta;
REGTHETA(1) = 0;
REGGRAD     = (lambda / m) .* REGTHETA;

grad = SUMTHETAGRAD + REGGRAD;

% =========================================================================

grad = grad(:);

end
