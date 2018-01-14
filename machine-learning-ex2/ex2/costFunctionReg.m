function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


% ====== COST ==================================================
HARG = X * theta;      % HIPOTHESIS ARGUMENT
HIPO = sigmoid(HARG);  % LOGISTIC HIPOTHESIS

POSCLASSCOST = -1 .* y .* log(HIPO);
NEGCLASSCOST = (1 .- y) .* log(1 .- HIPO);

REGTHETA    = theta;
REGTHETA(1) = 0;
REGSQUARE   = REGTHETA .^ 2;
REGULARIZER = (lambda / (2*m)) .* sum(REGSQUARE);

J = ((1 / m) * (sum(POSCLASSCOST) - sum(NEGCLASSCOST))) + REGULARIZER;
% =============================================================


% ====== GRAD ==================================================
HE = HIPO - y;                 % HIPOTHESIS ERROR
RG = (lambda / m) .* REGTHETA; % REGULARIZER DERIVATIVE VECTOR

% WE WILL TRANSPOSE X SO WE CAN MULTIPLY ALL FEATURES AT THE SAME TIME AND 
% SUM UP THE ERROR OF EACH THETA J AND THE REGULARIZER DERIVATIVE.
grad = ((1 / m) .* (X.' * HE)) + RG;

% =============================================================





% =============================================================

end
