function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%


% ====== COST ==================================================
HARG = X * theta;      % HIPOTHESIS ARGUMENT
HIPO = sigmoid(HARG);  % LOGISTIC HIPOTHESIS

POSCLASSCOST = -1 .* y .* log(HIPO);
NEGCLASSCOST = (1 .- y) .* log(1 .- HIPO);

J = (1 / m) * (sum(POSCLASSCOST) - sum(NEGCLASSCOST));
% =============================================================

% ====== GRAD ==================================================
HE = HIPO - y;                % HIPOTHESIS ERROR

grad = (1 / m) .* (X.' * HE); % WE WILL TRANSPOSE X SO WE CAN MULTIPLY ALL 
                              % FEATURES AT THE SAME TIME AND SUM UP THE ERROR
                              % OF EACH THETA J.

% =============================================================



% =============================================================

end