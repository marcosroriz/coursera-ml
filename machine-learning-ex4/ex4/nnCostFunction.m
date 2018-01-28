function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%%%% FEED FOWARD IMPLEMENTATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Add ones to the X data matrix
X = [ones(m, 1) X];

for i = 1:m
  a1 = X(i, :); % training sample
  
  z2 = Theta1 * a1.';
  a2 = sigmoid(z2);
  a2 = [1; a2]; % add a^2_0

  z3 = Theta2 * a2;
  a3 = sigmoid(z3);
  
  % y(i) returns the correct digit class (ex: 5). 
  % However we need a vector to represent that, precisely, one that have a digit
  % 1 in the 5th row and 0 in the remainding ones.
  yi = zeros(num_labels, 1);
  yi(y(i)) = 1;
  
%  % compute the cost of each class imperatively
%  xcost = 0;
%  for k = 1:num_labels
%    % Divide the cost equation in two parts (fp - sp)
%    fp = -yi(k) * log(a3(k));
%    sp = (1 - yi(k)) * log(1 - a3(k));
%    xcost = xcost + fp - sp;
%  end
  
  % compute k cost in a single step (alternative vectorized impl)
  % Divide the cost equation in two parts (fp - sp)
  % xcost vector = fp - sp;
  fp = -yi.' * log(a3);
  sp = (ones(size(yi), 1) - yi).' * log(ones(size(a3), 1) - a3);
  xcost = fp - sp;  
  
  J = J + xcost;
end

J = (1/m) * J;

% Now, compute the regularization cost R to add to the total cost J
% First, we need to remove the bias units from the parameters (Non Bias Thetas)
% Then, we need to square each element.

NonBiasTheta1 = (Theta1(:, 2:end)) .^ 2;
NonBiasTheta2 = (Theta2(:, 2:end)) .^ 2;

% Now, put everything in a single vector
NonBiasTheta = [NonBiasTheta1(:); NonBiasTheta2(:)];

% Add lamba term
R = (lambda / (2*m)) * sum(NonBiasTheta);

% Finally, add the regularization cost to the total cost J
J = J + R;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%% BACK PROPAGATION IMPLEMENTATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i = 1:m
  a1 = X(i, :); % training sample
  
  z2 = Theta1 * a1.';
  a2 = sigmoid(z2);
  a2 = [1; a2]; % add a^2_0

  z3 = Theta2 * a2;
  a3 = sigmoid(z3);
  
  % y(i) returns the correct digit class (ex: 5). 
  % However we need a vector to represent that, precisely, one that have a digit
  % 1 in the 5th row and 0 in the remainding ones.
  yi = zeros(num_labels, 1);
  yi(y(i)) = 1;
  
  delta3 = a3 - yi;
  delta2 = (Theta2.' * delta3);
  delta2 = delta2(2:end) .* sigmoidGradient(z2); % removing bias units
  
  deltaGrad1 = delta2 * a1;
  deltaGrad2 = delta3 * a2.';
  
  Theta1_grad = Theta1_grad + deltaGrad1;
  Theta2_grad = Theta2_grad + deltaGrad2;
end

% Now divide grad by number of samples
Theta1_grad = (1 / m) * Theta1_grad;
Theta2_grad = (1 / m) * Theta2_grad;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%% BACK PROPAGATION REGULARIZATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Regularization requires us to discard Theta0 (bias column) by making it zero

RegTheta1 = Theta1;
RegTheta1(:,1) = 0;

Theta1_grad = Theta1_grad + (lambda / m) * RegTheta1;


RegTheta2 = Theta2;
RegTheta2(:,1) = 0;

Theta2_grad = Theta2_grad + (lambda / m) * RegTheta2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
