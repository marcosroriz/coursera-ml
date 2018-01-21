function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];


% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% COMPUTE Z2
% NOTE THAT X = 401x1, while Theta1 is 25 x 401, HENCE we need to transpose T1
Z2 = X * Theta1.';

% NOW APPLY ACTIVATION FUNCTION (SIGMOID)
A2 = sigmoid(Z2);

% NOW ADD MISSING ONES FROM LAYER2
A2 = [ones(size(A2,1), 1), A2];


% COMPUTE Z3
Z3 = A2 * Theta2.';

% NOW APPLY ACTIVATION FUNCTION (SIGMOID)
A3 = sigmoid(Z3);

[_, clazz] = max(A3, [], 2);

p = clazz;
% =========================================================================


end
