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


% Approach 1 (Straight Forward) (RIGHT)

% function _prep_ones = prep_ones(b)
%     _prep_ones = [ones(size(b, 1), 1) b];
% end

% function _for_z = forward_z(X, Theta)
%     _for_z = [ones(size(X, 1), 1) X] * Theta';
% end

% function _forProp = forProp(X, Theta)
%     _forProp = sigmoid(forward_z(X, Theta));
% end

% a = forProp(X, Theta1);
% h = forProp(a, Theta2);

% for k=1:num_labels
%     J -= mean((y==k).*log(h(:, k)) + (1-(y==k)).*log(1 - h(:, k)));
% end

% J += (lambda/(2*m)) * (sumsq(Theta1(:, 2:end)(:)) + sumsq(Theta2(:, 2:end)(:)));


% % Calculating Delta3 and Delta2

% function _backProp = backProp(Theta, d, z)
%     _backProp = (d*Theta) .* sigmoidGradient(z);
% end

% delta3 = h;
% for k=1:num_labels
%     delta3(:, k) -= (y==k);  % delta3 is m * (num_labels)
% endfor

% % Theta2 is (hidden_layer_size+1) * num_labels
% % delta3 is m * num_labels
% % z is m * hidde_layer_size
% % delta2 is m * (hidden_layer_size+1)
% z = forward_z(X, Theta1);
% z = prep_ones(z);
% % delta2 = backProp(Theta2, delta3, z);

% % delta2 is m * 26
% delta2 = (delta3 * Theta2) .* sigmoidGradient(z);


% Theta1_grad = (1/m) * delta2(:, 2:end)' * prep_ones(X);
% Theta1_grad(:, 2:end) += (lambda/m) * Theta1(:, 2:end);

% Theta2_grad = (1/m) * delta3' * prep_ones(a);
% Theta2_grad(:, 2:end) += (lambda/m) * Theta2(:, 2:end);



% Approach 2 (For every entry in the data set) (NEEDS DEBUGGING)

% for k = 1:size(y, 1)

%     % Forward Propagation
%     a_1 = X(k, :)';
%     z_2 = Theta1 * [1; a_1];
%     a_2 = sigmoid(z_2);
%     z_3 = Theta2 * [1; a_2];
%     a_3 = sigmoid(z_3);

%     % Backward Propagation
%     _y = zeros(num_labels, 1);
%     _y(y(k)) = 1;

%     delta3 = a_3 - _y;
%     delta2 = (Theta2'*delta3)(2:end) .* (sigmoidGradient(z_2));

%     Theta1_grad += a_1 * delta2';
%     Theta2_grad += a_2 * delta3';

% endfor

% Theta1_grad = (1/m) * Theta1_grad;
% Theta1_grad += lambda * Theta1;
% Theta1_grad(:, 1) -= Theta1(:, 1);

% Theta1_grad = (1/m) * Theta2_grad;
% Theta2_grad += lambda * Theta2;
% Theta2_grad(:, 1) -= Theta2(:, 1);



% Approach 3 (Readable)

X = X'; y = y';
s1 = input_layer_size;
s2 = hidden_layer_size;
s3 = num_labels;


function [_z _a] = forward(Theta, aPrev)
	_z = Theta * aPrev;
	_a = [ones(1, size(aPrev, 2)); sigmoid(_z)];
end

a1 = [ones(1, size(X, 2)); X];
[z2, a2] = forward(Theta1, a1);
[z3, a3] = forward(Theta2, a2);

h = a3(2:end, :);  % Discarding the first row

for  k=1:s3
    J -= mean((y==k).*log(h(k, :)) + (1-(y==k)).*log(1 - h(k, :)));
endfor
J += (lambda/(2*m)) * (sumsq(Theta1(:, 2:end)(:)) + ...
            sumsq(Theta2(:, 2:end)(:)));


% Starting Backtracking now
delta3 = h;
for k=1:s3
    delta3(k, :) -= (y==k);
endfor
delta2 = (Theta2'*delta3)(2:end, :) .* sigmoidGradient(z2);

Theta2_grad = (1/m) * delta3 * a2';
Theta2_grad(:, 2:end) += (lambda/m) * Theta2(:, 2:end);

Theta1_grad = (1/m) * delta2 * a1';
Theta1_grad(:, 2:end) += (lambda/m) * Theta1(:, 2:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
