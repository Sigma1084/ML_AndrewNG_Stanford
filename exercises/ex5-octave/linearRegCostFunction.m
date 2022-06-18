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

% We need to have size(X, 2) = size(theta, 1)
% Otherwise, it is possible that a column of ones must be added to x
if size(X, 2) + 1 == size(theta, 1)
    X = [ones(size(X, 1), 1) X];
endif

if size(X, 2) != size(theta, 1)
    return;
endif


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

J = (1/(2*m)) * (sumsq(X*theta - y) + lambda * sumsq(theta(2:end, :)));
grad = (1/m) * X' * (X*theta - y);
grad(2:end) += (lambda/m) * theta(2:end);

% =========================================================================

grad = grad(:);

end
