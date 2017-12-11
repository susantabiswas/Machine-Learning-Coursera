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
    % cost function 
    J = (0.5/m) .* sum( (X * theta - y).^2 );
    %for regularization
    reg = (lambda * 0.5)/m .* ( sum( theta(2:end).^2 ) );
    J = J + reg;

    % computing the gradient
    grad = (1/m) * X'*( X*theta - y);
    % for j >= 1
    grad =  (lambda/m) .*theta + grad;
    % for j = 0: since we did  the regularization part for the j = 0 term also so we revert the changes made
    grad(1,1) = grad(1,1) -  (lambda/m) .*theta(1,1);
    




% =========================================================================

grad = grad(:);

end
