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


hypo = sigmoid( X*theta);
%for j >0
J =  (1/m) .* ( sum( -y .* log( hypo ) - ((1 - y) .* log(1 - hypo)) ) + (0.5).*lambda.*sum( theta(2:length(theta),1) .^2 ) );

%find the gradient for all the thetas including theta(1) then after that just update the value of 
%theta(1) using the formula for theta 0 
%for j > 0
grad =  (1/m) .*( ( X' * ( hypo - y ) ) + lambda .* theta);
%for j = 0
grad(1,1) = (1/m) .* sum((hypo - y) .* X(:,1));
% =============================================================

end
