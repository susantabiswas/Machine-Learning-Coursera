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

%%%%%%%%%%%%%%%% Part 1 
    %add bias input
    a1 = [ones(m,1) X];
    
    %find the input for layer 2
    z2 = a1 * Theta1';
    %compute the activation function for layer 2
    a2 = sigmoid(z2);
    %add the bias input to a2
    a2 = [ ones(size(a2,1),1) a2];

    %find inputs to layer 3
    z3 = a2 * Theta2';
    %find the ouput of layer 3
    a3 = sigmoid(z3);
    
       
   
 %for getting the output acc. to the output units
    %if output label is 5 then y_temp = 0 0 0 0 1 0 0 0 0 0
%    y_temp = zeros(1,num_labels);
    
%    val = 0;
%    for i = 1:m
        %make the ouput unit corresponding to the label as 1
%        y_temp( y(i) ) = 1;
%        val = val + sum( -y_temp .* log(a3(i,:)) - ( 1-y_temp ).*log(1-a3(i,:)) );
%        y_temp( y(i) ) = 0;
%    end
%    J = (1/m).*val;
   


    y_temp = zeros(m,num_labels);
    for i = 1:m
        y_temp(i, y(i) ) = 1;
    end

    J = (1/m) .* sum( sum( -y_temp.*log(a3)  - ( 1 - y_temp).*log( 1 - a3) ) );
   
    %regularization
    reg = 0;
    reg = (lambda/(2.*m)) .* ( sum( sum(Theta1(:, 2:end) .^2)) + sum(sum( Theta2(:, 2:end) .^2)) );
    J = J + reg;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





% -------------------------------------------------------------
%BACKPROPAGATION ALGO

    %here we take column vectors in the implementation
    %do backpropagation for eaech example
    for i = 1:m
        % 1.Step 1 :  Forward propagation
    
        %add bias input and make a column vector of a1
        a1 = [1; X(i,:)'];
        
        %find the input for layer 2
        z2 = Theta1 * a1;
     
        %compute the activation function for layer 2
        a2 = sigmoid(z2);
        %add the bias input to a2
        a2 = [ 1; a2];
    
        %find inputs to layer 3
        z3 = Theta2 * a2;
        %find the ouput of layer 3
        a3 = sigmoid(z3);
   
        % Step 2: find error in the output layer
        % make the output in terms of current class i.e, either belongs to current class
        % else doesn't belong to the current class
        correct_y = zeros(num_labels,1);
        correct_y( y(i) ) = 1;
        delta_3 = a3 - correct_y ;
        
        % Step 3: calculate the error for the hidden layers
        delta_2 = (Theta2' * delta_3) .*[1; sigmoidGradient( z2 )];
        %remove the bias term from delta2
        delta_2 = delta_2(2:end);
        
        % Step 4: accumulate the gradients
        Theta1_grad = Theta1_grad + delta_2 * a1';
        Theta2_grad = Theta2_grad + delta_3 * a2';
    end

    % Step 5: regularize the gradients
    Theta1_grad = (1/m) * Theta1_grad;
    Theta1_grad = Theta1_grad + lambda/m .*[ zeros( size(Theta1,1),1 ) Theta1(:, 2:end)];

    Theta2_grad = (1/m) * Theta2_grad;
    Theta2_grad = Theta2_grad + lambda/m.*[zeros( size(Theta2,1),1 ) Theta2(:, 2:end)];    
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
