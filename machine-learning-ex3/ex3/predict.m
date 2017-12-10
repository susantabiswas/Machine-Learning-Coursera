function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

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
    %input for the hidden layer
    z2 = size(m,25);

    %add the bias input to X
    X = [ones(m,1) X];
    
    z2 = X * Theta1';
    z2= sigmoid(z2);
    
    z2 = [ones(m,1) z2];
   
    z3 = z2 * Theta2';
    z3 = z3';
    %p gives the labels which the nn has identified
    [val p ] = max( z3(:,:));  
    %since indexing for 0 was 10 so make all instances of index 10 as zero in the output
    p(p == 10) = 0;







% =========================================================================


end
