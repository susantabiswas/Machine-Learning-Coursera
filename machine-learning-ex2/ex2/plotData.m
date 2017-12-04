function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%
    %for storing the +ve examples coordinates
    x_pos = [];
    y_pos = [];
    %for storing the negative examples coordinates
    x_neg = [];
    y_neg = [];


    for i = 1 : size(y,1),
        if y(i,1) == 1,
            x_pos = [x_pos, X(i,1)];
            y_pos = [y_pos, X(i,2)];
        else,
            x_neg = [x_neg, X(i,1)];
            y_neg = [y_neg, X(i,2)];
        end
    end

    plot(x_pos, y_pos, 'k+', 'LineWidth', 2, 'MarkerSize', 7)
    hold on;
    plot(x_neg, y_neg, 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7)
    xlabel('Exam 1 score')
    ylabel('Exam 2 score')
   



% =========================================================================



hold off;

end
