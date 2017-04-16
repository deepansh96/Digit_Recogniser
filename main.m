
% Main script for handwritten digit recognition
% using an ANN trained on MNIST data set


clear 
close all
clc

% Setting up the params 
input_layer_size  = 400;  % 20x20 pixel input images
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10 (mapping number "0" to 10)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load Training Data
fprintf('\nLoading training data\n')
fprintf('\nDisplaying a part of the data\n')


load('train.mat'); %MNIST dataset converted into .mat format
m = size(X, 1);

sel = randperm(size(X, 1)); %selecting random 100 points to display
sel = sel(1:100);

displayData(X(sel, :)); % Displays 2D data(stored in X) in a grid
fprintf('\nPress enter twice or thrice to begin training the Network...it may take a while\n')

pause; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Randomly initialize the two weight matrices
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];



% Train NN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\nTraining Neural Network...(100 iterations) \n')
options = optimset('MaxIter', 100);
lambda = 1;

% feed-forward and backprop implemented in nnCostFunction.m
costFunction = @(p) nnCostFunction(p, ...       % short hand for the cost function to be minimized      
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);


%using fmincg as an advanced optimization algo
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options); %fmincg imported from web

% Obtain trained Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
fprintf('\nPress enter to see what the network is thinking of\n')
pause;




% Shows what the NN is learning in it's layers %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\nDisplaying Hidden Units\n')

displayData(Theta1(:, 2:end)); 
fprintf('\nPress enter to compute accuracy\n')
pause;


% Computing Accuracy on training set %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pred = predict(Theta1, Theta2, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
fprintf('\nPress enter for a live script\n')
pause;


% Script for a live prediction simulation(using MNIST dataset) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rp = randperm(m);
for i = 1:m
    % Display 
    fprintf('\nDisplaying Example Image\n');
    displayData(X(rp(i), :));

    pred = predict(Theta1, Theta2, X(rp(i),:));
    fprintf('\nNN Prediction: %d (digit %d)\n', pred, mod(pred, 10));
    
    % Pause with quit option
    s = input('Paused - press enter to continue, q to exit:','s');
    if s == 'q'
      break
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%