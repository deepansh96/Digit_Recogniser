function p = predict(Theta1, Theta2, X)
% Given the trained weights from a NN, predicts the label of the input provided

m = size(X, 1);
num_labels = size(Theta2, 1);

% initialize
p = zeros(size(X, 1), 1);

h1 = sigmoid([ones(m, 1) X] * Theta1'); %feed-forward on layer 1
h2 = sigmoid([ones(m, 1) h1] * Theta2'); %feed-forward on layer 2
[dummy, p] = max(h2, [], 2); %choosing the index of max value
							 % much like argmax in numpy

end
