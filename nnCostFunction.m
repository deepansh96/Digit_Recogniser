function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)


% Computes & returns cost and gradient vectors of the NN. 
%the Grad vector is unrolled. Need to reshape.

% Reshaping nn_params back to Theta1 and Theta2, the weight matrices
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% We know this
m = size(X, 1);
         
% Initializing what we need to return
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

%	Feed-forwarding the NN and returning the cost in J %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	X = [ones(m,1) X];
	Yk = zeros(num_labels,m);

%	Making a binary matrix to accomodate all classes(10)	
	for i=1:m
		Yk(y(i),i) = 1; %Yk is 10x5000
	end

	Yk = Yk'; %Yk is 5000x10 now

	z2 = Theta1 * X'; 
	a2 = sigmoid(z2); %activation(2nd layer)
	a2 = [ones(m,1) a2'];   %%%activation(2nd layer) after bias term(a2 = 5000x26)
 
	z3 = Theta2 * a2';
	h = sigmoid(z3'); %activation(3rd layer)-output hypothesis(h is 5000x10) 

	J = (-1/m) * sum(sum(Yk.*log(h) + (-Yk + 1).*log(-h+1))); %Cost function for logistic reg

	%Excluding the first column of both Theta1 and Theta 2 as regularization
	%is not applied to weight of bias units usually
	
	t1 = Theta1(:,2:size(Theta1,2));
	t2 = Theta2(:,2:size(Theta2,2)); 

	reg = (lambda/(2*m)) * (sum(sum(t1.^2)) + sum(sum(t2.^2))); %computing the regularization
																%factor

	J = J+reg; %final regularized cost  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



	%Backprop  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	for i=1:m %going forward and coming back, calculating errors for each traiing eg.

		%forwardProp

		a1 = X(i,:); %activation of 1st layer
		
		z2 = Theta1 * a1';
		a2 = [1; sigmoid(z2)]; %activation of 2nd layer with bias
		
		z3 = Theta2 * a2;
		a3 = sigmoid(z3); %activation of 3rd layer

		z2=[1; z2]; % adding bias term



		%backProp

		actualY=Yk(i,:)';

		delta3 = a3 - actualY; %error due to layer 3
		delta2 = (Theta2'*delta3).*sigmoidGradient(z2); 
		delta2 = delta2(2:end); %error due to layer 2

		Theta1_grad = Theta1_grad + delta2*a1;
		Theta2_grad = Theta2_grad + delta3*a2';

	end  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% partial derivative of COST FX wrt Theta1
Theta1_grad(:, 1) = Theta1_grad(:, 1) ./ m;
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) ./ m + ((lambda/m) * Theta1(:, 2:end));

% partial derivative of COST FX wrt Theta2	
Theta2_grad(:, 1) = Theta2_grad(:, 1) ./ m;
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) ./ m + ((lambda/m) * Theta2(:, 2:end));

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
