function W = randInitializeWeights(L_in, L_out)

% Randomly chooses a weight matrix for a layer with L_in
% incoming connections and L_out outgoing connections

W = zeros(L_out, 1 + L_in);


% Note: The first column of W corresponds to the parameters for the bias unit

% Need to initialize every weight randomly- to break symmetry, otherwise
% we'll get the same activations in every iteration


EPSILON = 0.12;
W = rand(L_out,L_in+1)*2*EPSILON - EPSILON; %just a statistics thing

end
