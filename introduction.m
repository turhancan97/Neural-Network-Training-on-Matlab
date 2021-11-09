%% Introduction
% Brief Introduction To Neural Network Toolbox for Matlab
% Simple example: how to prepare teaching data, create the neural network, 
% train it and simulate.
clc
clear all
%%
% Create teaching input vector:
P = [0 1 2 3 4 5 6 7 8 9 10];
% teaching output vector (so called target vector):
T = [0 1 2 3 4 3 2 1 2 3 4];
% Values in P and T vectors are pairs: T(n) is expected output value for input P(n).
%%
% Create 'clean' network
net = newff([0 10],[5 1],{'tansig' ,'purelin'});
% See the description of the newff function parameters by typing:
help newff
% Set the number of teaching epochs:
net.trainParam.epochs = 50;
% and train the network:
[net, tr, Y, E] = train(net,P,T);
% See description of the train function parameters by typing:
help train
% Now you can simulate the network (check the response of the neural network):
A = sim(net, P);
% Neural network can be saved to file just like any other variable or object in Matlab.
%%
% Review transition functions available in the toolbox:
n = -5:0.1:5;
plot(n,hardlim(n),'r+:');
% Plot the following functions: purelin, logsig, tansig, satlin.