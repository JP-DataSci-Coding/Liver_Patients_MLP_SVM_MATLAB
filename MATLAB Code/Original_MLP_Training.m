clear all; close all;

rng default;

%% ----- Loading Data -----

liverPatient_data = load('LiverPatientData.mat');

liverPatientData = liverPatient_data.liverPatientData;

input = liverPatientData(:, 1:10);
target = liverPatientData(:, 11);

x = input';
t = target';

% Choose a Training Function.
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.
trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.

% Create a Pattern Recognition Network
hiddenLayerSize = 10;
net = patternnet(hiddenLayerSize, trainFcn);
configuration = net.inputs.processFcns; % Checking that the mapminmax function has been set by default.

% Setup Division of Data for Training, Validation, Testing.
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

%% ----- Train the Network -----

% For the sections below (and also for all other MLPs), the following 
% documentation was used as a close reference:

% https://uk.mathworks.com/help/deeplearning/gs/classify-patterns-with-a-neural-network.html
% https://uk.mathworks.com/help/deeplearning/ug/analyze-neural-network-performance-after-training.html

% ----- Setting the Training Parameters -----

% Setting the training parameters of our MLP.

% net.trainParam.goal = 0; % The error goal.
% net.trainParam.epochs = 100; % The maximum iterations.
% net.trainParam.max_fail = 30; % Maximum failures.

% net is the neural network and tr is the training record.
[net, tr] = train(net, x, t);

% Saving our neural net model as mat file.
% Commented out after final export.
save('Original_MLP_Model', 'net', 'tr');

trainParams = net.trainParam;

disp(trainParams); % View training parameters in command window.

figure;
plotperform(tr); % Plot the performance chart for tuning purposes.
subtitle('Original MLP Performance');

figure;
plottrainstate(tr);

%% ----- Resources -----

% https://uk.mathworks.com/help/deeplearning/ug/train-and-apply-multilayer-neural-networks.html
% https://uk.mathworks.com/help/deeplearning/ug/choose-a-multilayer-neural-network-training-function.html
% https://uk.mathworks.com/help/deeplearning/ug/choose-neural-network-input-output-processing-functions.html