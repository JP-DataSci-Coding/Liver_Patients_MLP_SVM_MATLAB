clear all; close all;

rng default;

%% ----- SMOTE -----

liverPatient_dataSMOTE_table = readtable('LiverPatientDataSmote.csv');

liverPatientData_SMOTE = liverPatient_dataSMOTE_table{:, :};

input_SMOTE = liverPatientData_SMOTE(:, 1:10);
target_SMOTE = liverPatientData_SMOTE(:, 11);

x_SMOTE = input_SMOTE';
t_SMOTE = target_SMOTE';

trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.

% Create a Pattern Recognition Network
hiddenLayerSize = 10;
net_SMOTE = patternnet(hiddenLayerSize, trainFcn);

% Setup Division of Data for Training, Validation, Testing using indices.
net_SMOTE.divideFcn = 'divideind';
net_SMOTE.divideParam.trainInd = 1:570;
net_SMOTE.divideParam.valInd = 571:657;
net_SMOTE.divideParam.testInd = 658:744;

% ----- Train the Network -----

% For the sections below, the following documentation was used as a close
% reference:

% https://uk.mathworks.com/help/deeplearning/ug/analyze-neural-network-performance-after-training.html

% ----- Setting the Training Parameters -----

% Commented out but left to show how the epoc and max_fail was changed.
% net_SMOTE.trainParam.epochs = 2000;
% net_SMOTE.trainParam.max_fail = 30; 

% net is the neural network and tr is the training record.
[net_SMOTE, tr_SMOTE] = train(net_SMOTE, x_SMOTE, t_SMOTE);

% Saving our neural net model as mat file.
% Commented out after final export.
save('SMOTE_MLP_Model', 'net_SMOTE', 'tr_SMOTE');

trainParams_SMOTE = net_SMOTE.trainParam;

disp(trainParams_SMOTE); % View training parameters in command window.

figure;
plotperform(tr_SMOTE); % Plot the performance chart for tuning purposes.
subtitle('SMOTE MLP Performance');

figure;
plottrainstate(tr_SMOTE);

% Checking to see if the training data was correctly set to be the SMOTE
% version.
trainX_SMOTE = x_SMOTE(:, tr_SMOTE.trainInd);

%% ----- NearMiss -----

liverPatient_dataMISS_table = readtable('LiverPatientDataMiss.csv');

liverPatientData_MISS = liverPatient_dataMISS_table{:, :};

input_MISS = liverPatientData_MISS(:, 1:10);
target_MISS = liverPatientData_MISS(:, 11);

x_MISS = input_MISS';
t_MISS = target_MISS';

% Create a Pattern Recognition Network
net_MISS = patternnet(hiddenLayerSize, trainFcn);

% Setup Division of Data for Training, Validation, Testing using indices.
net_MISS.divideFcn = 'divideind';
net_MISS.divideParam.trainInd = 1:240;
net_MISS.divideParam.valInd = 241:327;
net_MISS.divideParam.testInd = 328:414;

% ----- Train the Network -----

% ----- Setting the Training Parameters -----

% net is the neural network and tr is the training record.
[net_MISS, tr_MISS] = train(net_MISS, x_MISS, t_MISS);

% Saving our neural net model as mat file.
% Commented out after final export.
save('MISS_MLP_Model', 'net_MISS', 'tr_MISS');

trainParams_MISS = net_MISS.trainParam;

disp(trainParams_MISS); % View training parameters in command window.

figure;
plotperform(tr_MISS); % Plot the performance chart for tuning purposes.
subtitle('NearMiss MLP Performance');

figure;
plottrainstate(tr_MISS);

% Checking to see if the training data was correctly set to be the NearMiss
% version.
trainX_MISS = x_MISS(:, tr_MISS.trainInd);

%% ----- Resources -----

% https://uk.mathworks.com/help/deeplearning/ug/divide-data-for-optimal-neural-network-training.html