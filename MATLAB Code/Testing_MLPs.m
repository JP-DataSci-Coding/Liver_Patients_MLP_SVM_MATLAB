clear all; close all;

rng default;

% ----- Positive and Negative Classification -----

% Since the non-liver patient is the minority, we will focus on predicting 
% them and thus will be the positive classification, i.e. 1. So the
% negative classification shall be the liver patient, i.e. 0.

%% ----- Loading Our Neural Networks -----

original_MLP = load('Original_MLP_Model', 'net', 'tr');
originalNet = original_MLP.net;
original_tr = original_MLP.tr;

SMOTE_MLP = load('SMOTE_MLP_Model', 'net_SMOTE', 'tr_SMOTE');
SMOTE_Net = SMOTE_MLP.net_SMOTE;
SMOTE_tr = SMOTE_MLP.tr_SMOTE;

MISS_MLP = load('MISS_MLP_Model', 'net_MISS', 'tr_MISS');
MISS_Net = MISS_MLP.net_MISS;
MISS_tr = MISS_MLP.tr_MISS;

BayesOpt_MLP = load('BayesOpt_SMOTE_MLP_Model', 'net_SMOTE_bayesOpt', 'tr_SMOTE_bayesOpt');
BayesOpt_Net = BayesOpt_MLP.net_SMOTE_bayesOpt;
BayesOpt_tr = BayesOpt_MLP.tr_SMOTE_bayesOpt;

%% ------ Loading Data -----

% Original

liverPatient_data = load('LiverPatientData.mat');

liverPatientData = liverPatient_data.liverPatientData;

input = liverPatientData(:, 1:10);
target = liverPatientData(:, 11);

x = input';
t = target';

% SMOTE

liverPatient_dataSMOTE_table = readtable('LiverPatientDataSmote.csv');

liverPatientData_SMOTE = liverPatient_dataSMOTE_table{:, :};

input_SMOTE = liverPatientData_SMOTE(:, 1:10);
target_SMOTE = liverPatientData_SMOTE(:, 11);

x_SMOTE = input_SMOTE';
t_SMOTE = target_SMOTE';

% NearMiss

liverPatient_dataMISS_table = readtable('LiverPatientDataMiss.csv');

liverPatientData_MISS = liverPatient_dataMISS_table{:, :};

input_MISS = liverPatientData_MISS(:, 1:10);
target_MISS = liverPatientData_MISS(:, 11);

x_MISS = input_MISS';
t_MISS = target_MISS';

%% ----- Test Our Networks -----

%% ----- Original MLP Results -----

testX = x(:, original_tr.testInd);
testT = t(:, original_tr.testInd);
testY = originalNet(testX);

originalMLP_errors = gsubtract(testT, testY);
originalMLP_performance = perform(originalNet, testT, testY) % Test cross-entropy performance.

% Plotting confusion matrix.
figure;
plotconfusion(testT, testY, 'Original MLP'); % Plot the confusion matrix.

[originalC, original_CM] = confusion(testT, testY); % Confusion matrix.

% Calculating the test accuracy using confusion matrix results.
% You could also just add the diagonal percentages from the plot.
accuracy_Original_MLP = 100 * sum(diag(original_CM))./sum(original_CM(:))

% Calculating the recall.
recall_Original_MLP = 100 * original_CM(2, 2)/(original_CM(2, 2) + original_CM(2, 1))
 
% Calculating the precision.
precision_Original_MLP = 100 * original_CM(2, 2)/(original_CM(2, 2) + original_CM(1, 2))

figure;
plotroc(testT, testY, 'Original MLP');

% view(originalNet); % View the layer sizes of our NN.

%% ----- SMOTE MLP Results -----

testX_SMOTE = x_SMOTE(:, SMOTE_tr.testInd);
testT_SMOTE = t_SMOTE(:, SMOTE_tr.testInd);
testY_SMOTE = SMOTE_Net(testX_SMOTE);

SMOTE_MLP_errors = gsubtract(testT_SMOTE, testY_SMOTE);
SMOTE_MLP_performance = perform(SMOTE_Net, testT_SMOTE, testY_SMOTE) % Test cross-entropy performance.

% Plotting confusion matrix.
figure;
plotconfusion(testT_SMOTE, testY_SMOTE, 'SMOTE MLP'); % Plot the confusion matrix.

[SMOTE_C, SMOTE_CM] = confusion(testT_SMOTE, testY_SMOTE); % Confusion matrix.

% Calculating the test accuracy using confusion matrix results.
% You could also just add the diagonal percentages from the plot.
accuracy_SMOTE_MLP = 100 * sum(diag(SMOTE_CM))./sum(SMOTE_CM(:))

% Calculating the recall.
recall_SMOTE_MLP = 100 * SMOTE_CM(2, 2)/(SMOTE_CM(2, 2) + SMOTE_CM(2, 1))
 
% Calculating the precision.
precision_SMOTE_MLP = 100 * SMOTE_CM(2, 2)/(SMOTE_CM(2, 2) + SMOTE_CM(1, 2))

figure;
plotroc(testT_SMOTE, testY_SMOTE, 'SMOTE MLP');

%% ----- NearMiss MLP Results -----

testX_MISS = x_MISS(:, MISS_tr.testInd);
testT_MISS = t_MISS(:, MISS_tr.testInd);
testY_MISS = MISS_Net(testX_MISS);

MISS_MLP_errors = gsubtract(testT_MISS, testY_MISS);
MISS_MLP_performance = perform(MISS_Net, testT_MISS, testY_MISS) % Test cross-entropy performance.

% Plotting confusion matrix.
figure;
plotconfusion(testT_MISS, testY_MISS, 'MISS MLP'); % Plot the confusion matrix.

[MISS_C, MISS_CM] = confusion(testT_MISS, testY_MISS); % Confusion matrix.

% Calculating the test accuracy using confusion matrix results.
% You could also just add the diagonal percentages from the plot.
accuracy_MISS_MLP = 100 * sum(diag(MISS_CM))./sum(MISS_CM(:))

% Calculating the recall.
recall_MISS_MLP = 100 * MISS_CM(2, 2)/(MISS_CM(2, 2) + MISS_CM(2, 1))
 
% Calculating the precision.
precision_MISS_MLP = 100 * MISS_CM(2, 2)/(MISS_CM(2, 2) + MISS_CM(1, 2))

figure;
plotroc(testT_MISS, testY_MISS, 'MISS MLP');

%% ----- Bayes Opt MLP Results -----

testX_BayesOpt = x_SMOTE(:, BayesOpt_tr.testInd);
testT_BayesOpt = t_SMOTE(:, BayesOpt_tr.testInd);
testY_BayesOpt = BayesOpt_Net(testX_BayesOpt);

BayesOpt_MLP_errors = gsubtract(testT_BayesOpt, testY_BayesOpt);
BayesOpt_MLP_performance = perform(BayesOpt_Net, testT_BayesOpt, testY_BayesOpt) % Test cross-entropy performance.

% Plotting confusion matrix.
figure;
plotconfusion(testT_BayesOpt, testY_BayesOpt, 'Bayes Opt MLP'); % Plot the confusion matrix.

[BayesOpt_C, BayesOpt_CM] = confusion(testT_BayesOpt, testY_BayesOpt); % Confusion matrix.

% Calculating the test accuracy using confusion matrix results.
% You could also just add the diagonal percentages from the plot.
accuracy_BayesOpt_MLP = 100 * sum(diag(BayesOpt_CM))./sum(BayesOpt_CM(:))

% Calculating the recall.
recall_BayesOpt_MLP = 100 * BayesOpt_CM(2, 2)/(BayesOpt_CM(2, 2) + BayesOpt_CM(2, 1))
 
% Calculating the precision.
precision_BayesOpt_MLP = 100 * BayesOpt_CM(2, 2)/(BayesOpt_CM(2, 2) + BayesOpt_CM(1, 2))

figure;
plotroc(testT_BayesOpt, testY_BayesOpt, 'Bayes Opt MLP');
