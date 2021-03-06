clear all; close all;

rng default;

%% ----- Loading Data -----

%% ----- Original Dataset -----

liverPatient_data = load('LiverPatientData.mat');

liverPatientData = liverPatient_data.liverPatientData;

input = liverPatientData(:, 1:10);
target = liverPatientData(:, 11);

% Test percentage split:
p = 0.7;

[m, n] = size(input);
[r, c] = size(target);

shuffled_idx = randperm(m);

train_x = input(shuffled_idx(1:round(p * m)), :); 
train_y = target(shuffled_idx(1:round(p * r)), :);

test_x = input(shuffled_idx(round(p * m) + 1:end), :);
test_y = target(shuffled_idx(round(p * r) + 1:end), :);

for i = 1:10
    if i == 2
        continue;
    end
    
    train_column_data = train_x(:, i);
    test_column_data = test_x(:, i);
    
    train_x(:, i) = (train_column_data - min(train_column_data)) / (max(train_column_data) - min(train_column_data));
    
    test_x(:, i) = (test_column_data - min(train_column_data)) / (max(train_column_data) - min(train_column_data));
end

%% ----- SMOTE Dataset -----

liverPatient_dataSMOTE_table = readtable('LiverPatientDataSmote.csv');

liverPatientData_SMOTE = liverPatient_dataSMOTE_table{:, :};

input_SMOTE = liverPatientData_SMOTE(:, 1:10);
target_SMOTE = liverPatientData_SMOTE(:, 11);

train_x_SMOTE = input_SMOTE(1:570, :);
train_y_SMOTE = target_SMOTE(1:570, :);

test_x_SMOTE = input_SMOTE(571:744, :);
test_y_SMOTE = target_SMOTE(571:744, :);

for i = 1:10
    if i == 2
        continue;
    end
    
    train_column_data = train_x_SMOTE(:, i);
    test_column_data = test_x_SMOTE(:, i);
    
    train_x_SMOTE(:, i) = (train_column_data - min(train_column_data)) / (max(train_column_data) - min(train_column_data));
    
    test_x_SMOTE(:, i) = (test_column_data - min(train_column_data)) / (max(train_column_data) - min(train_column_data));
end

%% ----- Training SVMs -----

%% ----- Original SVM -----

svm = fitcsvm(train_x, train_y, 'KernelFunction', 'linear',...
    'ClassNames', {'0', '1'});

%% ----- SMOTE SVM -----

svm_SMOTE = fitcsvm(train_x_SMOTE, train_y_SMOTE, 'KernelFunction', 'linear',...
    'ClassNames', {'0', '1'});

%% ----- Bayes Opt SVM -----

svm_BayesOpt = fitcsvm(train_x_SMOTE, train_y_SMOTE, 'KernelFunction','linear',...
    'OptimizeHyperparameters', 'auto',...
    'HyperparameterOptimizationOptions', struct('AcquisitionFunctionName',...
    'expected-improvement'),...
    'ClassNames', {'0', '1'});

%% ----- Cross-Validation Of SVMs -----

% Original

cvSVM = crossval(svm, 'KFold', 10);
cvSVM_Loss = kfoldLoss(cvSVM);

% SMOTE

cvSVM_SMOTE = crossval(svm_SMOTE, 'KFold', 10);
cvSVM_SMOTE_Loss = kfoldLoss(cvSVM_SMOTE);

% Bayes Opt

cvSVM_BayesOpt = crossval(svm_BayesOpt, 'KFold', 10);
cvSVM_BayesOpt_Loss = kfoldLoss(cvSVM_BayesOpt);

%% ----- Saving SVMs and Testing Data -----

save('SVM_Models', 'svm', 'svm_SMOTE', 'svm_BayesOpt');
save('SVM_Testing_Datasets', 'test_x', 'test_y', 'test_x_SMOTE', 'test_y_SMOTE');

%% ----- Resources -----

% https://uk.mathworks.com/help/stats/support-vector-machines-for-binary-classification.html
% https://uk.mathworks.com/help/stats/optimize-an-svm-classifier-fit-using-bayesian-optimization.html
% https://uk.mathworks.com/help/stats/templatekernel.html#:~:text=A%20box%20constraint%20is%20a,lead%20to%20longer%20training%20times.