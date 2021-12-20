clear all; close all;

rng default;

% ----- Positive and Negative Classification -----

% Since the non-liver patient is the minority, we will focus on predicting 
% them and thus will be the positive classification, i.e. 1. So the
% negative classification shall be the liver patient, i.e. 0.

%% ----- Loading Our SVMs -----

SVM = load('SVM_Models');
SVM_Original = SVM.svm;
SVM_SMOTE = SVM.svm_SMOTE;
SVM_BayesOpt = SVM.svm_BayesOpt;

%% ----- Loading Datasets -----

% Original

liverPatient_data = load('SVM_Testing_Datasets.mat');

test_x = liverPatient_data.test_x;
test_y = liverPatient_data.test_y;

% SMOTE

test_x_SMOTE = liverPatient_data.test_x_SMOTE;
test_y_SMOTE = liverPatient_data.test_y_SMOTE;

%% ----- Testing SVMs -----

% Original

% The scores will have two columns, the second column is for the positive
% class.
[pred_original, scores_original] = predict(SVM_Original, test_x);
pred_original = str2double(pred_original);

results_original = confusionmat(test_y, pred_original, 'Order', [0 1]);

accuracy_Original = 100 * sum(diag(results_original))./sum(results_original(:));

% Calculating the recall.
recall_Original = 100 * results_original(2, 2)/(results_original(2, 2) + results_original(2, 1));
 
% Calculating the precision.
precision_Original = 100 * results_original(2, 2)/(results_original(2, 2) + results_original(1, 2));

figure;
confusionchart(results_original, [0, 1]);

[X_Original, Y_Original, T_Original, AUC_Original] = perfcurve(test_y,...
                                         scores_original(:, 2), 1);
                                     
% SMOTE

[pred_SMOTE, scores_SMOTE] = predict(SVM_SMOTE, test_x_SMOTE);
pred_SMOTE = str2double(pred_SMOTE);

results_SMOTE = confusionmat(test_y_SMOTE, pred_SMOTE, 'Order', [0 1]);

accuracy_SMOTE = 100 * sum(diag(results_SMOTE))./sum(results_SMOTE(:));

% Calculating the recall.
recall_SMOTE = 100 * results_SMOTE(2, 2)/(results_SMOTE(2, 2) + results_SMOTE(2, 1));
 
% Calculating the precision.
precision_SMOTE = 100 * results_SMOTE(2, 2)/(results_SMOTE(2, 2) + results_SMOTE(1, 2));

figure;
confusionchart(results_SMOTE, [0, 1]);

[X_SMOTE, Y_SMOTE, T_SMOTE, AUC_SMOTE] = perfcurve(test_y_SMOTE,...
                                         scores_SMOTE(:, 2), 1);

% Bayes Opt

[pred_BayesOpt, scores_BayesOpt] = predict(SVM_BayesOpt, test_x_SMOTE);
% For some reason we do not need to convert the pred_BayesOpt with
% str2double like the others.

results_BayesOpt = confusionmat(test_y_SMOTE, pred_SMOTE, 'Order', [0 1]);

accuracy_BayesOpt = 100 * sum(diag(results_BayesOpt))./sum(results_BayesOpt(:));

% Calculating the recall.
recall_BayesOpt = 100 * results_BayesOpt(2, 2)/(results_BayesOpt(2, 2) + results_BayesOpt(2, 1));
 
% Calculating the precision.
precision_BayesOpt = 100 * results_BayesOpt(2, 2)/(results_BayesOpt(2, 2) + results_BayesOpt(1, 2));

figure;
confusionchart(results_BayesOpt, [0, 1]);

[X_BayesOpt, Y_BayesOpt, T_BayesOpt, AUC_BayesOpt] = perfcurve(test_y_SMOTE,...
                                         scores_BayesOpt(:, 2), 1);

%% ----- Plot SVM ROC/AUC Curves -----

figure;
plot(X_Original, Y_Original);
hold on;
plot(X_SMOTE, Y_SMOTE);
hold on;
plot(X_BayesOpt, Y_BayesOpt);
legend('Original', 'SMOTE', 'Bayes Opt', 'Location', 'Best');
xlabel('False positive rate'); ylabel('True positive rate');
title("SVM ROC - Original AUC = " + num2str(AUC_Original) + ", SMOTE AUC = " + num2str(AUC_SMOTE) + ", Bayes Opt AUC = " + num2str(AUC_BayesOpt));
hold off;

%% ----- Resources -----

% https://uk.mathworks.com/help/stats/classreg.learning.classif.compactclassificationsvm.predict.html