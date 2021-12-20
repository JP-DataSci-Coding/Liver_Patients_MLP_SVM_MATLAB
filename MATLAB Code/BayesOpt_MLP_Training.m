clear all; close all;

rng default;

% ----- SMOTE -----

liverPatient_dataSMOTE_table = readtable('LiverPatientDataSmote.csv');

liverPatientData_SMOTE = liverPatient_dataSMOTE_table{:, :};

input_SMOTE = liverPatientData_SMOTE(:, 1:10);
target_SMOTE = liverPatientData_SMOTE(:, 11);

x_SMOTE = input_SMOTE';
t_SMOTE = target_SMOTE';

%% ----- Bayesian Optimisation -----

% We will optimise the hidden layer size and momentum.
optimVars = [optimizableVariable('LayerSize', [8 19], 'Type','integer')
    optimizableVariable('Momentum', [0.0 1.0])];

% Creating the objective function for the Bayesian optimizer, using the 
% training and validation data as inputs. 
% The objective function returns the classification error on the validation
% set. This function is defined at the end of this script. 
% Because bayesopt uses the error rate on the validation set to choose the 
% best model, it is possible that the final network overfits on the validation set. 
% The final chosen model is then tested on the independent test set to estimate the 
% generalization error.
ObjFcn = makeObjFcn(x_SMOTE, t_SMOTE);
BayesObjectResults = bayesopt(ObjFcn, optimVars,...
    'MaxTime', 14*60*60,...
    'IsObjectiveDeterministic', false,...
    'UseParallel', false);

%% ----- Evaluating the Final Network -----

% Load the best network found in the optimization and its validation accuracy.
bestIdx = BayesObjectResults.IndexOfMinimumTrace(end);
fileName = BayesObjectResults.UserDataTrace{bestIdx};
savedStruct = load(fileName);
net_SMOTE_bayesOpt = savedStruct.net_SMOTE_bayesOpt;
valError = savedStruct.valError;
tr_SMOTE_bayesOpt = savedStruct.tr_SMOTE_bayesOpt;

% Calculating test and validation errors.
predicted_output = net_SMOTE_bayesOpt(x_SMOTE);
testError = perform(net_SMOTE_bayesOpt, t_SMOTE, predicted_output);

% Saving our neural net model as mat file.
% Commented out after final export.
save('BayesOpt_SMOTE_MLP_Model', 'net_SMOTE_bayesOpt', 'tr_SMOTE_bayesOpt');

trainParams_SMOTE_bayesOpt = net_SMOTE_bayesOpt.trainParam;

disp(trainParams_SMOTE_bayesOpt); % View training parameters in command window.

figure;
plotperform(tr_SMOTE_bayesOpt); % Plot the performance chart for tuning purposes.
subtitle('Bayes Opt SMOTE MLP Performance');

figure;
plottrainstate(tr_SMOTE_bayesOpt);

%% ----- Objective Function for Bayesian Optimisation -----

function ObjFcn = makeObjFcn(XTrain, YTrain) % Using training and validation data.
    ObjFcn = @valErrorFun;
      function [valError, cons, fileName] = valErrorFun(optVars) % We are building the
          % the function based on the hyperparameters ranges we specified
          % earlier. "cons" stands for constraints, i.e. our ranges.
          
          trainFcn = 'trainscg'; 
          
          % Optimised hidden layer size.
          layer_size = optVars.LayerSize;
          hiddenLayerSize = (layer_size);
          net_SMOTE_bayesOpt = patternnet(hiddenLayerSize, trainFcn);
         
          % Specifying the data to be used for validation, training and
          % test.
          net_SMOTE_bayesOpt.divideFcn = 'divideind';
          net_SMOTE_bayesOpt.divideParam.trainInd = 1:570;
          net_SMOTE_bayesOpt.divideParam.valInd = 571:657;
          net_SMOTE_bayesOpt.divideParam.testInd = 658:744;
          
          % Training the Network.
          net_SMOTE_bayesOpt.trainParam.showWindow = false;
          net_SMOTE_bayesOpt.trainParam.showCommandLine = false;
          [net_SMOTE_bayesOpt, tr_SMOTE_bayesOpt] = train(net_SMOTE_bayesOpt, XTrain, YTrain);
         
          % Testing the Network.
          predicted_output = net_SMOTE_bayesOpt(XTrain);
          valError = perform(net_SMOTE_bayesOpt, YTrain, predicted_output);
          fileName = num2str(valError) + ".mat";
          save(fileName, 'net_SMOTE_bayesOpt', 'valError', 'tr_SMOTE_bayesOpt')
          cons = [];
      end
end
  
%% ----- Resources -----

% https://uk.mathworks.com/help/deeplearning/ug/deep-learning-using-bayesian-optimization.html
% https://uk.mathworks.com/help/stats/optimizablevariable.html
% https://uk.mathworks.com/help/stats/bayesopt.html