clear all;
rng('default');

liverPatientData_table = readtable('./Data/Indian Liver Patient Dataset (ILPD).csv');

% Checking for missing values.
summary(liverPatientData_table);

% Unfortunately we have four missing values. Since this is not a lot, we
% will simply remove all rows with missing values at the end.

[row, col] = size(liverPatientData_table);

genderCol = liverPatientData_table.Var2;

for i = 1:row
    gender = genderCol(i, 1);
    isMale = strcmp(gender, 'Male');
    
    if isMale == 1
        liverPatientData_table(i, 2) = {'0'};
    else
        liverPatientData_table(i, 2) = {'1'};
    end
end

% We also need to change the target from 1 and 2, to 0 and 1 for the Neural
% Network Pattern Recognition tool.
% 1 is Liver Patient and 2 is Not Liver Patient.

for i = 1:row
    target_value = table2array(liverPatientData_table(i, 11));
    
    if  target_value == 1 % Is liver patient.
        liverPatientData_table(i, 11) = {0};
    else
        liverPatientData_table(i, 11) = {1}; % Is NOT liver patient.
    end
end

liverPatientData_table.Var2 = str2double(liverPatientData_table.Var2);

liverPatientData_table.Properties.VariableNames = {'Age', 'Gender', 'Total Bilirubin',...
    'Direct Bilirubin', 'Alkaline Phosphotase', 'Alamine Aminotransferase',...
    'Aspartate Aminotransferase', 'Total Protiens', 'Albumin',...
    'Ratio Albumin and Globulin Ratio', 'Liver Patient'};

liverPatientData = liverPatientData_table{:, :};

liverPatientData = rmmissing(liverPatientData);

missing = sum(ismissing(liverPatientData)); % No missing values found.

[r, c] = size(liverPatientData);

% ---------- Class Imbalance Check ----------

figure;
[liverPatientGroupCounts, liverPatientGroup] = groupcounts(liverPatientData(:, 11)); 
bar(liverPatientGroup, liverPatientGroupCounts);
title('Liver Patients Class By Group', 'fontweight', 'bold', 'fontsize', 16);
set(gca, 'XTickLabel', {'Patient ' 'Not Patient'});

% Our histogram above shows us that there is a clear class imbalance with
% 414 patients and 165 non patients.

% ---------- Feature Distribution Checks ----------

featureNames = liverPatientData_table.Properties.VariableNames;

for i = 1:10
    figure;
    histogram(liverPatientData(:, i));
    title(featureNames(:, i), 'fontweight', 'bold', 'fontsize', 16);
end

% ---------- Box Plot Checks ----------

for i = 1:10
    figure;
    boxplot(liverPatientData(:, i));
    title(featureNames(:, i), 'fontweight', 'bold', 'fontsize', 16);
end

save('LiverPatientData.mat', 'liverPatientData');
