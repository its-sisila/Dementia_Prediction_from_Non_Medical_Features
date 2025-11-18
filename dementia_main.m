%% --- 1. SETUP AND ENVIRONMENT ---
clear;
clc;
close all;

disp("--- 1. Environment Cleared ---");

%% --- 2. LOAD DATA ---
% !! IMPORTANT !!: Change this to your file path
filepath = 'Dementia Prediction Dataset.csv'; 

columns_to_load = { ...
    'VISITYR', 'BIRTHYR', 'EDUC', 'SEX', 'MARISTAT', 'RACE', 'HANDED', 'NACCALZD' ...
};

try
    opts = detectImportOptions(filepath);
    opts.SelectedVariableNames = columns_to_load;
    T = readtable(filepath, opts);
    disp("--- 3. Successfully loaded required columns ---");
catch ME
    disp("ERROR: Could not load data.");
    disp(ME.message);
    return;
end

%% --- 4. FEATURE ENGINEERING & SELECTION ---
disp("--- 4. Engineering 'AGE' feature ---");
T.AGE = T.VISITYR - T.BIRTHYR;

allowed_features = {'AGE', 'EDUC', 'SEX', 'MARISTAT', 'RACE', 'HANDED'};
target_variable = 'NACCALZD';

%% --- 5. DATA CLEANING ---
disp("--- 5. Cleaning Data... ---");

codes_normal = 8;
codes_dementia = 1;

is_target = ismember(T.(target_variable), [codes_normal, codes_dementia]);
T_filtered = T(is_target, :);
T_filtered.target = (T_filtered.(target_variable) == codes_dementia);

disp("Filtered Target value counts (binary):");
groupcounts(T_filtered, 'target')

% Replace -4 with NaN
for i = 1:length(allowed_features)
    feat = allowed_features{i};
    if isnumeric(T_filtered.(feat))
        T_filtered.(feat)(T_filtered.(feat) == -4) = NaN;
    end
end

X_table = T_filtered(:, allowed_features);
y = T_filtered.target;

numeric_features = {'AGE', 'EDUC'};
categorical_features = {'SEX', 'MARISTAT', 'RACE', 'HANDED'};

%% --- 6. PREPROCESSING (FIXED) ---
disp("--- 6. Preprocessing Data (Imputing, Scaling, Encoding)... ---");

% --- 1. Imputation (FIXED) ---
% We must calculate the value first, then fill with 'constant'

% Numerical: Fill with Median
for i = 1:length(numeric_features)
    feat = numeric_features{i};
    data_col = X_table.(feat);
    
    % Calculate median ignoring NaNs
    med_val = median(data_col, 'omitnan');
    
    % Fill missing values
    X_table.(feat) = fillmissing(data_col, 'constant', med_val);
end

% Categorical: Fill with Mode (Most Frequent)
for i = 1:length(categorical_features)
    feat = categorical_features{i};
    data_col = X_table.(feat);
    
    % Calculate mode (most frequent value)
    % We assume these are numeric codes. If categorical, mode still works.
    mode_val = mode(data_col); 
    
    % If mode is NaN (empty col), default to 0
    if isnan(mode_val)
        mode_val = 0;
    end
    
    X_table.(feat) = fillmissing(data_col, 'constant', mode_val);
end

% --- 2. Scaling (Numerical) ---
% Use zscore normalization
for i = 1:length(numeric_features)
    feat = numeric_features{i};
    X_table.(feat) = normalize(X_table.(feat), 'zscore');
end

% --- 3. One-Hot Encoding (Categorical) ---
X_numeric = X_table(:, numeric_features);
X_encoded_cats = [];
all_encoded_names = {};

for i = 1:length(categorical_features)
    feat = categorical_features{i};
    C = categorical(X_table.(feat));
    D = dummyvar(C);
    
    % Create names for new columns
    cats = categories(C);
    % Clean category names (remove special chars if any)
    cats = regexprep(cats, '[^a-zA-Z0-9]', '');
    varNames = strcat(feat, '_', cats);
    
    T_dummy = array2table(D, 'VariableNames', varNames);
    X_encoded_cats = [X_encoded_cats, T_dummy];
    all_encoded_names = [all_encoded_names, varNames'];
end

% Combine
X_processed = [X_numeric, X_encoded_cats];
all_feature_names = [numeric_features, all_encoded_names];

disp("--- Preprocessing Complete. ---");

%% --- 7. TRAIN-TEST SPLIT ---
disp("--- 7. Splitting Data... ---");

if isnumeric(y)
    y = logical(y);
end

rng(42);
cv = cvpartition(y, 'HoldOut', 0.2, 'Stratify', true);
idxTrain = training(cv);
idxTest = test(cv);

X_train = X_processed{idxTrain, :};
X_test = X_processed{idxTest, :};
y_train = y(idxTrain);
y_test = y(idxTest);

disp("Training set size: " + num2str(sum(idxTrain)));

%% --- 8. MODEL BUILDING ---
disp("--- 8. Model Building ---");

disp("Training Logistic Regression...");
lr_model = fitclinear(X_train, y_train, 'Learner', 'logistic', ...
    'Solver', 'sparsa', 'Regularization', 'lasso', 'ClassNames', [0, 1]);

disp("Training Final Random Forest...");
% Parameters from Python result
best_n_est = 200;
best_min_leaf = 2;
t = templateTree('MaxNumSplits', 2^20, 'MinLeafSize', best_min_leaf);

rng(42);
rf_model_final = fitcensemble(X_train, y_train, ...
    'Method', 'Bag', ...
    'NumLearningCycles', best_n_est, ...
    'Learners', t, ...
    'ClassNames', [0, 1]);

disp("--- 8. Model training complete. ---");

%% --- 9. EVALUATION ---
disp("--- Evaluation ---");

% Logistic Regression
y_pred_lr = predict(lr_model, X_test);
C_lr = confusionmat(y_test, y_pred_lr);
[~, f1_lr, acc_lr] = C_to_metrics(C_lr);

% Random Forest
y_pred_rf = predict(rf_model_final, X_test);
C_rf = confusionmat(y_test, y_pred_rf);
[~, f1_rf, acc_rf] = C_to_metrics(C_rf);

% Comparison
Metrics = ["Accuracy"; "F1-Score (Dementia)"];
LR_Scores = [acc_lr; f1_lr];
RF_Scores = [acc_rf; f1_rf];
disp(table(Metrics, LR_Scores, RF_Scores));

%% --- 10. EXPLAINABILITY ---
disp("--- Feature Importance ---");
% Use OOB Permutation Importance
imp = oobPermutedPredictorImportance(rf_model_final);

% Display top features
[sorted_imp, idx] = sort(imp, 'descend');
sorted_names = all_feature_names(idx)';
disp(table(sorted_names, sorted_imp', 'VariableNames', {'Feature', 'Importance'}));

%% --- HELPER FUNCTION ---
function [report, f1_1, accuracy] = C_to_metrics(C)
    TN = C(1,1); FP = C(1,2);
    FN = C(2,1); TP = C(2,2);

    precision_1 = TP / (TP + FP);
    recall_1    = TP / (TP + FN);
    if (precision_1 + recall_1) == 0
        f1_1 = 0;
    else
        f1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1);
    end
    
    accuracy = (TP + TN) / sum(C, 'all');
    report = "F1: " + num2str(f1_1) + ", Acc: " + num2str(accuracy);
end