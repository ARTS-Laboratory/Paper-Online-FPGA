%% =========================================================
%  END-TO-END SCRIPT:
%  1) Import all .lvm impacts from folder (skip header lines)
%  2) Skip dataset 17 and 24 at import
%  3) Align each impact to peak of output acceleration
%     - start preSamples before the peak
%     - keep targetLength samples total (e.g., 6500)
%  4) Plot aligned impacts (output acceleration) using index starting at 0
%  5) Train MLP (window=512) on:
%       - Train: first 10 healthy + last 10 damaged
%       - Validate: middle 10 (first 5 healthy, second 5 damaged)
%     with early-stop + resume via checkpoints
%  6) Impact-level confusion matrix on validation impacts
% =========================================================

clear; clc; close all;

%% -------------------------------
% USER SETTINGS
% -------------------------------
dataDir = "C:\Users\trott\Dropbox\School\MLP\dataset-2\data-1";


numHeaderLines = 25;

% Alignment / trimming
preSamples   = 100;     % samples before peak
targetLength = 6500;    % samples after alignment (total kept)

% Windowing for MLP
windowSize = 128;
stride     = 32;
useAbs     = false;     % if true, uses abs(acc_output)

% Early-stop + resume
stopFlagFile = fullfile(pwd, "STOP_TRAINING.txt");
ckptDir      = fullfile(pwd, "checkpoints_mlp_01");
if ~exist(ckptDir, 'dir'); mkdir(ckptDir); end

%% -------------------------------
% 1) LIST + SORT FILES
% -------------------------------
files = dir(fullfile(dataDir, '*.lvm'));
if isempty(files)
    error("No .lvm files found in: %s", dataDir);
end

[~, idxSort] = sort({files.name});
files = files(idxSort);

fprintf('Total files found: %d\n', length(files));

%% -------------------------------
% 2) IMPORT (SKIP 17 & 24)
% -------------------------------
impacts = struct();
counter = 0;

for k = 1:length(files)

    % Skip dataset 17 and 24 (based on sorted file order)
    if k == 17 || k == 24
        fprintf('Skipping Test %02d (import stage)\n', k);
        continue;
    end

    filePath = fullfile(dataDir, files(k).name);

    data = readmatrix(filePath, ...
        'FileType','text', ...
        'NumHeaderLines', numHeaderLines);

    counter = counter + 1;

    impacts(counter).testNumber  = k;        % original test index in directory order
    impacts(counter).time        = data(:,1);
    impacts(counter).acc_input   = data(:,2);
    impacts(counter).acc_output  = data(:,3);
    impacts(counter).imp_freq    = data(:,4);
    impacts(counter).imp_mag     = data(:,5);
    impacts(counter).phase       = data(:,6);
end

fprintf('Import complete. Datasets loaded: %d\n', length(impacts));

%% -------------------------------
% 3) ALIGN TO PEAK + TRIM TO targetLength
% -------------------------------
alignedImpacts = struct();
counter = 0;

for k = 1:length(impacts)

    sigOut = impacts(k).acc_output(:);

    % Peak index based on max magnitude (robust to negative peaks)
    [~, peakIdx] = max(abs(sigOut));

    startIdx = peakIdx - preSamples;
    endIdx   = startIdx + targetLength - 1;

    % Skip if not enough samples to align/trim
    if startIdx < 1 || endIdx > length(sigOut)
        fprintf('Skipping Test %02d (insufficient length after alignment)\n', impacts(k).testNumber);
        continue;
    end

    counter = counter + 1;

    alignedImpacts(counter).testNumber  = impacts(k).testNumber;
    alignedImpacts(counter).acc_input   = impacts(k).acc_input(startIdx:endIdx);
    alignedImpacts(counter).acc_output  = impacts(k).acc_output(startIdx:endIdx);
    alignedImpacts(counter).imp_freq    = impacts(k).imp_freq(startIdx:endIdx);
    alignedImpacts(counter).imp_mag     = impacts(k).imp_mag(startIdx:endIdx);
    alignedImpacts(counter).phase       = impacts(k).phase(startIdx:endIdx);
end

fprintf('Alignment complete. Final aligned datasets: %d\n', length(alignedImpacts));

%% -------------------------------
% 4) PLOT ALIGNED OUTPUT ACCELERATION (INDEX STARTS AT 0)
% -------------------------------
figure;
hold on; grid on;

for k = 1:length(alignedImpacts)
    idx = 0:length(alignedImpacts(k).acc_output)-1;
    plot(idx, alignedImpacts(k).acc_output);
end

xlabel('Sample Index (Aligned, starts at 0)');
ylabel('Output Acceleration');
title(sprintf('Aligned Output Acceleration (%d Samples Each)', targetLength));

legendLabels = arrayfun(@(x) sprintf('Test %02d', x.testNumber), alignedImpacts, 'UniformOutput', false);
legend(legendLabels, 'Location', 'bestoutside');
hold off;

%% -------------------------------
% 5) TRAIN/VALIDATE MLP USING SPECIFIED IMPACT SPLIT
%    Train: first 10 healthy + last 10 damaged
%    Val: middle 10 (first 5 healthy, second 5 damaged)
% -------------------------------

% Sort aligned impacts by testNumber
testNums = [alignedImpacts.testNumber];
[~, sidx] = sort(testNums);
alignedImpacts = alignedImpacts(sidx);

if numel(alignedImpacts) < 30
    error('Need at least 30 processed impacts for this split. Found %d.', numel(alignedImpacts));
end

trainHealthy = alignedImpacts(1:10);
valMiddle    = alignedImpacts(11:20);
trainDamaged = alignedImpacts(end-9:end);

fprintf("\nTrain healthy tests:  %s\n", mat2str([trainHealthy.testNumber]));
fprintf("Val middle tests:     %s\n", mat2str([valMiddle.testNumber]));
fprintf("Train damaged tests:  %s\n\n", mat2str([trainDamaged.testNumber]));

% ---- Build TRAIN windows ----
XTrain = [];
YTrain = categorical();

for i = 1:numel(trainHealthy)
    sig = trainHealthy(i).acc_output(:);
    if useAbs; sig = abs(sig); end

    Xi = makeWindows(sig, windowSize, stride);
    Yi = repmat(categorical("healthy"), size(Xi,1), 1);

    XTrain = [XTrain; Xi];
    YTrain = [YTrain; Yi];
end

for i = 1:numel(trainDamaged)
    sig = trainDamaged(i).acc_output(:);
    if useAbs; sig = abs(sig); end

    Xi = makeWindows(sig, windowSize, stride);
    Yi = repmat(categorical("damaged"), size(Xi,1), 1);

    XTrain = [XTrain; Xi];
    YTrain = [YTrain; Yi];
end

% ---- Build VALIDATION blocks (impact-by-impact) ----
valBlocks = cell(10,1);
valTrue   = categorical(strings(10,1));

for i = 1:10
    sig = valMiddle(i).acc_output(:);
    if useAbs; sig = abs(sig); end

    valBlocks{i} = makeWindows(sig, windowSize, stride);

    if i <= 5
        valTrue(i) = categorical("healthy");
    else
        valTrue(i) = categorical("damaged");
    end
end

% ---- Standardize using TRAIN stats ONLY ----
mu = mean(XTrain, 1);
sg = std(XTrain, 0, 1);
sg(sg == 0) = 1;

XTrainZ = (XTrain - mu) ./ sg;

valBlocksZ = cell(size(valBlocks));
for i = 1:numel(valBlocks)
    valBlocksZ{i} = (valBlocks{i} - mu) ./ sg;
end

fprintf("Train windows: %d (healthy=%d, damaged=%d)\n", ...
    size(XTrainZ,1), sum(YTrain=="healthy"), sum(YTrain=="damaged"));

% For training-progress plot validation, stack all validation windows
XValAll = vertcat(valBlocksZ{:});
YValAll = categorical();
for i = 1:10
    YValAll = [YValAll; repmat(valTrue(i), size(valBlocksZ{i},1), 1)];
end

%% -------------------------------
% Define MLP: windowSize -> 128 -> 64 -> 32 -> 2
% -------------------------------
layers = [
    featureInputLayer(windowSize, "Name","in")

    fullyConnectedLayer(32, "Name","fc1")
    reluLayer("Name","relu1")
    dropoutLayer(0.2, "Name","drop1")

    fullyConnectedLayer(16, "Name","fc2")
    reluLayer("Name","relu2")
    dropoutLayer(0.2, "Name","drop2")

    fullyConnectedLayer(8, "Name","fc3")
    reluLayer("Name","relu3")

    fullyConnectedLayer(2, "Name","fc_out")
    softmaxLayer("Name","sm")
    classificationLayer("Name","cls")
];

%% -------------------------------
% Training options (checkpoint + early-stop + resume)
% -------------------------------
miniBatchSize = 256;

opts = trainingOptions("adam", ...
    "MaxEpochs", 10000, ...
    "MiniBatchSize", miniBatchSize, ...
    "InitialLearnRate", 1e-3, ...
    "Shuffle", "every-epoch", ...
    "ValidationData", {XValAll, YValAll}, ...
    "ValidationFrequency", max(1, floor(size(XTrainZ,1)/miniBatchSize)), ...
    "Verbose", true, ...
    "Plots", "training-progress", ...
    "CheckpointPath", ckptDir, ...
    "OutputFcn", @(info) stopIfFlagExists(info, stopFlagFile) ...
);

%% -------------------------------
% Resume if checkpoint exists, else train fresh
% -------------------------------
ckptFiles = dir(fullfile(ckptDir, "*.mat"));

if ~isempty(ckptFiles)
    [~, newestIdx] = max([ckptFiles.datenum]);
    ckptPath = fullfile(ckptDir, ckptFiles(newestIdx).name);
    S = load(ckptPath);

    if isfield(S, "net")
        fprintf("Resuming from checkpoint: %s\n", ckptFiles(newestIdx).name);
        net0 = S.net;
        net = trainNetwork(XTrainZ, YTrain, net0.Layers, opts);
    else
        warning("Checkpoint found but no 'net' variable. Starting fresh.");
        net = trainNetwork(XTrainZ, YTrain, layers, opts);
    end
else
    fprintf("No checkpoint found. Starting fresh training.\n");
    net = trainNetwork(XTrainZ, YTrain, layers, opts);
end

%% -------------------------------
% 6) IMPACT-LEVEL VALIDATION + CONFUSION MATRIX
% -------------------------------
classNames = categories(YTrain);      % authoritative class order
numClasses = numel(classNames);

valPred  = categorical(strings(10,1), classNames);
valScore = zeros(10, numClasses);

for i = 1:10
    Xi = valBlocksZ{i};

    % scores: [nWindows x numClasses]
    scores = predict(net, Xi);

    meanScore = mean(scores, 1);
    valScore(i,:) = meanScore;

    [~, mx] = max(meanScore);
    valPred(i) = categorical(string(classNames{mx}), classNames);
end

impactAcc = mean(valPred == valTrue);
fprintf("Impact-level validation accuracy (middle 10): %.2f%%\n", 100*impactAcc);

figure;
cm = confusionchart(valTrue, valPred);
cm.Title = "Impact-level Confusion Matrix (Middle 10 Validation Impacts)";
cm.RowSummary = "row-normalized";
cm.ColumnSummary = "column-normalized";

% Optional: show per-impact probabilities
T = table([valMiddle.testNumber].', valTrue, valPred, 'VariableNames', ...
    {'TestNumber','TrueLabel','PredLabel'});
for c = 1:numClasses
    T.("P_" + string(classNames{c})) = valScore(:,c);
end
disp(T);


%% ---------------------------------------------------
% FPGA-FRIENDLY EXPORT (TRANSPOSE WEIGHTS + EXPORT NORM)
% ---------------------------------------------------

exportDir = fullfile(pwd, "fpga_export_128_32_16_8");
if ~exist(exportDir, 'dir'); mkdir(exportDir); end

% Export input normalization (required to match MATLAB)
writematrix(mu(:).', fullfile(exportDir, "input_mean.csv"));  % 1 x windowSize
writematrix(sg(:).', fullfile(exportDir, "input_std.csv"));   % 1 x windowSize

% Export class names (optional)
classNames = categories(YTrain);
writecell(cellstr(classNames), fullfile(exportDir, "class_names.csv"));

layers = net.Layers;

metaLines = {};
metaLines{end+1} = sprintf("windowSize=%d", numel(mu));
metaLines{end+1} = sprintf("numClasses=%d", numel(classNames));
metaLines{end+1} = "Weight export convention: W is [inputs x neurons], y = x*W + b";
metaLines{end+1} = "Activation: ReLU after fc1, fc2, fc3; Softmax after fc_out";

for i = 1:length(layers)

    if isa(layers(i), 'nnet.cnn.layer.FullyConnectedLayer')

        layerName = string(layers(i).Name);

        W_matlab = layers(i).Weights;   % MATLAB: [neurons x inputs]
        b        = layers(i).Bias;      % [neurons x 1]

        % FPGA/HLS-friendly: [inputs x neurons]
        W_fpga = W_matlab.';            % transpose

        % Write
        writematrix(W_fpga, fullfile(exportDir, layerName + "_W.csv"));
        writematrix(b,      fullfile(exportDir, layerName + "_b.csv"));

        metaLines{end+1} = sprintf("%s: W=%dx%d (inputs x neurons), b=%dx1", ...
            layerName, size(W_fpga,1), size(W_fpga,2), numel(b));
        fprintf("Exported %s (W: %dx%d, b: %dx1)\n", layerName, size(W_fpga,1), size(W_fpga,2), numel(b));
    end
end

% Save metadata text file
fid = fopen(fullfile(exportDir, "model_meta.txt"), "w");
for k = 1:numel(metaLines)
    fprintf(fid, "%s\n", metaLines{k});
end
fclose(fid);

fprintf("\nFPGA export complete. Files written to: %s\n", exportDir);

%% ==========================================================
% LOCAL FUNCTIONS
% ==========================================================
function Xw = makeWindows(sig, windowSize, stride)
    sig = sig(:);
    N = numel(sig);
    lastStart = N - windowSize + 1;
    if lastStart < 1
        error("Signal too short for windowSize=%d (len=%d).", windowSize, N);
    end
    starts = 1:stride:lastStart;
    Xw = zeros(numel(starts), windowSize);
    for ii = 1:numel(starts)
        s = starts(ii);
        Xw(ii,:) = sig(s:s+windowSize-1).';
    end
end

function stop = stopIfFlagExists(info, stopFlagFile)
    stop = false;
    if info.State == "iteration"
        if exist(stopFlagFile, "file") == 2
            fprintf("\nSTOP flag detected (%s). Stopping training...\n", stopFlagFile);
            stop = true;
        end
    end
end

