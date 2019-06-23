%% Examine the Dataset
datafolder = fullfile('E:\SURYA-PRO\male-female detection\en');
ads = audioDatastore(datafolder,'IncludeSubfolders',true,'FileExtensions','','LabelSource','foldernames')
%%
% ads = audioDatastore(fullfile(datafolder,"cv-valid-train"));
metadata = readtable(fullfile(datafolder,"all.csv"));
% metadata = readtable(fullfile(datafolder,"validated.csv"));
% metadata = readtable(fullfile(datafolder,"test.csv"));
% metadata = readtable(fullfile(datafolder,"other.csv"));
% metadata = readtable(fullfile(datafolder,"invalidated.csv"));
% metadata = readtable(fullfile(datafolder,"dev.csv"));
head(metadata)
%%
csvFiles   = metadata.client_id;
[~,csvInd] = sort(csvFiles);

gender     = metadata.gender;
age        = metadata.age;

adsFiles   = ads.Files;
[~,adsInd] = sort(adsFiles);

gender     = gender(csvInd(adsInd));
age        = age(csvInd(adsInd));

%%
ads.Labels = gender;

%%
ads = shuffle(ads);
%%
maleOrfemale = categorical(ads.Labels) == "male" | categorical(ads.Labels) == "female";
isAdult      = categorical(age) ~= "" & categorical(age) ~= "teens";
ads          = subset(ads,maleOrfemale & isAdult);

%%
countEachLabel(ads)

%%
ismale   = find(categorical(ads.Labels) == "male");
isfemale = find(categorical(ads.Labels) == "female");
numFilesPerGender = 3000;
ads = subset(ads,[ismale(1:numFilesPerGender) isfemale(1:numFilesPerGender)]);

%%
ads = shuffle(ads);

%%
countEachLabel(ads);

%% Isolate Speech Segments

%%
[audio,info] = read(ads);
Fs           = info.SampleRate;

%%
timeVector = (1/Fs) * (0:numel(audio)-1);
figure
plot(timeVector,audio)
ylabel("Amplitude")
xlabel("Time (s)")
title("Sample Audio")
grid on

sound(audio,Fs)

%%
audio        = audio ./ max(abs(audio)); % Normalize amplitude
windowLength = 50e-3 * Fs;
segments     = buffer(audio,windowLength);

%%
win = hann(windowLength,'periodic');
signalEnergy = sum(segments.^2,1)/windowLength;
centroid = spectralCentroid(segments,Fs,'Window',win,'OverlapLength',0);

%%
T_E            = mean(signalEnergy)/2;
T_C            = 5000;
isSpeechRegion = (signalEnergy>=T_E) & (centroid<=T_C);

%%
CC = repmat(centroid,windowLength,1);
CC = CC(:);
EE = repmat(signalEnergy,windowLength,1);
EE = EE(:);
flags2 = repmat(isSpeechRegion,windowLength,1);
flags2 = flags2(:);

figure

subplot(3,1,1)
plot(timeVector, CC(1:numel(audio)), ...
     timeVector, repmat(T_C,1,numel(timeVector)), "LineWidth",2)
xlabel("Time (s)")
ylabel("Normalized Centroid")
legend("Centroid","Threshold")
title("Spectral Centroid")
grid on

subplot(3,1,2)
plot(timeVector, EE(1:numel(audio)), ...
     timeVector, repmat(T_E,1,numel(timeVector)),"LineWidth",2)
ylabel("Normalized Energy")
legend("Energy","Threshold")
title("Window Energy")
grid on

subplot(3,1,3)
plot(timeVector, audio, ...
     timeVector,flags2(1:numel(audio)),"LineWidth",2)
ylabel("Audio")
legend("Audio","Speech Region")
title("Audio")
grid on
ylim([-1 1.1])

%%
regionStartPos = find(diff([isSpeechRegion(1)-1, isSpeechRegion]));

RegionLengths  = diff([regionStartPos, numel(isSpeechRegion)+1]);

isSpeechRegion = isSpeechRegion(regionStartPos) == 1;
regionStartPos = regionStartPos(isSpeechRegion);
RegionLengths  = RegionLengths(isSpeechRegion);

startIndices = zeros(1,numel(RegionLengths));
endIndices   = zeros(1,numel(RegionLengths));
for index=1:numel(RegionLengths)
   startIndices(index) = max(1, (regionStartPos(index) - 5) * windowLength + 1); 
   endIndices(index)   = min(numel(audio), (regionStartPos(index) + RegionLengths(index) + 5) * windowLength); 
end

%%
activeSegment       = 1;
isSegmentsActive    = zeros(1,numel(startIndices));
isSegmentsActive(1) = 1;
for index = 2:numel(startIndices)
    if startIndices(index) <= endIndices(activeSegment)
        if endIndices(index) > endIndices(activeSegment)
           endIndices(activeSegment) =  endIndices(index);
        end
    else
        activeSegment = index;
        isSegmentsActive(index) = 1;
    end
end
numSegments = sum(isSegmentsActive);
segments    = cell(1,numSegments);
limits      = zeros(2,numSegments);
speechSegmentsIndices  = find(isSegmentsActive);
for index = 1:length(speechSegmentsIndices)
    segments{index} = audio(startIndices(speechSegmentsIndices(index)): ...
                            endIndices(speechSegmentsIndices(index)));
    limits(:,index) = [startIndices(speechSegmentsIndices(index)) ; ...
                       endIndices(speechSegmentsIndices(index))];
end

%%
figure

plot(timeVector,audio)
hold on
myLegend = cell(1,numel(segments)+1);
myLegend{1} = "Original Audio";
for index = 1:numel(segments)
    plot(timeVector(limits(1,index):limits(2,index)),segments{index});
    myLegend{index+1} = sprintf("Output Audio Segment %d",index);
end
xlabel("Time (s)")
ylabel("Audio")
grid on
legend(myLegend)

%% Audio Features
win = hamming(0.03*Fs,"periodic");
overlapLength = 0.75*numel(win);
featureParams = struct("SampleRate",Fs, ...
                 "Window",win, ...
                 "OverlapLength",overlapLength);

%%
sequenceParams = struct("NumFeatures",50, ...
                 "SequenceLength",40, ...
                 "HopLength",20);
             
%% Extract Features Using Tall Arrays
T = tall(ads)
segments = cellfun(@(x)HelperSegmentSpeech(x,Fs),T,"UniformOutput",false);
FeatureSequences = cellfun(@(x)HelperGetFeatureSequences(x,featureParams,sequenceParams),...
                           segments,"UniformOutput",false);
FeatureSequences = gather(FeatureSequences);
featuresMatrix = cat(3,FeatureSequences{:});

sequencesMeans = zeros(1,sequenceParams.NumFeatures);
sequenceStds   = zeros(1,sequenceParams.NumFeatures);

for index = 1:sequenceParams.NumFeatures
    localFeatures             = featuresMatrix(:,index,:);
    sequencesMeans(index)     = mean(localFeatures(:));
    sequenceStds(index)       = std(localFeatures(:));
    featuresMatrix(:,index,:) = (localFeatures - sequencesMeans(index))/sequenceStds(index);
end

%%
features = cell(1,size(featuresMatrix,3));
for index = 1:size(featuresMatrix,3)
    features{index} = featuresMatrix(:,:,index).';
end

%%
numSequences = cellfun(@(x)size(x,3), FeatureSequences);
mylabels     = ads.Labels;
gender       = cell(sum(numSequences),1);
count        = 1;
for index1 = 1:numel(numSequences)
    for index2 = 1:numSequences(index1)
        gender{count} = mylabels{index1};
        count = count + 1;
    end
end

%% Define the LSTM Network Architecture
layers = [ ...
    sequenceInputLayer(sequenceParams.NumFeatures)
    bilstmLayer(100,"OutputMode","sequence")
    bilstmLayer(100,"OutputMode","last")
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

%%
options = trainingOptions("adam", ...
    "MaxEpochs",10, ...
    "MiniBatchSize",128, ...
    "Plots","training-progress", ...
    "Verbose",false, ...
    "Shuffle","every-epoch", ...
    "LearnRateSchedule","piecewise", ...
    "LearnRateDropFactor",0.1, ...
    "LearnRateDropPeriod",5);

%% Train the LSTM Network
net = trainNetwork(features,categorical(gender),layers,options);

%% Visualize the Training Accuracy
trainPred = classify(net,features);

%%
figure;
cm = confusionchart(categorical(gender),trainPred,'title','Training Accuracy');
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';

%%
numFiles        = numel(numSequences);
actualGender    = categorical(ads.Labels);
predictedGender = actualGender;
counter         = 1;
for index = 1:numFiles
    % Get output classes from sequences corresponding to this file:
    predictions = trainPred(counter: counter + numSequences(index) - 1);
    % Set predicted gender to the most frequently predicted class
    predictedGender(index) = mode(predictions);
    counter = counter + numSequences(index);
end

%%
figure
cm = confusionchart(actualGender,predictedGender,'title','Training Accuracy - Majority Rule');
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';