function featureSequence = HelperGetFeatureSequences(segments,fParams,sParams)

% Preallocate feature sequences
featureSequence = zeros(sParams.SequenceLength,sParams.NumFeatures,0);

% Loop over segments
for indexS = 1:numel(segments)
    
    % Isolate segment and set NaNs to zero.
    audioIn = segments{indexS};
    audioIn(isnan(audioIn)) = 0;
    
    % Extract GTCC, delta GTCC, and delta-delta GTCC
    [coeffs,delta,deltaDelta] = gtcc(audioIn,fParams.SampleRate, ...
                                     "WindowLength",numel(fParams.Window), ...
                                     "OverlapLength",fParams.OverlapLength, ...
                                     "NumCoeffs",12, ...
                                     "FilterDomain","Time");
                                 
    f0 = pitch(audioIn,fParams.SampleRate, ...
               "WindowLength",numel(fParams.Window), ...
               "OverlapLength",fParams.OverlapLength);

    hr = harmonicRatio(audioIn,fParams.SampleRate, ...
                       "Window",fParams.Window, ...
                       "OverlapLength",fParams.OverlapLength);
    
    % The following functions accept time-domain or frequency-domain input.
    % Convert to the frequency domain only once to lower the computational
    % cost of individual features.
    [~,frequencyVector,~,S] = spectrogram(audioIn,fParams.Window, ...
        fParams.OverlapLength, numel(fParams.Window), fParams.SampleRate , ...
        "power" , "onesided");
    
    slope             = spectralSlope(S,frequencyVector);
    [skewness,spread] = spectralSkewness(S,frequencyVector);
    flux              = spectralFlux(S,frequencyVector);
    centroid          = spectralCentroid(S,frequencyVector);
    rolloff           = spectralRolloffPoint(S,frequencyVector);
    decrease          = spectralDecrease(S,frequencyVector);
    flatness          = spectralFlatness(S,frequencyVector);
    kurtosis          = spectralKurtosis(S,frequencyVector);
    
    % Concatenate the features
    features = [coeffs, delta, deltaDelta, f0, hr, slope, skewness, flux, ...
                spread, centroid, rolloff, decrease, flatness, kurtosis];
    features(~isfinite(features)) = 0;
    
    % Break features into sequences. Overlap of 50% between consecutive
    % sequences.
    for index = 1: sParams.HopLength : size(features,1)-sParams.SequenceLength + 1
        F = features(index:index+sParams.SequenceLength - 1,:);
        featureSequence(:,:,end+1) = F;         %#ok
    end
end