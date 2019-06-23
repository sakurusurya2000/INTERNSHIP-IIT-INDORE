function segments = HelperSegmentSpeech(audio,Fs)
%HELPERGETSPEECHSEGMENTS Extract speech segments from audio signal.

% Copyright (c) Theodoros Giannakopoulos All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are
% met:
% 
% * Redistributions of source code must retain the above copyright notice,
% this list of conditions and the following disclaimer.
% * Redistributions in binary form must reproduce the above copyright
% notice, this list of conditions and the following disclaimer in the
% documentation and/or other materials provided with the distribution
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
% IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
% THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
% PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
% CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
% EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
% PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
% PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
% LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
% NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
% SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
%
% This function is for internal use only and may change in the future.

%%
% Break the audio into 50-millisecond non-overlapping frames.
audio               = audio ./ max(abs(audio)); % Normalize amplitude
audio(isnan(audio)) = 0;
WindowLength        = 50e-3 * Fs;
segments            = buffer(audio,WindowLength);

%%
% Compute the energy and spectral centroid for each frame.
win          = hamming(WindowLength,'periodic');
signalEnergy = sum(segments.^2,1)/WindowLength;
centroid     = spectralCentroid(segments,Fs,'Window',win,'OverlapLength',0);

E = signalEnergy;
C = centroid;

%%
% Next, set and apply thresholds based on the features.

% Set threshold:
T_E = mean(E)/2;
T_C = 5000; % Hz

% Apply Threshold:
isSpeechRegion = (E>=T_E) & (C<=T_C);
regionStartPos = find(diff([isSpeechRegion(1)-1, isSpeechRegion]));
regionLengths  = diff([regionStartPos, numel(isSpeechRegion)+1]);

% Get speech-only regions.
isSpeechRegion  = isSpeechRegion(regionStartPos) == 1;
regionStartPos  = regionStartPos(isSpeechRegion);
regionLengths   = regionLengths(isSpeechRegion);

% Get start and end indices for each speech region. Extend the region by 5
% windows on each side.
extension = 5;
startIndices = zeros(1,numel(regionLengths));
endIndices   = zeros(1,numel(regionLengths));
for index=1:numel(regionLengths)
   startIndices(index) = max(1, (regionStartPos(index) - extension) * WindowLength + 1); 
   endIndices(index)   = min(numel(audio), (regionStartPos(index) + regionLengths(index) + extension) * WindowLength); 
end

%%
% Finally, merge intersecting speech segments.
activeSegment       = 1;
isSegmentsActive    = zeros(1,numel(startIndices));
isSegmentsActive(1) = 1;
for index = 2:numel(startIndices)
    if startIndices(index) <= endIndices(activeSegment)
        % Current segment intersects with previous segment
        if endIndices(index) > endIndices(activeSegment)
           endIndices(activeSegment) =  endIndices(index);
        end
    else
        % New speech segment detected
        activeSegment = index;
        isSegmentsActive(index) = 1;
    end
end

if ~isempty(startIndices) &&  ~isempty(endIndices)
    numSegments = sum(isSegmentsActive);
    segments    = cell(1,numSegments);
    limits      = zeros(2,numSegments);
    speechSegmentsIndices  = find(isSegmentsActive);
    for index = 1:length(speechSegmentsIndices)
        segments{index} = audio(startIndices(speechSegmentsIndices(index)):endIndices(speechSegmentsIndices(index)));
        limits(:,index) = [startIndices(speechSegmentsIndices(index)) ; endIndices(speechSegmentsIndices(index))];
    end
else
    segments = {audio};
end