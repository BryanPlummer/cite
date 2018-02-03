% Code should be placed in the pl-clc directory to run and assumes
% the dataset has been downloaded as specified in that repo.
net = fullfile('models', 'voc_2007_trainvaltest_2012_trainval', 'vgg16_fast_rcnn_iter_100000.caffemodel');
def = fullfile('models', 'fastrcnn_feat.prototxt');
image_dir = fullfile('datasets', 'Flickr30kEntities', 'Images');
output_dir = '.';

% code assumes train is the first to be processed
splits = {'train', 'test', 'val'};
phraseCoeff = [];
phraseMean = [];
for i = 1:length(splits)
    load(fullfile('data', 'flickr30k', sprintf('%sData.mat',splits{i}),'imData');

    imData.filterNonvisualPhrases();
    if strcmp(splits{i},'train')
        imData.concatenateGTBoxes();
    end

    % getPhraseWords returns a nested image x sentence x phrase
    % cell array
    uniquePhrases = imData.getPhraseWords();
    uniquePhrases = vertcat(uniquePhrases{:});
    maxPhrases = max(cellfun(@length, uniquePhrases));
    uniquePhrases = vertcat(uniquePhrases{:});
    uniquePhrases(cellfun(@isempty, uniquePhrases)) = [];
    uniquePhrases = cellfun(@(f)strrep(f,'+',' '), uniquePhrases, 'UniformOutput',false);
    uniquePhrases = cellfun(@(f)strtrim(removePunctuation(f)), uniquePhrases, 'UniformOutput',false);
    uniquePhrases(cellfun(@isempty, uniquePhrases)) = [];
    uniquePhrases = unique(uniquePhrases);
    uniquePhrases = cellfun(@(f)strrep(f,' ','+'), uniquePhrases, 'UniformOutput',false);
    phraseFeatures = single(getHGLMMFeatures(uniquePhrases))';
    uniquePhrases = [uniquePhrases;{'unk'}];

    % compute PCA parameters on first iteration (train split) 
    if isempty(phraseCoeff)
        phraseMean = mean(phraseFeatures, 1);
        phraseFeatures = bsxfun(@minus, phraseFeatures, phraseMean);
        phraseCoeff = pca(phraseFeatures, 'NumComponents', 6000);
    else
        phrase_features = bsxfun(@minus, phraseFeatures, phraseMean);
    end

    phraseFeatures = phraseFeatures * phraseCoeff;
    phraseFeatures = [phraseFeatures; zeros(1, 6000, 'single')];
    phraseFeatures = phraseFeatures';

    hdf5fn = fullfile(output_dir, sprintf('%s_imfeats.h5', splits{i}));
    hdf5write(hdf5fn, 'max_phrases', maxPhrases, 'phrase_features', phraseFeatures, 'phrases', uniquePhrases);
    clear phraseFeatures

    pairs = cell(imData.nImages, 1);
    for x = 1:imData.nImages
        boxes = imData.getBoxes(x);
        pairs{x} = cell(imData.nSentences(x), 1);
        for y = 1:imData.nSentences(x)
            pairs{x}{y} = cell(imData.nPhrases(x, y), 1);
            for z = 1:imData.nPhrases(x, y)
                box = imData.getPhraseGT(x, y, z);
                phrase = imData.getPhrase(x,y,z).getPhraseString(imData.stopwords);
                if ~isempty(phrase)
                    phrase = strtrim(removePunctuation(strrep(phrase,'+',' ')));
                end

                if isempty(phrase)
                    phrase = 'unk';
                end
                phrase = strrep(phrase,' ','+');

                p_id = sprintf('%i_%i', y, z);
                pairs{x}{y}{z} = {imData.imagefns{x}, phrase, p_id};
                overlaps = getIOU(box, boxes);
                phrase_id = sprintf('%s_%s_%s', imData.imagefns{x}, phrase, p_id);
                overlaps = [overlaps; box'];
                hdf5write(hdf5fn, phrase_id, overlaps, 'WriteMode', 'append');
            end
        end
    end

    pairs = vertcat(pairs{:});
    pairs = vertcat(pairs{:});
    pairs = vertcat(pairs{:});
    hdf5write(hdf5fn, 'pairs', pairs, 'WriteMode', 'append');

    % separate into batches since getFastRCNNFeatures operates on a
    % batch level
    batchSize = 1000;
    nBatches = ceil(imData.nImages/batchSize);
    for batch = 1:batchSize:imData.nImages
        batchEnd = min(batch+batchSize-1,imData.nImages);
        imagefns = imData.imagefns(batch:batchEnd);
        imagedir = imData.imagedir;
        ext = imData.ext;
        stopwords = imData.stopwords;
        batchData = ImageSetData(imData.split,imagefns,imagedir,ext,stopwords);
        batchData.phrase = imData.phrase(batch:batchEnd);
        batchData.relationship = imData.phrase(batch:batchEnd);
        batchData.annotations = imData.annotations(batch:batchEnd);
        batchData.boxes = imData.boxes(batch:batchEnd);
        regionFeatures = getFastRCNNFeatures(batchData, net, def);
        for j = 1:batchData.nImages
            boxes = batchData.getBoxes(j);
            imDims = batchData.imSize(j);
            boxFeatures = boxes;
            boxFeatures(:, 1) = boxFeatures(:,1) / imDims(2);
            boxFeatures(:, 2) = boxFeatures(:,2) / imDims(1);
            boxFeatures(:, 3) = boxFeatures(:,3) / imDims(2);
            boxFeatures(:, 4) = boxFeatures(:,4) / imDims(1);
            boxWidth = boxes(:,3) - boxes(:,1);
            boxHeight = boxes(:,4) - boxes(:,2);
            boxFeatures = [boxFeatures, (boxWidth.*boxHeight)./(imDims(1)*imDims(2))];
            features = [regionFeatures{j}; boxFeatures'; boxes'];
            hdf5write(hdf5fn, batchData.imagefns{j}, features, 'WriteMode', 'append');
        end
    end
end

