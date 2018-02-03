# Conditional Image-Text Embedding Networks

The code currently assumes datasets are divided into three hdf5 files named `<split>_imfeats.h5` where `split` takes on the value train, test, or val.  It assumes it has the following items:

1. phrase_features: #num_phrase X 6000 dimensional matrix of phrase features
2. phrases: array of #num_phrase strings corresponding to the phrase features
3. pairs: 3 x M matrix where each column contains a string representation for the `[image name, phrase, pair identifier]` pairs in the split.
4. Each `<image name>` should return a #num_boxes x feature_dimensional matrix of the visual features.  The features should contain the visual representation as well as the spatial features for the box followed by its coordinates (i.e. the precomputed features we released are 4096 (VGG) + 5 (spatial) + 4 (box coordinates) = 4105 dimensional).
5. Each `<image name>_<phrase>_<pair identifier> should contain a vector containing the intersection over union with the ground truth box followed by the box's coordinates (i.e. for N boxes the vector should be N + 4 dimensional).


The example script uses the [pl-clc](https://github.com/BryanPlummer/pl-clc) repo for parsing and computing features of the Flick30K Entities dataset.  It assumes it uses the built-in MATLAB PCA function, and not the one in the `toolbox` external module.