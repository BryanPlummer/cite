# Conditional Image-Text Embedding Networks

**cite** contains a Tensorflow implementation for our [paper](https://arxiv.org/abs/1711.08389).  If you find this code useful in your research, please consider citing:

    @article{plummerCITE2017,
	Author = {Bryan A. Plummer and Paige Kordas and M. Hadi Kiapour and Shuai Zheng and Robinson Piramuthu and Svetlana Lazebnik},
	Title = {Conditional Image-Text Embedding Networks},
	Journal = {arXiv:1711.08389},
	Year = {2017}
    }

This code was tested on an Ubuntu 16.04 system using Tensorflow 1.2.1.

### Phrase Localization Evaluation Demo
After you download our precomputed features/model you can test it using:

    python main.py --test --spatial --resume runs/cite_spatial_k4/model_best

### Training New Models
Our code contains everything required to train or test models using precomputed features.  You can train a new model on Flickr30K Entites using:

    python main.py --name <name of experiment>

When it completes training it will output the localization accuracy using the best model on the testing and validation sets.  Note that the above does not use the spatial features we used in our paper (needs the `--spatial` flag). You can see a listing and description of many tuneable parameters with:

    python main.py --help

### Precomputed Features

Along with our example data processing script in `data_processing_example` you can download our precomputed (PASCAL) features for the Flickr30K Entities dataset [here](https://drive.google.com/file/d/1m5DQ3kh2rCkPremgM91chQgJYZxnEbZw/view?usp=sharing) (52G).  Unpack the features in a folder named `data` or update the path in the data loader class.

Our best CITE model on Flickr30K Entities using these precomputed features can be found [here](https://drive.google.com/open?id=1rmeIqYTCIduNc2QWUEdXLHFGrlOzz2xO).


Many thanks to [Kevin Shih](https://scholar.google.com/citations?user=4x3DhzAAAAAJ&hl=en) and [Liwei Wang](https://scholar.google.com/citations?user=qnbdnZEAAAAJ&hl=en) for access to their Similarity Network code that was used as the basis for this implementation.