# Conditional Image-Text Embedding Networks

**cite** contains a Tensorflow implementation for our [paper](https://arxiv.org/abs/1711.08389).  If you find this code useful in your research, please consider citing:

    @inproceedings{plummerCITE2018,
	Author = {Bryan A. Plummer and Paige Kordas and M. Hadi Kiapour and Shuai Zheng and Robinson Piramuthu and Svetlana Lazebnik},
	Title = {Conditional Image-Text Embedding Networks},
	Booktitle  = {ECCV},
	Year = {2018}
    }

This code was tested on an Ubuntu 16.04 system using Tensorflow 1.2.1.

### Phrase Localization Evaluation
You can test a model using:

    python main.py --test --spatial --resume runs/<experiment_name>/model_best

You can test the ReferIt dataset by setting the dataset flag and adjusting the number of embeddings to match the trained model, e.g. to train a model with 12 conditional embeddings you would use:

    python main.py --test --spatial --dataset referit --num_embeddings 12 --resume runs/<experiment_name>/model_best

### Training New Models
Our code contains everything required to train or test models using precomputed features.  You can train a new model on Flickr30K Entites using:

    python main.py --name <name of experiment>

When it completes training it will output the localization accuracy using the best model on the testing and validation sets.  Note that the above does not use the spatial features we used in our paper (needs the `--spatial` flag). You can see a listing and description of many tuneable parameters with:

    python main.py --help

### Precomputed Features

We recommend using the `data/cache_cite_features.sh` script from the [phrase detection repository](https://github.com/BryanPlummer/phrase_detection) to obtain the precomputed features to use with our model.  These will obtain better performance than our original paper as seen in [this paper](https://arxiv.org/pdf/1811.07212.pdf), i.e. about 72/54 localization accuracy on Flickr30K Entities and Referit, respectively.  You can also find an explanation of the format of the dataset in the `data_processing_example`.

You can also find precomputed HGLMM features used in our work [here](http://ai.bu.edu/grovle/).


Many thanks to [Kevin Shih](https://scholar.google.com/citations?user=4x3DhzAAAAAJ&hl=en) and [Liwei Wang](https://scholar.google.com/citations?user=qnbdnZEAAAAJ&hl=en) for providing to their implementation of the [Similarity Network](https://arxiv.org/abs/1704.03470) that was used as the basis for this repo.
