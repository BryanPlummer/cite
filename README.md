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

You can test the ReferIt dataset by setting the dataset flag and adjusting the number of embeddings to match the trained model:

    python main.py --test --spatial --dataset referit --num_embeddings 12 --resume runs/referit_spatial_k12/model_best

### Training New Models
Our code contains everything required to train or test models using precomputed features.  You can train a new model on Flickr30K Entites using:

    python main.py --name <name of experiment>

When it completes training it will output the localization accuracy using the best model on the testing and validation sets.  Note that the above does not use the spatial features we used in our paper (needs the `--spatial` flag). You can see a listing and description of many tuneable parameters with:

    python main.py --help

### Precomputed Features

Along with our example data processing script in `data_processing_example` you can download our precomputed (PASCAL) features for the Flickr30K Entities dataset [here](https://drive.google.com/open?id=10h55xBQnaYAEwODsi8Wy5CEsajAoZuzc) (126G) and ReferIt dataset [here](https://drive.google.com/open?id=1tQNG4iUXiGatnbeaO6HV3por7U5WoruH) (88G).  Unpack the features in a folder named `data` or update the path in the data loader class.

Our best CITE model using these precomputed features can be on Flickr30K Entities can be found [here](https://drive.google.com/open?id=1vsFqVPVd3vtYfhYTcCmS3HvHOajTycbo) and ReferIt dataset [here](https://drive.google.com/open?id=1P9g9C-BjY-DWIptvV80HE-hEbCDMk6jM).

You can download the raw Flickr30K Entities data [here](http://web.engr.illinois.edu/~bplumme2/Flickr30kEntities/) and ReferIt [here](http://tamaraberg.com/referitgame/), but isn't necessary to use our precomputed features.


Many thanks to [Kevin Shih](https://scholar.google.com/citations?user=4x3DhzAAAAAJ&hl=en) and [Liwei Wang](https://scholar.google.com/citations?user=qnbdnZEAAAAJ&hl=en) for access to their [Similarity Network](https://arxiv.org/abs/1704.03470) code that was used as the basis for this implementation.