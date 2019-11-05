import numpy as np
import h5py
import os
import pickle
from tqdm import tqdm

def load_word_embeddings(word_embedding_filename, embedding_length):
    with open(word_embedding_filename, 'r') as f:
        tok2idx = {}
        vecs = [np.zeros(embedding_length, np.float32)]
        for i, line in enumerate(f):
            if i % 10000 == 0:
                print('Reading word embedding vector %i' % i)

            line = line.strip()
            if not line:
                continue

            vec = line.split(',')
            if len(vec) != embedding_length + 1:
                continue

            label = vec[0].lower()
            vec = np.array([float(x) for x in vec[1:]], np.float32)
            assert len(vec) == embedding_length
            if label not in tok2idx:
                # 0th index is always padding, so no need to -1
                tok2idx[label] = len(vecs)
                vecs.append(vec)

    vecs = np.vstack(vecs)
    print('reading done, word embedding shape:', vecs.shape)
    return tok2idx, vecs

class DataLoader:
    """Class minibatches from data on disk in HDF5 format"""
    def __init__(self, args, region_dim, split, tok2idx, train_phrases = None):
        """Constructor

        Arguments:
        args -- command line arguments passed into the main function
        region_dim -- dimensions of the region features
        phrase_dim -- dimensions of the phrase features
        plh -- placeholder dictory containing the tensor inputs
        split -- the data split (i.e. 'train', 'test', 'val')
        """
        datafn = os.path.join('data', args.dataset, '%s_features.h5' % split)
        self.data = h5py.File(datafn, 'r')

        phrases = list(self.data['phrases'])
        token_dict =  {}
        max_length = 0
        for index, phrase in enumerate(phrases):
            tokens = [tok2idx[token] for token in phrase.split('+') if token in tok2idx]
            max_length = max(max_length, len(tokens))
            max_length = min(max_length, args.max_tokens)
            # if the phrase is too long, keep the last part
            if len(tokens) > max_length:
                tokens = tokens[-max_length:]
            token_dict[phrase] =  tokens

        # mapping from uniquePhrase to w2v
        self.token_dict = token_dict
        self.max_length = max_length
        self.pairs = self.data['pairs']
        if not args.use_augmented:
            pairs = []
            for im_id, phrase, p_id, is_gt in self.pairs:
                if is_gt:
                    pairs.append((im_id, phrase, p_id, is_gt))

            self.pairs = pairs

        self.n_pairs = len(self.pairs)
        self.phrases = phrases

        self.im2pairs = {}
        self.max_phrases = 0
        for sample_id in range(self.n_pairs):
            im_id = self.pairs[sample_id][0]
            if im_id not in self.im2pairs:
                self.im2pairs[im_id] = []

            self.im2pairs[im_id].append(sample_id)
            self.max_phrases = max(self.max_phrases, len(self.im2pairs[im_id]))

        if args.max_phrases > 0:
            self.max_phrases = min(args.max_phrases, self.max_phrases)

        self.im_ids = self.im2pairs.keys()
        self.n_pairs = len(self.im_ids)
        self.pair_index = range(self.n_pairs)

        self.split = split
        self.is_train = split == 'train'
        self.neg_to_pos_ratio = args.neg_to_pos_ratio
        self.batch_size = args.batch_size
        self.max_boxes = args.max_boxes
        self.num_pos = args.num_pos
        if self.is_train:
            self.success_thresh = args.train_success_thresh
        else:
            self.success_thresh = args.test_success_thresh

        if args.npa and split == 'val':
            valfn = 'data/%s/confusion_table_features.pkl' % args.dataset
            if os.path.isfile(valfn):
                datafile = pickle.load(open(valfn, 'rb'))
                self.im2phrase = datafile['phrase']
                self.im2feat = datafile['im']
                self.max_boxes_gt = datafile['max_gt']
            else:
                pair2pos = []
                for im_id, phrase, p_id, _ in self.pairs:
                    overlaps = np.array(self.data['%s_%s_%s' % (im_id, phrase, p_id)])[:-4]
                    pos = np.where(overlaps >= self.success_thresh)[0]
                    pair2pos.append(pos)

                self.im2phrase = {}
                self.im2feat = {}
                self.max_boxes_gt = 0
                for im_id, pairs in tqdm(self.im2pairs.iteritems(), desc='caching gt feats', total=len(self.im2pairs)):
                    features = np.array(self.data[im_id], np.float32)[:, :region_dim]
                    all_boxes = set()
                    for sample_id in pairs:
                        phrase = self.pairs[sample_id][1]
                        if phrase not in train_phrases:
                            continue

                        all_boxes.update(pair2pos[sample_id])

                    all_boxes = list(all_boxes)
                    if len(all_boxes) < 1:
                        continue

                    self.im2phrase[im_id] = {}
                    boxid2idx = dict(zip(all_boxes, range(len(all_boxes))))
                    for sample_id in pairs:
                        phrase = self.pairs[sample_id][1]
                        if phrase not in train_phrases:
                            continue

                        pos = pair2pos[sample_id]
                        if len(pos) < 1:
                            continue

                        for box_id in pos:
                            box_idx = boxid2idx[box_id]
                            if box_idx not in self.im2phrase[im_id]:
                                self.im2phrase[im_id][box_idx] = set()

                            self.im2phrase[im_id][box_idx].add(phrase)

                    self.im2feat[im_id] = features[all_boxes]
                    self.max_boxes_gt = max(self.max_boxes_gt, len(self.im2feat[im_id]))

                save_data = {'phrase' : self.im2phrase, 'im' : self.im2feat, 'max_gt' : self.max_boxes_gt}
                pickle.dump(save_data, open(valfn, 'wb'))

            self.confusion_batch_size = args.batch_size * int(np.floor(self.max_boxes / float(self.max_boxes_gt)))

        self.region_feature_dim = region_dim
        self.confusion_table = None

    def __len__(self):
        return self.n_pairs

    def set_max_length(self, val):
        self.max_length = val

    def shuffle(self):
        ''' Shuffles the order of the pairs being sampled
        '''
        np.random.shuffle(self.pair_index)

    def num_batches(self):
        return int(np.ceil(float(len(self)) / self.batch_size))

    def num_batches_confusion(self):
        return int(np.ceil(float(len(self.im2feat.keys())) / self.confusion_batch_size))

    def num_batches_phrases(self):
        return int(np.ceil(len(self.phrases) / float(self.max_phrases)))

    def get_phrase_batch(self, batch_id):
        ims = []
        start_pair = batch_id * self.max_phrases
        end_pair = min(start_pair + self.max_phrases, len(self.phrases))
        num_pairs = end_pair - start_pair
        phrase_features = np.zeros((1, self.max_phrases, self.max_length), dtype=np.float32)
        for pair_id in range(num_pairs):
            phrase = self.phrases[start_pair + pair_id]
            tokens = self.token_dict[phrase]
            phrase_features[0, pair_id, :len(tokens)] = tokens
            ims.append(phrase)

        return phrase_features, num_pairs, ims

    def get_phrase_labels(self, all_index):
        labels = np.zeros((len(self.im_ids), len(self.phrases)), dtype=np.float32)
        phrase_counts = np.zeros(len(self.phrases), np.int32)
        phrase2index = dict(zip(self.phrases, range(len(self.phrases))))
        for im_index, im_id in enumerate(self.im_ids):
            sample_ids = self.im2pairs[im_id]
            pairs = [self.pairs[sample] for sample in sample_ids]
            for _, phrase, p_id, _ in pairs:
                phrase_index = phrase2index[phrase]
                predicted = all_index[im_index, phrase_index]
                label = float(self.data['%s_%s_%s' % (im_id, phrase, p_id)][predicted] >= self.success_thresh)
                phrase_counts[phrase_index] += 1
                labels[im_index, phrase_index] = max(labels[im_index, phrase_index], label)

        return labels, phrase_counts

    def num_batches_images(self):
        return int(np.ceil(len(self.im_ids) / float(self.batch_size)))

    def get_image_batch(self, batch_id):
        start_pair = batch_id * self.batch_size
        im_id = self.im_ids[start_pair]
        features = np.array(self.data[im_id], np.float32)
        num_boxes = min(len(features), self.max_boxes)
        features = features[:num_boxes, :self.region_feature_dim].reshape((1, num_boxes, -1))
        return features, num_boxes
        
    def get_batch_confusion(self, batch_id, plh, max_phrases):
        start_pair = batch_id * self.confusion_batch_size
        ims = self.im2feat.keys()
        end_pair = min(len(ims), start_pair + self.confusion_batch_size)
        num_pairs = end_pair - start_pair
        region_features = np.zeros((self.confusion_batch_size, self.max_boxes_gt,
                                    self.region_feature_dim), dtype=np.float32)
        gt_labels = np.zeros((self.confusion_batch_size, max_phrases, self.max_boxes_gt),
                             dtype=np.float32)
        ims = ims[start_pair:end_pair]
        nn = []
        for sample_id, im in enumerate(ims):
            features = self.im2feat[im]
            num_boxes = len(features)
            nn.append(num_boxes)
            region_features[sample_id, :num_boxes, :] = features[:num_boxes, :self.region_feature_dim]

        feed_dict = {plh['phrase'] : [],
                     plh['region'] : region_features,
                     plh['train_phase'] : self.is_train,
                     plh['num_boxes'] : self.max_boxes_gt,
                     plh['num_phrases'] : max_phrases,
                     plh['phrase_denom'] : self.confusion_batch_size * self.max_phrases,
                     plh['labels'] : gt_labels
        }

        return feed_dict, ims, nn

    def get_batch(self, batch_id, plh):
        """Returns a minibatch given a valid id for it

        Arguments:
        batch_id -- number between 0 and self.num_batches()

        Returns:
        feed_dict -- dictionary containing minibatch data
        gt_labels -- indicates positive/negative regions
        num_pairs -- number of pairs without padding
        """
        region_features = np.zeros((self.batch_size, self.max_boxes,
                                    self.region_feature_dim), dtype=np.float32)
        num_pairs = self.batch_size
        start_pair = batch_id * num_pairs
        end_pair = min(start_pair + num_pairs, len(self))
        num_pairs = end_pair - start_pair

        im_ids = [self.im_ids[self.pair_index[start_pair + pair_id]] for pair_id in range(num_pairs)]
        num_phrases = [min(len(self.im2pairs[im_id]), self.max_phrases) for im_id in im_ids]

        max_phrases = max(num_phrases)
        if self.confusion_table is not None:
            max_phrases += int(np.ceil(max_phrases / 4.))

        gt_labels = np.zeros((self.batch_size, max_phrases, self.max_boxes),
                             dtype=np.float32)
        phrase_features = np.zeros((self.batch_size, max_phrases, self.max_length),
                                   dtype=np.int32)

        for pair_id in range(num_pairs):
            im_id = self.im_ids[self.pair_index[start_pair + pair_id]]
            features = np.array(self.data[im_id], np.float32)
            num_boxes = min(len(features), self.max_boxes)
            features = features[:num_boxes, :self.region_feature_dim]
            region_features[pair_id, :num_boxes, :] = features

            np.random.shuffle(self.im2pairs[im_id])
            num_phrase = num_phrases[pair_id]
            for i, sample_id in enumerate(self.im2pairs[im_id][:num_phrase]):
                # paired image
                assert(self.pairs[sample_id][0] == im_id)
                
                # paired phrase
                phrase = self.pairs[sample_id][1]

                # phrase instance identifier
                p_id = self.pairs[sample_id][2]
                
                overlaps = np.array(self.data['%s_%s_%s' % (im_id, phrase, p_id)])[:num_boxes]
                tokens = self.token_dict[phrase]
                phrase_features[pair_id, i, :len(tokens)] = tokens
                gt_labels[pair_id, i, :num_boxes] = overlaps >= self.success_thresh
                if self.is_train:
                    if self.confusion_table is not None and (i + num_phrase) < max_phrases:
                        # adds augmented hard-negative phrases
                        candidates = self.confusion_table[phrase]
                        if len(candidates) > 0:
                            phrase = np.random.choice(candidates)
                            tokens = self.token_dict[phrase]
                            phrase_features[pair_id, i + num_phrase, :len(tokens)] = tokens
                            gt_labels[pair_id, i + num_phrase, np.where(overlaps >= self.success_thresh)[0]] = -1
                    
                    num_pos = int(np.sum(gt_labels[pair_id, :]))
                    num_neg = num_pos * self.neg_to_pos_ratio
                    negs = np.random.permutation(np.where(overlaps < 0.3)[0])
                
                    if len(negs) < num_neg: # if not enough negatives
                        negs = np.random.permutation(np.where(overlaps < 0.4)[0])

                    # logistic loss only counts a region labeled as -1 negative
                    gt_labels[pair_id, i, negs[:num_neg]] = -1

        feed_dict = {plh['phrase'] : phrase_features,
                     plh['region'] : region_features,
                     plh['train_phase'] : self.is_train,
                     plh['num_boxes'] : self.max_boxes,
                     plh['num_phrases'] : max_phrases,
                     plh['phrase_denom'] : np.sum(num_phrases).astype(np.float32) + 1e-6,
                     plh['labels'] : gt_labels
        }

        return feed_dict, gt_labels, num_phrases

