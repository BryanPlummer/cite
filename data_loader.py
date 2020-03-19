import numpy as np
import h5py
import os
import pickle
from tqdm import tqdm
from collections import Counter

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

            vec = line.split()
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
        if args.datadir == 'data':
            datafn = os.path.join(args.datadir, args.dataset, '%s_features.h5' % split)
        else:
            datafn = os.path.join(args.datadir, '%s_features.h5' % split)

        self.data = h5py.File(datafn, 'r')

        self.phrases = list(self.data['phrases'])
        token_dict =  {}
        max_length = 0
        for index, phrase in enumerate(self.phrases):
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
        self.im2numgt = {}
        self.im2pairs = {}
        self.max_phrases = 0
        for sample_id in range(len(self.pairs)):
            im_id, _, _, is_gt = self.pairs[sample_id]
            if im_id not in self.im2pairs:
                self.im2pairs[im_id] = []
            else:
                prev_sample = self.im2pairs[im_id][-1]
                assert not is_gt or (is_gt and self.pairs[prev_sample][-1])

            if is_gt:
                self.im2numgt[im_id] = len(self.im2pairs[im_id])

            self.im2pairs[im_id].append(sample_id)
            self.max_phrases = max(self.max_phrases, len(self.im2pairs[im_id]))

        self.im_ids = self.im2pairs.keys()
        self.pair_index = range(len(self))

        self.split = split
        self.is_train = split == 'train'
        if args.max_phrases > 0 and self.is_train:
            self.max_phrases = min(args.max_phrases, self.max_phrases)

        self.neg_to_pos_ratio = args.neg_to_pos_ratio
        if self.is_train:
            self.success_thresh = args.train_success_thresh
            if args.ifs:
                phrase_counts = Counter()
                for im, sample_ids in self.im2pairs.iteritems():
                    # count only annotated pairs
                    sample_ids = sample_ids[:self.im2numgt[im]]
                    phrase_counts.update([self.pairs[sample_id][1] for sample_id in sample_ids])

                self.sample_prob = {}
                for im, sample_ids in self.im2pairs.iteritems():
                    aug_samples = sample_ids[self.im2numgt[im]:]
                    counts = [phrase_counts[self.pairs[sample_id][1]] for sample_id in aug_samples]
                    counts = np.array(counts, np.float32)
                    total = np.sum(counts)
                    if total > 0 and len(counts) > 1:
                        self.sample_prob[im] = 1 - counts / total
                    else:
                        self.sample_prob[im] = np.ones(len(counts), np.float32) / len(counts)

                    if np.sum(self.sample_prob[im]) != 1:
                        print(counts)
                        print(self.sample_prob[im])
                        assert(False)

        else:
            self.success_thresh = args.test_success_thresh

        self.region_feature_dim = region_dim
        self.confusion_table = None
        self.args = args
        if args.npa and split == 'val':
            self.set_confusion_data(train_phrases)

    def __len__(self):
        return len(self.im_ids)

    def set_max_length(self, val):
        self.max_length = val

    def shuffle(self):
        ''' Shuffles the order of the pairs being sampled
        '''
        np.random.shuffle(self.pair_index)

    def num_batches(self):
        return int(np.ceil(float(len(self)) / self.args.batch_size))

    def num_batches_phrases(self):
        return int(np.ceil(len(self.phrases) / float(self.max_phrases)))

    def set_confusion_data(self, train_phrases):
        cachefn = os.path.join(self.args.datadir, self.args.dataset)
        if self.args.use_augmented:
            cachefn = os.path.join(cachefn, 'confusion_table_features_augmented.pkl')
        else:
            cachefn = os.path.join(cachefn, 'confusion_table_features.pkl')

        if os.path.isfile(cachefn):
            datafile = pickle.load(open(cachefn, 'rb'))
            self.im2phrase = datafile['phrase']
            self.confusion_region_batches = datafile['im']
            self.max_boxes_gt = datafile['max_gt']
        else:
            pair2pos = []
            for im_id, phrase, p_id, _ in self.pairs:
                overlaps = np.array(self.data['%s_%s_%s' % (im_id, phrase, p_id)])[:-4]
                pos = np.where(overlaps >= self.success_thresh)[0]
                pair2pos.append(pos)

            self.im2phrase = {}
            self.max_boxes_gt = 0
            im2feat = {}
            for im_id, pairs in tqdm(self.im2pairs.iteritems(), desc='caching gt feats', total=len(self.im2pairs)):
                features = np.array(self.data[im_id], np.float32)[:, :self.region_feature_dim]
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

                im2feat[im_id] = features[all_boxes]
                self.max_boxes_gt = max(self.max_boxes_gt, len(im2feat[im_id]))

            self.confusion_region_batches = ([], [], [])
            confusion_batch_size = self.args.batch_size * int(np.floor(self.args.max_boxes / float(self.max_boxes_gt)))
            ims = im2feat.keys()
            n_batches = int(np.ceil(float(len(ims)) / confusion_batch_size))
            for batch_id in range(n_batches):
                start_pair = batch_id * confusion_batch_size
                end_pair = min(len(ims), start_pair + confusion_batch_size)
                num_pairs = end_pair - start_pair
                region_features = np.zeros((confusion_batch_size, self.max_boxes_gt,
                                            self.region_feature_dim), dtype=np.float32)

                batch_ims = ims[start_pair:end_pair]
                batch_num_boxes = []
                for sample_id, im in enumerate(batch_ims):
                    features = im2feat[im]
                    num_boxes = len(features)
                    batch_num_boxes.append(num_boxes)
                    region_features[sample_id, :num_boxes, :] = features

                self.confusion_region_batches[0].append(region_features)
                self.confusion_region_batches[1].append(batch_ims)
                self.confusion_region_batches[2].append(batch_num_boxes)

            save_data = {'phrase' : self.im2phrase, 'im' : self.confusion_region_batches, 'max_gt' : self.max_boxes_gt}
            pickle.dump(save_data, open(cachefn, 'wb'))

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

    def update_confusion_table(self, model, val_loader, plh, sess):
        assert self.is_train
        assert val_loader.split == 'val'
        self.confusion_table = {}

        # lets score every phrase for every image
        region_weights = model[3]
        region_batches, ims, num_boxes_in_batch = val_loader.confusion_region_batches
        n_phrase_iters = self.num_batches_phrases()
        gt_labels = np.zeros((len(region_batches), self.max_phrases, val_loader.max_boxes_gt),
                             dtype=np.float32)
        for batch_id in tqdm(range(n_phrase_iters), desc='updating confusion table', total=n_phrase_iters):
            phrase_features, num_phrases, all_phrases = self.get_phrase_batch(batch_id)
            feed_dict = {plh['phrases'] : phrase_features,
                         plh['train_phase'] : False,
                         plh['boxes_per_image'] : val_loader.max_boxes_gt,
                         plh['phrases_per_image'] : self.max_phrases,
                         plh['phrase_count'] : np.sum(num_phrases).astype(np.float32) + 1e-6,
                         plh['labels'] : gt_labels
            }

            all_scores = []
            for num_boxes, region_features in zip(num_boxes_in_batch, region_batches):
                feed_dict[plh['regions']] = region_features
                scores = sess.run(region_weights, feed_dict = feed_dict)

                # lets set the padded phrases and regions to a value enuring they
                # won't accidently get chosen
                scores[:, num_phrases:, :] = -np.inf
                for i, n in enumerate(num_boxes):
                    scores[i, :, n:] = -np.inf

                all_scores.append(scores)

            all_scores = np.concatenate(all_scores)[:len(ims)]
            for phrase_id, phrase in enumerate(all_phrases):
                scores = all_scores[:, phrase_id, :]
                n_boxes = float(scores.shape[1])
                order = np.argsort(scores.reshape(-1))[::-1]
                predicted_phrases = []
                order_index = 0
                while len(predictied_phrases) < self.args.num_confusion_phrases and order_index < len(order):
                    phrase_index = order[order_index]
                    order_index += 1
                    
                    index = int(np.floor(phrase_index / n_boxes))
                    im = ims[index]
                    box_idx = int(phrase_index - index * n_boxes)
                    phrases = val_loader.im2phrase[im][box_idx]
                    if phrase not in phrases:
                        predicted_phrases += list(phrases)

                self.confusion_table[phrase] = predicted_phrases[:self.args.num_confusion_phrases]

    def get_batch(self, batch_id, plh):
        """Returns a minibatch given a valid id for it

        Arguments:
        batch_id -- number between 0 and self.num_batches()

        Returns:
        feed_dict -- dictionary containing minibatch data
        gt_labels -- indicates positive/negative regions
        num_pairs -- number of pairs without padding
        """
        region_features = np.zeros((self.args.batch_size, self.args.max_boxes,
                                    self.region_feature_dim), dtype=np.float32)
        num_pairs = self.args.batch_size
        start_pair = batch_id * num_pairs
        end_pair = min(start_pair + num_pairs, len(self))
        num_pairs = end_pair - start_pair

        im_ids = [self.im_ids[self.pair_index[start_pair + pair_id]] for pair_id in range(num_pairs)]
        num_phrases = [min(len(self.im2pairs[im_id]), self.max_phrases) for im_id in im_ids]

        max_phrases = max(num_phrases)
        if self.confusion_table is not None:
            max_phrases += int(np.ceil(max_phrases / 4.))

        gt_labels = np.zeros((self.args.batch_size, max_phrases, self.args.max_boxes),
                             dtype=np.float32)
        phrase_features = np.zeros((self.args.batch_size, max_phrases, self.max_length),
                                   dtype=np.int32)

        for pair_id in range(num_pairs):
            im_id = self.im_ids[self.pair_index[start_pair + pair_id]]
            features = np.array(self.data[im_id], np.float32)
            num_boxes = min(len(features), self.args.max_boxes)
            features = features[:num_boxes, :self.region_feature_dim]
            region_features[pair_id, :num_boxes, :] = features

            num_phrase = num_phrases[pair_id]
            sample_ids = self.im2pairs[im_id]
            if num_phrase < len(sample_ids):
                # code should be setup to use all phrases at test time
                assert self.is_train
                sample_ids = np.random.choice(sample_ids, size=num_phrase, replace=False)
            elif self.args.ifs and self.is_train:
                num_gt = min(self.im2numgt[im_id], num_phrase)
                aug_samples = sample_ids[num_gt:]
                sample_ids = sample_ids[:num_gt]
                num_to_sample = min(len(aug_samples), num_phrase - num_gt)
                if num_to_sample > 0:
                    sample_prob = self.sample_prob[im_id]
                    aug_samples = np.random.choice(aug_samples, size=num_to_sample, p=sample_prob, replace=False)
                    sample_ids += list(aug_samples)
            
            for i, sample_id in enumerate(sample_ids):
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

        feed_dict = {plh['phrases'] : phrase_features,
                     plh['regions'] : region_features,
                     plh['train_phase'] : self.is_train,
                     plh['boxes_per_image'] : self.args.max_boxes,
                     plh['phrases_per_image'] : max_phrases,
                     plh['phrase_count'] : np.sum(num_phrases).astype(np.float32) + 1e-6,
                     plh['labels'] : gt_labels
        }

        return feed_dict, gt_labels, num_phrases

