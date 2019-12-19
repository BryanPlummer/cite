import os
import sys
import argparse
import pickle

from tqdm import tqdm
import numpy as np
import tensorflow as tf

from collections import Counter
from model import CITE
from data_loader import DataLoader, load_word_embeddings

parser = argparse.ArgumentParser(description='Conditional Image-Text Similarity Network')
parser.add_argument('--name', default='Conditional_Image-Text_Similarity_Network', type=str,
                    help='name of experiment')
parser.add_argument('--dataset', default='flickr', type=str,
                    help='name of the dataset to use')
parser.add_argument('--datadir', default='data', type=str,
                    help='directory containing the hdf5 data files')
parser.add_argument('--language_model', default='avg', type=str,
                    help='type of language model to use, types: avg (default), attend, gru')
parser.add_argument('--r_seed', type=int, default=42,
                    help='random seed (default: 42)')
parser.add_argument('--info_iterval', type=int, default=1000,
                    help='number of batches to process before outputing training status')
parser.add_argument('--resume', default='', type=str,
                    help='filename of model to load (default: none)')
parser.add_argument('--cca_parameters', default='', type=str,
                    help='filename of cca parameters to load (default: none)')
parser.add_argument('--test', dest='test', action='store_true', default=False,
                    help='Run model on test set')
parser.add_argument('--batch-size', type=int, default=6,
                    help='input batch size for training (default: 6)')
parser.add_argument('--lr', type=float, default=5e-5, metavar='LR',
                    help='learning rate (default: 5e-5)')
parser.add_argument('--embed_l1', type=float, default=5e-5,
                    help='weight of the L1 regularization term used on the concept weight branch (default: 5e-5)')
parser.add_argument('--max_epoch', type=int, default=0,
                    help='maximum number of epochs, less than 1 indicates no limit (default: 0)')
parser.add_argument('--no_gain_stop', type=int, default=5,
                    help='number of epochs used to perform early stopping based on validation performance (default: 5)')
parser.add_argument('--neg_to_pos_ratio', type=int, default=2,
                    help='ratio of negatives to positives used during training (default: 2)')
parser.add_argument('--minimum_gain', type=float, default=5e-4, metavar='N',
                    help='minimum performance gain for a model to be considered better (default: 5e-4)')
parser.add_argument('--cca_weight_reg', type=float, default=5e-5,
                    help='learning rate (default: 1)')
parser.add_argument('--train_success_thresh', type=float, default=0.6,
                    help='minimum training intersection-over-union threshold for success (default: 0.6)')
parser.add_argument('--test_success_thresh', type=float, default=0.5,
                    help='minimum testing intersection-over-union threshold for success (default: 0.5)')
parser.add_argument('--dim_embed', type=int, default=256,
                    help='how many dimensions in the final embedding (default: 256)')
parser.add_argument('--max_boxes', type=int, default=300,
                    help='maximum number of edge boxes per image (default: 300)')
parser.add_argument('--max_phrases', type=int, default=-1,
                    help='maximum number of phrases per image, values of less than will use all of them (default: -1)')
parser.add_argument('--max_tokens', type=int, default=10,
                    help='maximum number of words allowed in a phrase (default: 10)')
parser.add_argument('--num_embeddings', type=int, default=4,
                    help='number of embeddings to train (default: 4)')
parser.add_argument('--region_norm_axis', type=int, default=1,
                    help='axis=1 treats all regions like a single image (better for localization-only) and for axis=2 L2 norm is done for each region')
parser.add_argument('--spatial', dest='spatial', action='store_true', default=False,
                    help='flag indicating whether to use spatial features')
parser.add_argument('--npa', action='store_true', default=False,
                    help='use hard-negative phrase mining')
parser.add_argument('--use_augmented', dest='use_augmented', action='store_true', default=False,
                    help='flag indicating whether to use augmented positive phrases (default: use gt only)')
parser.add_argument('--ifs', action='store_true', default=False,
                    help='uses inverse frequency sampling when training with augmented phrases')
parser.add_argument('--word_embedding', type=str, default='data/hglmm_6kd.txt',
                    help='full path to space separated language embedding features to load')
parser.add_argument('--embedding_ft', dest='embedding_ft', action='store_true', default=False,
                    help='flag indicating whether to fine-tune the language features')
parser.add_argument('--embed_weight', type=float, default=1e-5,
                    help='L2 regularization weight for fine-tuning language features (default: 1e-5)')

def main():
    global args
    args = parser.parse_args()
    assert args.language_model in ['avg', 'attend', 'gru']
    np.random.seed(args.r_seed)
    tf.set_random_seed(args.r_seed)
    phrase_feature_dim = 6000
    region_feature_dim = 2048
    tok2idx, vecs = load_word_embeddings(args.word_embedding, phrase_feature_dim)

    if args.spatial:
        region_feature_dim += 5

    test_loader, train_loader, val_loader = get_data_loaders(region_feature_dim, tok2idx)
    model_constructor = CITE(args, vecs, test_loader.max_length, region_feature_dim)
    model = model_constructor.setup_model()
    plh = model_constructor.get_placeholders()
    if args.test:
        test(model, test_loader, plh, model_name=args.resume)
        sys.exit()

    save_model_directory =  os.path.join('runs', args.dataset, args.name)
    if not os.path.exists(save_model_directory):
        os.makedirs(save_model_directory)
    # training with Adam
    acc, best_adam = train(model, train_loader, val_loader, plh, args.resume)

    # finetune with SGD after loading the best model trained with Adam
    best_model_filename = os.path.join('runs', args.dataset, args.name, 'model_best')
    acc, best_sgd = train(model, train_loader, val_loader, plh,
                          best_model_filename, False, acc)
    best_epoch = best_adam + best_sgd
    
    # get performance on test set
    test_acc = test(model, test_loader, plh, model_name=best_model_filename)
    print('best model at epoch {}: {:.2f}% (val {:.2f}%)'.format(
        best_epoch, round(test_acc*100, 2), round(acc*100, 2)))

def test(model, test_loader, plh, sess=None, model_name = None):
    if sess is None:
        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, model_name)
        
    region_weights = model[3]
    correct = 0.0
    total = 0.0
    n_iterations = test_loader.num_batches()
    for batch_id in range(n_iterations):
        feed_dict, gt_labels, num_phrases = test_loader.get_batch(batch_id, plh)
        scores = sess.run(region_weights, feed_dict = feed_dict)
        total += np.sum(num_phrases)
        for i, num_pairs in enumerate(num_phrases):
            for pair_index in range(num_pairs):
                best_region_index = np.argmax(scores[i, pair_index, :])
                correct += gt_labels[i, pair_index, best_region_index]
                
    acc = correct/total
    print('\n{} set localization accuracy: {:.2f}% for {:d} instances\n'.format(
        test_loader.split, round(acc*100, 2), int(total)))
    return acc

def update_confusion_table(model, test_loader, train_loader, plh, sess):
    region_weights = model[3]
    correct = 0.0
    n_iterations = test_loader.num_batches_confusion()
    feeds = []
    ims = []
    num_boxes = []
    for batch_id in range(n_iterations):
        feed_dict, ii, jj = test_loader.get_batch_confusion(batch_id, plh, train_loader.max_phrases)
        feeds.append(feed_dict)
        ims += ii
        num_boxes.append(jj)

    train_loader.confusion_table = {}
    n_phrase_iters = train_loader.num_batches_phrases()
    for batch_id in tqdm(range(n_phrase_iters), desc='updating confusion table', total=n_phrase_iters):

        phrase_features, num_phrases, all_phrase = train_loader.get_phrase_batch(batch_id)
        all_scores = []
        for nn, feed_dict in zip(num_boxes, feeds):
            feed_dict[plh['phrase']] = phrase_features
            s = sess.run(region_weights, feed_dict = feed_dict)
            s[:, num_phrases:, :] = -np.inf
            for i, n in enumerate(nn):
                s[i, :, n:] = -np.inf

            all_scores.append(s)

        all_scores = np.concatenate(all_scores)[:len(ims)]
        for phrase_id, phrase in enumerate(all_phrase):
            scores = all_scores[:, phrase_id, :]
            n_boxes = float(scores.shape[1])
            order = np.argsort(scores.reshape(-1))[::-1]
            N = 500
            predicted_phrases = []
            for i in order:
                if N < 1:
                    break

                index = int(np.floor(i / n_boxes))
                im = ims[index]
                box_idx = int(i - index * n_boxes)
                if box_idx not in test_loader.im2phrase[im]:
                    continue

                phrases = test_loader.im2phrase[im][box_idx]
                if phrase not in phrases:
                    N -= 1
                    predicted_phrases += list(phrases)
                
            train_loader.confusion_table[phrase] = predicted_phrases

def process_epoch(model, train_loader, plh, sess, train_step, epoch, suffix):
    train_loader.shuffle()
    
    # extract elements from model tuple
    loss = model[0]
    region_loss = model[1]
    l1_loss = model[2]
    
    n_iterations = train_loader.num_batches()
    for batch_id in range(n_iterations):
        feed_dict, _, _ = train_loader.get_batch(batch_id, plh)
        (_, total, region, concept_l1) = sess.run([train_step, loss,
                                                   region_loss, l1_loss],
                                                  feed_dict = feed_dict)

        if batch_id % args.info_iterval == 0:
            print('loss: {:.5f} (region: {:.5f} concept: {:.5f}) '
                  '[{}/{}] (epoch: {}) {}'.format(total, region, concept_l1,
                                                  (batch_id*args.batch_size),
                                                  len(train_loader), epoch,
                                                  suffix))

def train(model, train_loader, test_loader, plh, model_weights, use_adam = True,
          best_acc = 0.):
    sess = tf.Session()
    if use_adam:
        optim = tf.train.AdamOptimizer(args.lr)
        suffix = ''
    else:
        optim = tf.train.GradientDescentOptimizer(args.lr / 10.)
        suffix = 'ft'
        
    weights_norm = tf.losses.get_regularization_losses()
    weights_norm_sum = tf.add_n(weights_norm)
    loss = model[0]
    train_step = optim.minimize(loss + weights_norm_sum)
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    epoch = 1
    best_epoch = 0
    with sess.as_default():
        init.run()
        if model_weights:
            saver.restore(sess, model_weights)
            if use_adam:
                best_acc = test(model, test_loader, plh, sess)
                if args.npa:
                    update_confusion_table(model, test_loader, train_loader, plh, sess)
            
        # model trains until args.max_epoch is reached or it no longer
        # improves on the validation set
        while (epoch - best_epoch) < args.no_gain_stop and (args.max_epoch < 1 or epoch <= args.max_epoch):
            update_table = args.npa and epoch > 1 and epoch % 3 == 0
            if update_table:
                update_confusion_table(model, test_loader, train_loader, plh, sess)

            process_epoch(model, train_loader, plh, sess, train_step, epoch, suffix)
            saver.save(sess, os.path.join('runs', args.dataset, args.name, 'checkpoint'),
                       global_step = epoch)

            acc = test(model, test_loader, plh, sess)

            # the first time we update the table localization accuracy may drop a lot
            # so let's reset the baseline of what is good
            if update_table and epoch - 3 == 0 and use_adam:
                best_acc = acc

            if acc > best_acc:
                saver.save(sess, os.path.join('runs', args.dataset, args.name, 'model_best'))
                if (acc - args.minimum_gain) > best_acc:
                    best_epoch = epoch

                best_acc = acc

            epoch += 1

    return best_acc, best_epoch

def get_data_loaders(region_feature_dim, tok2idx):
    test_loader = DataLoader(args, region_feature_dim, 'test', tok2idx)

    if args.test:
        return test_loader, None, None

    max_length = test_loader.max_length
    train_loader = DataLoader(args, region_feature_dim, 'train', tok2idx)
    max_length = max(max_length, train_loader.max_length)
    val_loader = DataLoader(args, region_feature_dim, 'val', tok2idx, set(train_loader.phrases))
    max_length = max(max_length, val_loader.max_length)
    test_loader.set_max_length(max_length)
    train_loader.set_max_length(max_length)
    val_loader.set_max_length(max_length)
    return test_loader, train_loader, val_loader

if __name__ == '__main__':
    main()
