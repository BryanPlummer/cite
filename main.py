import os
import sys
import argparse

import numpy as np
import tensorflow as tf

from model import setup_model
from data_loader import DataLoader

parser = argparse.ArgumentParser(description='Conditional Image-Text Similarity Network')
parser.add_argument('--name', default='Conditional_Image-Text_Similarity_Network', type=str,
                    help='name of experiment')
parser.add_argument('--dataset', default='flickr', type=str,
                    help='name of the dataset to use')
parser.add_argument('--r_seed', type=int, default=42,
                    help='random seed (default: 42)')
parser.add_argument('--info_iterval', type=int, default=250,
                    help='number of batches to process before outputing training status')
parser.add_argument('--resume', default='', type=str,
                    help='filename of model to load (default: none)')
parser.add_argument('--test', dest='test', action='store_true', default=False,
                    help='Run model on test set')
parser.add_argument('--batch-size', type=int, default=200,
                    help='input batch size for training (default: 200)')
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
parser.add_argument('--train_success_thresh', type=float, default=0.6,
                    help='minimum training intersection-over-union threshold for success (default: 0.6)')
parser.add_argument('--test_success_thresh', type=float, default=0.5,
                    help='minimum testing intersection-over-union threshold for success (default: 0.5)')
parser.add_argument('--dim_embed', type=int, default=256,
                    help='how many dimensions in final embedding (default: 256)')
parser.add_argument('--max_boxes', type=int, default=200,
                    help='maximum number of edge boxes per image (default: 200)')
parser.add_argument('--num_embeddings', type=int, default=4,
                    help='number of embeddings to train (default: 4)')
parser.add_argument('--spatial', dest='spatial', action='store_true', default=False,
                    help='Flag indicating whether to use spatial features')

def main():
    global args
    args = parser.parse_args()
    np.random.seed(args.r_seed)
    tf.set_random_seed(args.r_seed)
    phrase_feature_dim = 6000
    region_feature_dim = 4096
    if args.spatial:
        if args.dataset == 'flickr':
            region_feature_dim += 5
        else:
            region_feature_dim += 8
            
    # setup placeholders
    labels_plh = tf.placeholder(tf.float32, shape=[args.batch_size, None])
    phrase_plh = tf.placeholder(tf.float32, shape=[args.batch_size,
                                                   phrase_feature_dim])
    region_plh = tf.placeholder(tf.float32, shape=[args.batch_size, None,
                                                   region_feature_dim])
    train_phase_plh = tf.placeholder(tf.bool, name='train_phase')
    num_boxes_plh = tf.placeholder(tf.int32)
    
    plh = {}
    plh['num_boxes'] = num_boxes_plh
    plh['labels'] = labels_plh
    plh['phrase'] = phrase_plh
    plh['region'] = region_plh
    plh['train_phase'] = train_phase_plh
    
    test_loader = DataLoader(args, region_feature_dim, phrase_feature_dim,
                             plh, 'test')
    model = setup_model(args, phrase_plh, region_plh, train_phase_plh,
                        labels_plh, num_boxes_plh, region_feature_dim)
    if args.test:
        test(model, test_loader, model_name=args.resume)
        sys.exit()

    save_model_directory =  os.path.join('runs', args.name)
    if not os.path.exists(save_model_directory):
        os.makedirs(save_model_directory)

    train_loader = DataLoader(args, region_feature_dim, phrase_feature_dim,
                              plh, 'train')
    val_loader = DataLoader(args, region_feature_dim, phrase_feature_dim,
                            plh, 'val')

    # training with Adam
    acc, best_adam = train(model, train_loader, val_loader, args.resume)

    # finetune with SGD after loading the best model trained with Adam
    best_model_filename = os.path.join('runs', args.name, 'model_best')
    acc, best_sgd = train(model, train_loader, val_loader,
                          best_model_filename, False, acc)
    best_epoch = best_adam + best_sgd
    
    # get performance on test set
    test_acc = test(model, test_loader, model_name=best_model_filename)
    print('best model at epoch {}: {:.2f}% (val {:.2f}%)'.format(
        best_epoch, round(test_acc*100, 2), round(acc*100, 2)))

def test(model, test_loader, sess=None, model_name = None):
    if sess is None:
        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, model_name)
        
    region_weights = model[3]
    correct = 0.0
    n_iterations = test_loader.num_batches()
    for batch_id in range(n_iterations):
        feed_dict, gt_labels, num_pairs = test_loader.get_batch(batch_id)
        scores = sess.run(region_weights, feed_dict = feed_dict)
        for pair_index in range(num_pairs):
            best_region_index = np.argmax(scores[pair_index, :])
            correct += gt_labels[pair_index, best_region_index]
            
    acc = correct/len(test_loader)
    print('\n{} set localization accuracy: {:.2f}%\n'.format(
        test_loader.split, round(acc*100, 2)))
    return acc

def process_epoch(model, train_loader, sess, train_step, epoch, suffix):
    train_loader.shuffle()
    
    # extract elements from model tuple
    loss = model[0]
    region_loss = model[1]
    l1_loss = model[2]
    
    n_iterations = train_loader.num_batches()
    for batch_id in range(n_iterations):
        feed_dict, _, _ = train_loader.get_batch(batch_id)
        (_, total, region, concept_l1) = sess.run([train_step, loss,
                                                   region_loss, l1_loss],
                                                  feed_dict = feed_dict)

        if batch_id % args.info_iterval == 0:
            print('loss: {:.5f} (region: {:.5f} concept: {:.5f}) '
                  '[{}/{}] (epoch: {}) {}'.format(total, region, concept_l1,
                                                  (batch_id*args.batch_size),
                                                  len(train_loader), epoch,
                                                  suffix))

def train(model, train_loader, test_loader, model_weights, use_adam = True,
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
                best_acc = test(model, test_loader, sess)

        # model trains until args.max_epoch is reached or it no longer
        # improves on the validation set
        while (epoch - best_epoch) < args.no_gain_stop and (args.max_epoch < 1 or epoch <= args.max_epoch):
            process_epoch(model, train_loader, sess, train_step, epoch, suffix)
            saver.save(sess, os.path.join('runs', args.name, 'checkpoint'),
                       global_step = epoch)
            acc = test(model, test_loader, sess)
            if acc > best_acc:
                saver.save(sess, os.path.join('runs', args.name, 'model_best'))
                if (acc - args.minimum_gain) > best_acc:
                    best_epoch = epoch

                best_acc = acc

            epoch += 1

    return best_acc, best_epoch

if __name__ == '__main__':
    main()
