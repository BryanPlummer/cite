import pickle
import numbers
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import nn
from tensorflow.python.platform import tf_logging as logging

from tensorflow.contrib.layers.python.layers import batch_norm
from tensorflow.contrib.layers.python.layers import convolution2d
from tensorflow.contrib.layers.python.layers import fully_connected
from tensorflow.contrib.layers.python.layers import l2_regularizer

def add_fc(x, outdim, train_phase_plh, scope_in):
    """Returns the output of a FC-BNORM-ReLU sequence.

    Arguments:
    x -- input tensor
    outdim -- desired output dimensions
    train_phase_plh -- indicator whether model is in training mode
    scope_in -- scope prefix for the desired layers
    """
    l2_reg = tf.contrib.layers.l2_regularizer(0.0005)
    fc = tf.contrib.layers.fully_connected(x, outdim, activation_fn = None,
                                           weights_regularizer = l2_reg,
                                           scope = scope_in + '/fc')
    fc_bnorm = batch_norm_layer(fc, train_phase_plh, scope_in + '/bnorm')
    return tf.nn.relu(fc_bnorm, scope_in + '/relu')

def concept_layer(x, outdim, train_phase_plh, concept_id, weights):
    """Returns the weighted value of a fully connected layer.

    Arguments:
    x -- input tensor
    outdim -- desired output dimensions
    train_phase_plh -- indicator whether model is in training mode
    concept_id -- identfier for the desired concept layer
    weights -- vector of weights to be applied the concept outputs
    """
    concept = add_fc(x, outdim, train_phase_plh, 'concept_%i' % concept_id)
    weighted_concept = concept * tf.expand_dims(tf.expand_dims(weights[:, :, concept_id-1], 2), 2)
    return weighted_concept

def batch_norm_layer(x, train_phase, scope_bn):
    """Returns the output of a batch norm layer."""
    bn = tf.contrib.layers.batch_norm(x, decay=0.99, center=True, scale=True,
                                      is_training=train_phase,
                                      reuse=None,
                                      trainable=True,
                                      updates_collections=None,
                                      scope=scope_bn)
    return bn

def embedding_branch(x, embed_dim, train_phase_plh, scope_in, do_l2norm = True, outdim = None, norm_axis = 1):
    """Applies a pair of fully connected layers to the input tensor.

    Arguments:
    x -- input_tensor
    embed_dim -- dimension of the input to the second fully connected layer
    train_phase_plh -- indicator whether model is in training mode
    scope_in -- scope prefix for the desired layers
    do_l2norm -- indicates if the output should be l2 normalized
    outdim -- dimension of the output embedding, if None outdim=embed_dim
    """
    embed_fc1 = add_fc(x, embed_dim, train_phase_plh, scope_in + '_embed_1')
    if outdim is None:
        outdim = embed_dim

    l2_reg = tf.contrib.layers.l2_regularizer(0.001)
    embed_fc2 = fully_connected(embed_fc1, outdim, activation_fn = None,
                                weights_regularizer = l2_reg,
                                scope = scope_in + '_embed_2')
    if do_l2norm:
        embed_fc2 = tf.nn.l2_normalize(embed_fc2, norm_axis)

    return embed_fc2

def extract_axis_1(data, ind):
    """
    Get specified elements along the first axis of tensor.
    :param data: Tensorflow tensor that will be subsetted.
    :param ind: Indices to take (one for each element along axis 0 of data).
    :return: Subsetted tensor.

    function copied from - https://stackoverflow.com/questions/41273361/get-the-last-output-of-a-dynamic-rnn-in-tensorflow
    """
    batch_range = tf.range(tf.shape(data)[0])
    indices = tf.stack([batch_range, ind], axis=1)
    res = tf.gather_nd(data, indices)
    return res

def weight_l2_regularizer(initial_weights, scale, scope=None):
  """Returns a function that can be used to apply L2 regularization to weights.
  Small values of L2 can help prevent overfitting the training data.
  Args:
    scale: A scalar multiplier `Tensor`. 0.0 disables the regularizer.
    scope: An optional scope name.
  Returns:
    A function with signature `l2(weights)` that applies L2 regularization.
  Raises:
    ValueError: If scale is negative or if scale is not a float.
  """
  if isinstance(scale, numbers.Integral):
    raise ValueError('scale cannot be an integer: %s' % (scale,))
  if isinstance(scale, numbers.Real):
    if scale < 0.:
      raise ValueError('Setting a scale less than 0 on a regularizer: %g.' %
                       scale)
    if scale == 0.:
      logging.info('Scale of 0 disables regularizer.')
      return lambda _: None

  def l2(weights):
    """Applies l2 regularization to weights."""
    with ops.name_scope(scope, 'l2_regularizer', [weights]) as name:
      my_scale = ops.convert_to_tensor(scale,
                                       dtype=weights.dtype.base_dtype,
                                       name='scale')
      weight_diff = initial_weights - weights
      return standard_ops.multiply(my_scale, nn.l2_loss(weight_diff), name=name)

  return l2

def setup_initialize_fc_layers(args, feats, parameters, scope_in, train_phase, norm_axis = 2):
    for i, params in enumerate(parameters):
        scaling = params['scaling']
        outdim = len(scaling)
        cca_mean, cca_proj = params[scope_in + '_mean'], params[scope_in + '_proj']
        weights_init = tf.constant_initializer(cca_proj, dtype=tf.float32)
        weight_reg = weight_l2_regularizer(params[scope_in + '_proj'], args.cca_weight_reg)
        if (i + 1) < len(parameters):
            activation_fn = tf.nn.relu
        else:
            activation_fn = None
            
        feats = fully_connected(feats - cca_mean, outdim, activation_fn=activation_fn,
                                weights_initializer = weights_init,
                                weights_regularizer = weight_reg,
                                #trainable=False,
                                scope = scope_in + '_embed_' + str(i)) * scaling

    feats = tf.nn.l2_normalize(feats, norm_axis, epsilon=1e-10)
    return feats

class CITE():
    def __init__(self, args, vecs = None, max_length = None, region_feature_dim = None):
        self.args = args
        self.embeddings = vecs
        self.phrase_length = max_length
        self.region_dim = region_feature_dim
        self.final_embed = self.args.dim_embed
        self.embed_dim = self.final_embed * 4
        self.train_phase = None
        self.labels = None

    def compute_loss(self, region_scores, concept_loss, embed_l2reg):
        labels = tf.reshape(self.labels, [self.args.batch_size, self.phrases_per_image, self.boxes_per_image])
        ind_labels = tf.abs(labels)

        eps = 1e-10
        num_samples = tf.reduce_sum(ind_labels) + eps
        region_loss = tf.reduce_sum(tf.log(1+tf.exp(-region_scores*labels))*ind_labels)/num_samples
        total_loss = region_loss + concept_loss * self.args.embed_l1 + embed_l2reg * self.args.embed_weight
        return total_loss, region_loss

    def get_phrase_scores(self, phrase_embed, region_embed, concept_weights):
        elementwise_prod = tf.expand_dims(phrase_embed, 2)*tf.expand_dims(region_embed, 1)
        joint_embed_1 = add_fc(elementwise_prod, self.embed_dim, self.train_phase, 'joint_embed_1')
        joint_embed_2 = concept_layer(joint_embed_1, self.final_embed, self.train_phase, 1, concept_weights)
        for concept_id in range(2, self.args.num_embeddings+1):
            joint_embed_2 += concept_layer(joint_embed_1, self.final_embed, self.train_phase,
                                           concept_id, concept_weights)

        joint_embed_3 = fully_connected(joint_embed_2, 1, activation_fn=None ,
                                        weights_regularizer = l2_regularizer(0.005),
                                        scope = 'joint_embed_3')
        joint_embed_3 = tf.squeeze(joint_embed_3, [3])
        region_prob = 1. / (1. + tf.exp(-joint_embed_3))
        return region_prob, joint_embed_3

    def get_max_phrase_scores(self, phrase_embed, concept_weights, region_embed):
        if self.train_phase is None:
            self.set_region_placeholders()
            self.set_phrase_placeholders()

        region_embed = tf.reshape(region_embed, shape=[self.args.batch_size, self.boxes_per_image, self.embed_dim])
        phrase_embed = tf.reshape(phrase_embed, shape=[1, self.phrases_per_image, self.embed_dim])
        concept_weights = tf.reshape(concept_weights, shape=[1, self.phrases_per_image, self.args.num_embeddings])
        region_prob, _ = self.get_phrase_scores(phrase_embed, region_embed, concept_weights)
        
        best_index = tf.reshape(tf.argmax(region_prob, axis=2), [-1])
        ind = tf.stack([tf.cast(tf.range(self.phrases_per_image * self.args.batch_size), tf.int64), best_index], axis=1)
        region_prob = tf.gather_nd(tf.reshape(region_prob, [self.phrases_per_image * self.args.batch_size, -1]), ind)
        
        best_index = tf.reshape(best_index, [self.args.batch_size, self.phrases_per_image])
        region_prob = tf.reshape(region_prob, [self.args.batch_size, self.phrases_per_image])
        return region_prob, best_index

    def encode_regions(self):
        if self.train_phase is None:
            self.set_region_placeholders()

        region_plh = tf.reshape(self.regions, [-1, self.boxes_per_image, self.region_dim])
        if self.args.cca_parameters:
            parameters = pickle.load(open(self.args.cca_parameters, 'rb'))
            region_embed = setup_initialize_fc_layers(self.args, self.regions, parameters, 'vis', self.train_phase, norm_axis=self.args.region_norm_axis)
        else:
            region_embed = embedding_branch(self.regions, self.embed_dim, self.train_phase, 'region', norm_axis=self.args.region_norm_axis)

        return region_embed

    def encode_phrases(self):
        if self.train_phase is None:
            self.set_phrase_placeholders()

        phrase_plh = tf.reshape(self.phrases, [-1, self.phrases_per_image, self.phrase_length])

        # sometimes finetuning word embedding helps (with l2 reg), but often doesn't
        # seem to make a big difference
        word_embeddings = tf.get_variable('word_embeddings', self.embeddings.shape, initializer=tf.constant_initializer(self.embeddings), trainable = self.args.embedding_ft)
        embedded_words = tf.nn.embedding_lookup(word_embeddings, self.phrases)

        embed_l2reg = tf.squeeze(tf.zeros(1))
        if self.args.embedding_ft:
            embed_l2reg = tf.nn.l2_loss(word_embeddings - vecs)

        eps = 1e-10
        if self.args.language_model == 'gru':
            phrases = tf.reshape(self.phrases, [-1, self.phrase_length])
            source_sequence_length = tf.reduce_sum(tf.cast(phrases > 0, tf.int32), 1)
            embedded_words = tf.reshape(embedded_words, [-1, self.phrase_length, self.embeddings.shape[1]])
            encoder_cell = tf.nn.rnn_cell.GRUCell(self.final_embed)
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                encoder_cell, embedded_words, dtype=encoder_cell.dtype,
                sequence_length=source_sequence_length)
            final_outputs = extract_axis_1(encoder_outputs, source_sequence_length-1)
            phrase_input = tf.reshape(final_outputs, [-1, self.phrases_per_image, self.final_embed])

            outputs = fully_connected(phrase_input, self.embed_dim, activation_fn = None,
                                      weights_regularizer = tf.contrib.layers.l2_regularizer(0.005),
                                      scope = 'phrase_encoder')
            phrase_embed = tf.nn.l2_normalize(outputs, 2, epsilon=eps)
        else:
            num_words = tf.reduce_sum(tf.to_float(self.phrases > 0), 2, keep_dims=True) + eps
            phrase_input = tf.nn.l2_normalize(tf.reduce_sum(embedded_words, 2) / num_words, 2)
            if self.args.language_model == 'attend':
                context_vector = tf.tile(tf.expand_dims(phrase_input, 2), (1, 1, self.phrase_length, 1))
                attention_inputs = tf.concat((context_vector, embedded_words), 3)
                attention_weights = fully_connected(attention_inputs, 1, 
                                                    weights_regularizer = l2_regularizer(0.0005),
                                                    scope = 'self_attend')
                attention_weights = tf.nn.softmax(tf.squeeze(attention_weights))
                phrase_input = tf.nn.l2_normalize(tf.reduce_sum(embedded_words * tf.expand_dims(attention_weights, 3), 2), 2)
                phrase_input = tf.reshape(phrase_input, [-1, self.phrases_per_image, self.embeddings.shape[1]])
                
            if self.args.cca_parameters:
                parameters = pickle.load(open(self.args.cca_parameters, 'rb'))
                phrase_embed = setup_initialize_fc_layers(self.args, phrase_input, parameters, 'lang', self.train_phase)
            else:
                phrase_embed = embedding_branch(phrase_input, self.embed_dim, self.train_phase, 'phrase')
                
        concept_weights = embedding_branch(phrase_input, self.embed_dim, self.train_phase, 'concept_weight',
                                           do_l2norm = False, outdim = self.args.num_embeddings)
        concept_loss = tf.reduce_sum(tf.norm(concept_weights, axis=2, ord=1)) / self.phrase_count
        concept_weights = tf.nn.softmax(concept_weights)
        return phrase_embed, concept_weights, concept_loss, embed_l2reg

    def get_placeholders(self, placeholders = {}):
        placeholders['labels'] = self.labels
        if self.train_phase is None:
            self.set_phrase_placeholders()
            self.set_region_placeholders()

        placeholders = self.get_region_placeholders(placeholders)
        placeholders = self.get_phrase_placeholders(placeholders)
        return placeholders

    def get_region_placeholders(self, placeholders = {}):
        if self.train_phase is None:
            self.set_region_placeholders()

        placeholders['regions'] = self.regions
        placeholders['train_phase'] = self.train_phase
        placeholders['boxes_per_image'] = self.boxes_per_image
        return placeholders

    def get_phrase_placeholders(self, placeholders = {}):
        if self.train_phase is None:
            self.set_phrase_placeholders()

        placeholders['phrases'] = self.phrases
        placeholders['train_phase'] = self.train_phase
        placeholders['phrases_per_image'] = self.phrases_per_image
        placeholders['phrase_count'] = self.phrase_count
        return placeholders

    def set_region_placeholders(self):
        if self.train_phase is None:
            self.train_phase = tf.placeholder(tf.bool, name='train_phase')

        self.boxes_per_image = tf.placeholder(tf.int32)
        self.regions = tf.placeholder(tf.float32, shape=[None, None, None])

    def set_phrase_placeholders(self):
        if self.train_phase is None:
            self.train_phase = tf.placeholder(tf.bool, name='train_phase')

        self.phrases_per_image = tf.placeholder(tf.int32)
        self.phrase_count = tf.placeholder(tf.float32)
        self.phrases = tf.placeholder(tf.int32, shape=[None, None, None])
        
    def setup_model(self):
        """
        Defines the computational graph used for the CITE model

        Returns:
          total_loss -- weighted combination of the region and concept loss
          region_loss -- logistic loss for phrase-region prediction
          concept_loss -- L1 loss for the output of the concept weight branch
          region_prob -- each row contains the probability a region is associated with a phrase
        """
        self.set_region_placeholders()
        self.set_phrase_placeholders()
        self.labels = tf.placeholder(tf.float32, shape=[None, None, None])

        phrase_embed, concept_weights, concept_loss, embed_l2reg = self.encode_phrases()
        region_embed = self.encode_regions()
        region_prob, region_scores = self.get_phrase_scores(phrase_embed, region_embed, concept_weights)
        total_loss, region_loss = self.compute_loss(region_scores, concept_loss, embed_l2reg)
        return total_loss, region_loss, concept_loss, region_prob
