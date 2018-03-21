import tensorflow as tf

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
    concept = tf.reshape(concept, [tf.shape(concept)[0], -1])
    weighted_concept = concept * tf.expand_dims(weights[:, concept_id-1], 1)
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

def embedding_branch(x, embed_dim, train_phase_plh, scope_in, do_l2norm = True, outdim = None):
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
    embed_fc2 = tf.contrib.layers.fully_connected(embed_fc1, outdim, 
                                                  activation_fn = None,
                                                  weights_regularizer = l2_reg,
                                                  scope = scope_in + '_embed_2')
    if do_l2norm:
        embed_fc2 = tf.nn.l2_normalize(embed_fc2, 1)

    return embed_fc2

def setup_model(args, phrase_plh, region_plh, train_phase_plh, labels_plh, num_boxes_plh, region_feature_dim):
    """Describes the computational graph and returns the losses and outputs.

    Arguments:
    args -- command line arguments passed into the main function
    phrase_plh -- tensor containing the phrase features
    region_plh -- tensor containing the region features
    train_phase_plh -- indicator whether model is in training mode
    labels_plh -- indicates positive (1), negative (-1), or ignore (0)
    num_boxes_plh -- number of boxes per example in the batch
    region_feature_dim -- dimensions of the region features

    Returns:
    total_loss -- weighted combination of the region and concept loss
    region_loss -- logistic loss for phrase-region prediction
    concept_loss -- L1 loss for the output of the concept weight branch
    region_prob -- each row contains the probability a region is associated with a phrase
    """
    final_embed = args.dim_embed
    embed_dim = final_embed * 4
    phrase_embed = embedding_branch(phrase_plh, embed_dim, train_phase_plh, 'phrase')
    region_embed = embedding_branch(region_plh, embed_dim, train_phase_plh, 'region')
    concept_weights = embedding_branch(phrase_plh, embed_dim, train_phase_plh, 'concept_weight',
                                       do_l2norm = False, outdim = args.num_embeddings)
    concept_loss = tf.reduce_mean(tf.norm(concept_weights, axis=1, ord=1))
    concept_weights = tf.nn.softmax(concept_weights)
    
    elementwise_prod = tf.expand_dims(phrase_embed, 1)*region_embed
    joint_embed_1 = add_fc(elementwise_prod, embed_dim, train_phase_plh, 'joint_embed_1')
    joint_embed_2 = concept_layer(joint_embed_1, final_embed, train_phase_plh, 1, concept_weights)
    for concept_id in range(2, args.num_embeddings+1):
        joint_embed_2 += concept_layer(joint_embed_1, final_embed, train_phase_plh,
                                       concept_id, concept_weights)
        
    joint_embed_2 = tf.reshape(joint_embed_2, [tf.shape(joint_embed_2)[0], num_boxes_plh, final_embed])
    joint_embed_3 = tf.contrib.layers.fully_connected(joint_embed_2, 1, activation_fn=None ,
                                                      weights_regularizer = tf.contrib.layers.l2_regularizer(0.005),
                                                      scope = 'joint_embed_3')
    joint_embed_3 = tf.squeeze(joint_embed_3, [2])
    region_prob = 1. / (1. + tf.exp(-joint_embed_3))
    
    ind_labels = tf.abs(labels_plh)
    num_samples = tf.reduce_sum(ind_labels)
    region_loss = tf.reduce_sum(tf.log(1+tf.exp(-joint_embed_3*labels_plh))*ind_labels)/num_samples
    total_loss = region_loss + concept_loss * args.embed_l1
    return total_loss, region_loss, concept_loss, region_prob
