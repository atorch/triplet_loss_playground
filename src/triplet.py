"""Semi-hard triplet loss.

A port of `tfa.losses.TripletSemiHardLoss` (TensorFlow Addons is deprecated and doesn't
install on current TensorFlow/Python), following FaceNet (Schroff et al. 2015):
for every (anchor, positive) pair in the batch, pick the closest negative that is still
*farther* from the anchor than the positive is ("semi-hard"); if there isn't one, use the
farthest negative. Loss = mean over positive pairs of max(0, d(a, p) - d(a, n) + margin).

Labels that appear only once in a batch (e.g. outlier-exposure samples that each get a
unique label) never act as anchors -- they have no positive -- but they do act as negatives,
so they get pushed away from every digit cluster without being pulled toward each other.
"""

import tensorflow as tf


def _pairwise_distance(embeddings):
    sq_norms = tf.reduce_sum(tf.square(embeddings), axis=1, keepdims=True)
    sq_dists = sq_norms + tf.transpose(sq_norms) - 2.0 * tf.matmul(embeddings, embeddings, transpose_b=True)
    sq_dists = tf.maximum(sq_dists, 0.0)
    # Avoid the infinite gradient of sqrt at 0
    zero_mask = tf.cast(sq_dists <= 0.0, tf.float32)
    dists = tf.sqrt(sq_dists + zero_mask * 1e-16) * (1.0 - zero_mask)
    n = tf.shape(embeddings)[0]
    return dists * (tf.ones_like(dists) - tf.eye(n))


def _masked_maximum(data, mask, axis=1):
    axis_min = tf.reduce_min(data, axis, keepdims=True)
    return tf.reduce_max((data - axis_min) * mask, axis, keepdims=True) + axis_min


def _masked_minimum(data, mask, axis=1):
    axis_max = tf.reduce_max(data, axis, keepdims=True)
    return tf.reduce_min((data - axis_max) * mask, axis, keepdims=True) + axis_max


def triplet_semihard_loss(y_true, y_pred, margin=1.0):
    labels = tf.reshape(tf.cast(y_true, tf.int64), [-1, 1])
    embeddings = tf.cast(y_pred, tf.float32)
    pdist = _pairwise_distance(embeddings)

    adjacency = tf.equal(labels, tf.transpose(labels))
    adjacency_not = tf.logical_not(adjacency)
    batch_size = tf.size(labels)

    # Row (i * B + j) of the tiled matrix asks: for anchor j and positive i, which negatives k
    # have d(j, k) > d(j, i)?
    pdist_tile = tf.tile(pdist, [batch_size, 1])
    mask = tf.logical_and(
        tf.tile(adjacency_not, [batch_size, 1]),
        pdist_tile > tf.reshape(tf.transpose(pdist), [-1, 1]),
    )
    mask_final = tf.transpose(
        tf.reshape(tf.reduce_sum(tf.cast(mask, tf.float32), 1, keepdims=True) > 0.0, [batch_size, batch_size])
    )
    adjacency_not_f = tf.cast(adjacency_not, tf.float32)
    mask_f = tf.cast(mask, tf.float32)

    negatives_outside = tf.transpose(tf.reshape(_masked_minimum(pdist_tile, mask_f), [batch_size, batch_size]))
    negatives_inside = tf.tile(_masked_maximum(pdist, adjacency_not_f), [1, batch_size])
    semi_hard_negatives = tf.where(mask_final, negatives_outside, negatives_inside)

    loss_mat = margin + pdist - semi_hard_negatives
    mask_positives = tf.cast(adjacency, tf.float32) - tf.eye(batch_size)
    num_positives = tf.maximum(tf.reduce_sum(mask_positives), 1.0)
    return tf.reduce_sum(tf.maximum(loss_mat * mask_positives, 0.0)) / num_positives
