import tensorflow as tf

def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""

  # Input Layer
  input_layer = tf.reshape(features['x'], [-1, 64, 64, 3])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=16,
      kernel_size=[8, 8],
      padding="valid",
      activation=tf.nn.relu)

  #Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=7,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  #Pooling Layer #2
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, 4116])
  dense = tf.layers.dense(inputs=pool2_flat, units=256, activation=tf.nn.relu)

  # output Layer
  output_descriptors = tf.layers.dense(inputs=dense, units=16)

  predictions = {
      # Generate descriptors and corresp nearest neighbour (for PREDICT and EVAL mode)
      "descriptor": output_descriptors,
      "nearest neighbours": 0
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  #triplet and pair based
  diff_pos = tf.subtract(output_descriptors[0:len(features['x']):3], output_descriptors[1:len(features['x']):3])
  diff_neg = tf.subtract(output_descriptors[0:len(features['x']):3], output_descriptors[2:len(features['x']):3])
  square_norm_diff_pos = tf.square(tf.norm(diff_pos, ord=2, axis=1)) #axis?????
  square_norm_diff_neg = tf.square(tf.norm(diff_neg, ord=2, axis=1))

  loss_pairs = tf.reduce_sum(square_norm_diff_pos)

  m = 0.01  #margin for classification
  fraction = tf.divide(square_norm_diff_neg, tf.add(m, square_norm_diff_pos))
  loss_triplets = tf.reduce_sum(tf.maximum(0., tf.subtract(1., fraction)))
  loss = tf.add(loss_triplets, loss_pairs)

  metrics = {'descriptors:': output_descriptors}

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": 999}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

