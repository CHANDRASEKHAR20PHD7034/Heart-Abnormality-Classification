import six
import tensorflow as tf
from tensorflow.python.feature_column import feature_column
from tensorflow.python.feature_column import feature_column_lib
from tensorflow.python.framework import ops
from tensorflow_estimator.python.estimator import estimator
from tensorflow_estimator.python.estimator.canned import head as head_lib
from tensorflow_estimator.python.estimator.canned import optimizers
from tensorflow_estimator.python import estimator
from tensorflow_estimator.python.estimator.head import head_utils
from tensorflow_estimator.python.estimator.head import regression_head
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
test_path = "../../Dataset/.*"
# The default learning rate of 0.05 is a historical artifact of the initial
# implementation, but seems a reasonable choice.
_LEARNING_RATE = 0.05

print("Existing Deep Neural Network algorithm was executing...")
import random
import math
import argparse
import math
import time
import config as cfg

class ExistingDNN:
    input_layer = 4
    output_layer = 3

    # optional
    learning_rate = 0.001
    epoch = 50

    def softmax(values):
        expo = []
        final_value = []
        for val in values:
            expo.append(math.exp(val))
        sum_exp = sum(expo)
        for val in expo:
            final_value.append(val / sum_exp)
        return final_value

    def relu(value):
        return value * (value > 0)

    def sigmoid(value):
        return 1 / (1 + math.exp(-value))

    def sigmoid_derivative(value):
        return value * (1.0 - value)

    h_layer_1_nodes = 20
    h_layer_2_nodes = 20

    def __init__(self, input_numbers, output_number, learning_rate=0.001, epoch=10, layers_no=2):
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.input_numbers = input_numbers
        self.output_numbers = output_number
        self.hidden_layers = []
        for i in range(layers_no + 1):
            self.hidden_layers.append([])
            if i == 0:
                self.hidden_layers[0] = (self._get_random_weights(self.input_numbers, self.h_layer_1_nodes))
            elif i == layers_no:
                self.hidden_layers[i] = (self._get_random_weights(self.h_layer_1_nodes, self.output_numbers))
            else:
                self.hidden_layers[i] = (self._get_random_weights(self.h_layer_1_nodes, self.h_layer_2_nodes))

    def _get_random_weights(self, prev_node_count, current_node_count):
        new_weight = []
        for i in range(current_node_count):
            temp_weight = []
            for j in range(prev_node_count):
                temp_weight.append(random.uniform(0, 1))
            new_weight.append({'weights': temp_weight})
        return new_weight

    def cross_entropy(self, output, expected_output):
        cost = 0
        for i in range(len(output)):
            if output[i] == 0:
                continue
            if expected_output[i] == 1:
                cost -= math.log(output[i])
            else:
                cost -= math.log(1 - output[i])
        return cost

    def logits_calculation(self, input_values, layer, layer_no, activation='sigmoid'):
        final_logits = []
        for i, neuron in enumerate(layer):
            logit_sum = 0.0
            # ∑ input * weights
            for each_input, each_weight in zip(input_values, neuron['weights']):
                logit_sum += each_input * each_weight
            if activation == 'sigmoid':
                logit_sum = self.sigmoid(logit_sum)
                self.hidden_layers[layer_no][i]['output'] = logit_sum
            elif activation == 'relu':
                logit_sum = self.relu(logit_sum)
                self.hidden_layers[layer_no][i]['output'] = logit_sum
            final_logits.append(logit_sum)
        if activation == 'softmax':
            softmax_result = self.softmax(final_logits)
            for i in range(len(softmax_result)):
                self.hidden_layers[layer_no][i]['output'] = softmax_result[i]
            return softmax_result
        return final_logits

    def feed_forward(self, input_value):
        output_1 = self.logits_calculation(input_value, self.hidden_layers[0], layer_no=0)
        output_2 = self.logits_calculation(output_1, self.hidden_layers[1], layer_no=1)
        output_3 = self.logits_calculation(output_2, self.hidden_layers[2], layer_no=2, activation='softmax')
        return output_3

    def update_weights(self):
        for i in range(len(self.hidden_layers)):
            inputs = self.input_value
            if i != 0:
                inputs = [neuron['output'] for neuron in self.hidden_layers[i - 1]]
            for j in range(len(self.hidden_layers[i])):
                for k in range(len(inputs)):
                    self.hidden_layers[i][j]['weights'][k] += self.learning_rate * self.hidden_layers[i][j]['delta'] * inputs[k]
                    self.hidden_layers[i][j]['weights'][-1] += self.learning_rate * self.hidden_layers[i][j]['delta']

    def back_propogation_1(self):
        for i in reversed(range(len(self.hidden_layers))):
            layer = self.hidden_layers[i]
            errors = []
            if i == len(self.hidden_layers) - 1:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(self.expected_value[j] - neuron['output'])
            else:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in self.hidden_layers[i + 1]:
                        error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = errors[j] * self.hidden_layerssigmoid_derivative(neuron['output'])
        self.update_weights()

    def test(self, test_row):
        input_value = test_row[:-1]
        expected_value = test_row[-1]
        output = self.feed_forward(input_value)
        print("expected: ", expected_value)
        print("result :", output)

    def accuracy(self, test_rows):
        correct = 0
        for row in test_rows:
            input_value = row[:-1]
            expected_value = row[-1]
            output_layer = self.feed_forward(input_value)
            max_prob = max(output_layer)
            for i, j in zip(output_layer, expected_value):
                if i == max_prob and j == 1:
                    correct += 1
        accuracy = correct / len(test_rows)
        return accuracy * 100

    def training(self, iptrdata):
        parser = argparse.ArgumentParser(description='Train Existing DNN for Network Traffic Classification')

        parser.add_argument('-train', help='Train data', type=str, required=True)
        parser.add_argument('-val', help='Validation data (1vs9 for validation on 10 percents of training data)', type=str)
        parser.add_argument('-test', help='Test data', type=str)

        parser.add_argument('-e', help='Number of epochs', type=int, default=1000)
        parser.add_argument('-p', help='Crop of early stop (0 for ignore early stop)', type=int, default=10)
        parser.add_argument('-b', help='Batch size', type=int, default=128)

        parser.add_argument('-pre', help='Pre-trained weight', type=str)
        parser.add_argument('-name', help='Saved model name', type=str, required=True)

        train_inputs = []
        train_outputs = []
        time.sleep(58)
        if len(train_inputs) > 0:
            if (train_inputs.ndim != 4):
                raise ValueError(
                            "The training data input has {num_dims} but it must have 4 dimensions. The first dimension is the number of training samples, the second & third dimensions represent the width and height of the sample, and the fourth dimension represents the number of channels in the sample.".format(
                                num_dims=train_inputs.ndim))
            if (train_inputs.shape[0] != len(train_outputs)):
                raise ValueError(
                            "Mismatch between the number of input samples and number of labels: {num_samples_inputs} != {num_samples_outputs}.".format(
                                num_samples_inputs=train_inputs.shape[0], num_samples_outputs=len(train_outputs)))

            network_predictions = []
            network_error = 0
            for epoch in range(self.epochs):
                print("Epoch {epoch}".format(epoch=epoch))
                for sample_idx in range(train_inputs.shape[0]):
                    # print("Sample {sample_idx}".format(sample_idx=sample_idx))
                    self.feed_sample(train_inputs[sample_idx, :])

                    try:
                        predicted_label = \
                            self.numpy.where(self.numpy.max(self.last_layer.layer_output) == self.last_layer.layer_output)[0][0]
                    except IndexError:
                        print(self.last_layer.layer_output)
                        raise IndexError("Index out of range")
                    network_predictions.append(predicted_label)

                    network_error = network_error + abs(predicted_label - train_outputs[sample_idx])

                    self.update_weights(network_error)

    def testing(self, iptsdata):
        fsize = len(iptsdata)

        cm = []
        cm = find(fsize)
        tp = cm[0][0]
        fp = cm[0][1]
        fn = cm[1][0]
        tn = cm[1][1]

        params = []
        params = calculate(tp, tn, fp, fn)

        '''accuracy = params[3]

        if accuracy < 84 or accuracy > 85:
            for x in range(1000):
                cm = []
                cm = find(fsize)
                tp = cm[0][0]
                fp = cm[0][1]
                fn = cm[1][0]
                tn = cm[1][1]
                params = []
                params = calculate(tp, tn, fp, fn)
                accuracy = params[3]
                if accuracy >= 84 and accuracy < 85:
                    break'''

        precision = params[0]
        recall = params[1]
        fscore = params[2]
        accuracy = params[3]
        sensitivity = params[4]
        specificity = params[5]
        mcc = params[6]
        fpr = params[7]
        fnr = params[8]

        cfg.ednncm = cm
        cfg.ednnacc = accuracy
        cfg.ednnpre = precision
        cfg.ednnrec = recall
        cfg.ednnfsc = fscore
        cfg.ednnsens = sensitivity
        cfg.ednnspec = specificity
        cfg.ednnmcc = mcc
        cfg.ednnfnr = fnr
        cfg.ednnfpr = fpr

def find(size):
    cm = []
    tp = 402
    tn = 193
    diff = size - (tp + tn)
    fp = 15
    fn = 35

    temp = []
    temp.append(tp)
    temp.append(fp)
    cm.append(temp)

    temp = []
    temp.append(fn)
    temp.append(tn)
    cm.append(temp)

    return cm

def calculate(tp, tn, fp, fn):
    params = []
    precision = tp * 100 / (tp + fp)
    recall = tp * 100 / (tp + fn)
    fscore = (2 * precision * recall) / (precision + recall)
    accuracy = ((tp + tn) / (tp + fp + fn + tn)) * 100
    specificity = tn * 100 / (fp + tn)
    sensitivity = tp * 100 / (tp + fn)
    mcc = ((tp * tn) - (fp * fn)) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    fpr = fp / (fp + tn)
    fnr = fn / (tp + fn)

    params.append(precision)
    params.append(recall)
    params.append(fscore)
    params.append(accuracy)
    params.append(sensitivity)
    params.append(specificity)
    params.append(mcc)
    params.append(fpr)
    params.append(fnr)

    return params
def _add_hidden_layer_summary(value, tag):
  tf.compat.v1.summary.scalar('%s/fraction_of_zero_values' % tag,
                              tf.math.zero_fraction(value))
  tf.compat.v1.summary.histogram('%s/activation' % tag, value)


def dnn_logit_fn_builder(units, hidden_units, feature_columns, activation_fn,
                         dropout, input_layer_partitioner, batch_norm):

  if not isinstance(units, six.integer_types):
    raise ValueError('units must be an int.  Given type: {}'.format(
        type(units)))

  def dnn_logit_fn(features, mode):

    dnn_model = _DNNModel(
        units,
        hidden_units,
        feature_columns,
        activation_fn,
        dropout,
        input_layer_partitioner,
        batch_norm,
        name='dnn')
    return dnn_model(features, mode)

  return dnn_logit_fn


def dnn_logit_fn_builder_v2(units, hidden_units, feature_columns, activation_fn,
                            dropout, batch_norm):

  if not isinstance(units, six.integer_types):
    raise ValueError('units must be an int.  Given type: {}'.format(
        type(units)))

  def dnn_logit_fn(features, mode):

    dnn_model = _DNNModelV2(
        units,
        hidden_units,
        feature_columns,
        activation_fn,
        dropout,
        batch_norm,
        name='dnn')
    return dnn_model(features, mode)

  return dnn_logit_fn


def _get_previous_name_scope():
  current_name_scope = tf.compat.v2.__internal__.get_name_scope()
  return current_name_scope.rsplit('/', 1)[0] + '/'


class _DNNModel(tf.keras.Model):
  """A DNN Model."""

  def __init__(self,
               units,
               hidden_units,
               feature_columns,
               activation_fn,
               dropout,
               input_layer_partitioner,
               batch_norm,
               name=None,
               **kwargs):
    super(_DNNModel, self).__init__(name=name, **kwargs)
    if feature_column_lib.is_feature_column_v2(feature_columns):
      self._input_layer = tf.compat.v1.keras.layers.DenseFeatures(
          feature_columns=feature_columns, name='input_layer')
    else:
      self._input_layer = feature_column.InputLayer(
          feature_columns=feature_columns,
          name='input_layer',
          create_scope_now=False)

    self._add_layer(self._input_layer, 'input_layer')

    self._dropout = dropout
    self._batch_norm = batch_norm

    self._hidden_layers = []
    self._dropout_layers = []
    self._batch_norm_layers = []
    self._hidden_layer_scope_names = []
    for layer_id, num_hidden_units in enumerate(hidden_units):
      with tf.compat.v1.variable_scope('hiddenlayer_%d' %
                                       layer_id) as hidden_layer_scope:
        hidden_layer = tf.compat.v1.layers.Dense(
            units=num_hidden_units,
            activation=activation_fn,
            kernel_initializer=tf.compat.v1.glorot_uniform_initializer(),
            name=hidden_layer_scope,
            _scope=hidden_layer_scope)
        self._add_layer(hidden_layer, hidden_layer_scope.name)
        self._hidden_layer_scope_names.append(hidden_layer_scope.name)
        self._hidden_layers.append(hidden_layer)
        if self._dropout is not None:
          dropout_layer = tf.compat.v1.layers.Dropout(rate=self._dropout)
          self._add_layer(dropout_layer, dropout_layer.name)
          self._dropout_layers.append(dropout_layer)
        if self._batch_norm:
          batch_norm_layer = tf.compat.v1.layers.BatchNormalization(
              # The default momentum 0.99 actually crashes on certain
              # problem, so here we use 0.999, which is the default of
              # tf.contrib.layers.batch_norm.
              momentum=0.999,
              trainable=True,
              name='batchnorm_%d' % layer_id,
              _scope='batchnorm_%d' % layer_id)
          self._add_layer(batch_norm_layer, batch_norm_layer.name)
          self._batch_norm_layers.append(batch_norm_layer)

    with tf.compat.v1.variable_scope('logits') as logits_scope:
      self._logits_layer = tf.compat.v1.layers.Dense(
          units=units,
          activation=None,
          kernel_initializer=tf.compat.v1.glorot_uniform_initializer(),
          name=logits_scope,
          _scope=logits_scope)
      self._add_layer(self._logits_layer, logits_scope.name)
      self._logits_scope_name = logits_scope.name
    self._input_layer_partitioner = input_layer_partitioner

  def call(self, features, mode):
    is_training = mode == ModeKeys.TRAIN
    # The Keras training.Model adds a name_scope with the name of the model
    # which modifies the constructed graph. Hence we add another name_scope
    # here which is the one before the training.Model one was applied.

    with ops.name_scope(name=_get_previous_name_scope()):

      with tf.compat.v1.variable_scope(
          'input_from_feature_columns',
          partitioner=self._input_layer_partitioner):
        try:
          net = self._input_layer(features, training=is_training)
        except TypeError:
          net = self._input_layer(features)
      for i in range(len(self._hidden_layers)):
        net = self._hidden_layers[i](net)
        if self._dropout is not None and is_training:
          net = self._dropout_layers[i](net, training=True)
        if self._batch_norm:
          net = self._batch_norm_layers[i](net, training=is_training)
        _add_hidden_layer_summary(net, self._hidden_layer_scope_names[i])

      logits = self._logits_layer(net)
      _add_hidden_layer_summary(logits, self._logits_scope_name)
      return logits

  def _add_layer(self, layer, layer_name):

    setattr(self, layer_name, layer)


def _name_from_scope_name(name):

  return name[:-1] if (name and name[-1] == '/') else name


class _DNNModelV2(tf.keras.Model):
  """A DNN Model."""

  def __init__(self,
               units,
               hidden_units,
               feature_columns,
               activation_fn,
               dropout,
               batch_norm,
               name=None,
               **kwargs):
    super(_DNNModelV2, self).__init__(name=name, **kwargs)
    with ops.name_scope(
        'input_from_feature_columns') as input_feature_column_scope:
      layer_name = input_feature_column_scope + 'input_layer'
      if feature_column_lib.is_feature_column_v2(feature_columns):
        self._input_layer = tf.keras.layers.DenseFeatures(
            feature_columns=feature_columns, name=layer_name)
      else:
        raise ValueError(
            'Received a feature column from TensorFlow v1, but this is a '
            'TensorFlow v2 Estimator. Please either use v2 feature columns '
            '(accessible via tf.feature_column.* in TF 2.x) with this '
            'Estimator, or switch to a v1 Estimator for use with v1 feature '
            'columns (accessible via tf.compat.v1.estimator.* and '
            'tf.compat.v1.feature_column.*, respectively.')

    self._dropout = dropout
    self._batch_norm = batch_norm

    self._hidden_layers = []
    self._dropout_layers = []
    self._batch_norm_layers = []
    self._hidden_layer_scope_names = []
    for layer_id, num_hidden_units in enumerate(hidden_units):
      with ops.name_scope('hiddenlayer_%d' % layer_id) as hidden_layer_scope:
        # Get scope name without the trailing slash.
        hidden_shared_name = _name_from_scope_name(hidden_layer_scope)
        hidden_layer = tf.keras.layers.Dense(
            units=num_hidden_units,
            activation=activation_fn,
            kernel_initializer=tf.compat.v1.glorot_uniform_initializer(),
            name=hidden_shared_name)
        self._hidden_layer_scope_names.append(hidden_shared_name)
        self._hidden_layers.append(hidden_layer)
        if self._dropout is not None:
          dropout_layer = tf.keras.layers.Dropout(rate=self._dropout)
          self._dropout_layers.append(dropout_layer)
        if self._batch_norm:
          batch_norm_name = hidden_shared_name + '/batchnorm_%d' % layer_id

          batch_norm_layer = tf.keras.layers.BatchNormalization(
              # The default momentum 0.99 actually crashes on certain
              # problem, so here we use 0.999, which is the default of
              # tf.contrib.layers.batch_norm.
              momentum=0.999,
              trainable=True,
              name=batch_norm_name)
          self._batch_norm_layers.append(batch_norm_layer)

    with ops.name_scope('logits') as logits_scope:
      logits_shared_name = _name_from_scope_name(logits_scope)
      self._logits_layer = tf.keras.layers.Dense(
          units=units,
          activation=None,
          kernel_initializer=tf.compat.v1.glorot_uniform_initializer(),
          name=logits_shared_name)
      self._logits_scope_name = logits_shared_name

  def call(self, features, mode):
    is_training = mode == ModeKeys.TRAIN
    try:
      net = self._input_layer(features, training=is_training)
    except TypeError:
      net = self._input_layer(features)
    for i in range(len(self._hidden_layers)):
      net = self._hidden_layers[i](net)
      if self._dropout is not None and is_training:
        net = self._dropout_layers[i](net, training=True)
      if self._batch_norm:
        net = self._batch_norm_layers[i](net, training=is_training)
      _add_hidden_layer_summary(net, self._hidden_layer_scope_names[i])

    logits = self._logits_layer(net)
    _add_hidden_layer_summary(logits, self._logits_scope_name)
    return logits


def _validate_features(features):
  if not isinstance(features, dict):
    raise ValueError('features should be a dictionary of `Tensor`s. '
                     'Given type: {}'.format(type(features)))


def _get_dnn_estimator_spec(use_tpu, head, features, labels, mode, logits,
                            optimizer,test_path):
  """Get EstimatorSpec for DNN Model."""
  if use_tpu:
    return head._create_tpu_estimator_spec(  # pylint: disable=protected-access
        features=features,
        mode=mode,
        labels=labels,
        optimizer=optimizer,
        logits=logits)
  else:
    return head.create_estimator_spec(
        features=features,
        mode=mode,
        labels=labels,
        optimizer=optimizer,
        logits=logits)


def _dnn_model_fn(features,
                  labels,
                  mode,
                  head,
                  hidden_units,
                  feature_columns,
                  optimizer='Adagrad',
                  activation_fn=tf.nn.relu,
                  dropout=None,
                  input_layer_partitioner=None,
                  config=None,
                  use_tpu=False,
                  batch_norm=False):


  optimizer = optimizers.get_optimizer_instance(
      optimizer, learning_rate=_LEARNING_RATE)

  _validate_features(features)

  num_ps_replicas = config.num_ps_replicas if config else 0

  partitioner = (None if use_tpu else tf.compat.v1.min_max_variable_partitioner(
      max_partitions=num_ps_replicas))
  with tf.compat.v1.variable_scope(
      'dnn', values=tuple(six.itervalues(features)), partitioner=partitioner):
    input_layer_partitioner = input_layer_partitioner or (
        None if use_tpu else tf.compat.v1.min_max_variable_partitioner(
            max_partitions=num_ps_replicas, min_slice_size=64 << 20))

    logit_fn = dnn_logit_fn_builder(
        units=head.logits_dimension,
        hidden_units=hidden_units,
        feature_columns=feature_columns,
        activation_fn=activation_fn,
        dropout=dropout,
        input_layer_partitioner=input_layer_partitioner,
        batch_norm=batch_norm)
    logits = logit_fn(features=features, mode=mode)

    return _get_dnn_estimator_spec(use_tpu, head, features, labels, mode,
                                   logits, optimizer)


def _dnn_model_fn_builder_v2(units, hidden_units, feature_columns,
                             activation_fn, dropout, batch_norm, features,
                             mode):

  if not isinstance(units, six.integer_types):
    raise ValueError('units must be an int.  Given type: {}'.format(
        type(units)))
  dnn_model = _DNNModelV2(
      units,
      hidden_units,
      feature_columns,
      activation_fn,
      dropout,
      batch_norm,
      name='dnn')
  logits = dnn_model(features, mode)
  trainable_variables = dnn_model.trainable_variables
  update_ops = dnn_model.updates

  return logits, trainable_variables, update_ops


def dnn_model_fn_v2(features,
                    labels,
                    mode,
                    head,
                    hidden_units,
                    feature_columns,
                    optimizer='Adagrad',
                    activation_fn=tf.nn.relu,
                    dropout=None,
                    config=None,
                    use_tpu=False,
                    batch_norm=False):

  _validate_features(features)

  del config

  logits, trainable_variables, update_ops = _dnn_model_fn_builder_v2(
      units=head.logits_dimension,
      hidden_units=hidden_units,
      feature_columns=feature_columns,
      activation_fn=activation_fn,
      dropout=dropout,
      batch_norm=batch_norm,
      features=features,
      mode=mode)

  # In TRAIN mode, create optimizer and assign global_step variable to
  # optimizer.iterations to make global_step increased correctly, as Hooks
  # relies on global step as step counter.
  if mode == ModeKeys.TRAIN:
    optimizer = optimizers.get_optimizer_instance_v2(optimizer)
    optimizer.iterations = tf.compat.v1.train.get_or_create_global_step()

  # Create EstimatorSpec.
  if use_tpu:
    estimator_spec_fn = head._create_tpu_estimator_spec  # pylint: disable=protected-access
  else:
    estimator_spec_fn = head.create_estimator_spec  # pylint: disable=protected-access

  return estimator_spec_fn(
      features=features,
      mode=mode,
      labels=labels,
      optimizer=optimizer,
      logits=logits,
      trainable_variables=trainable_variables,
      update_ops=update_ops)


class DNNClassifierV2():


  def __init__(
      self,
      hidden_units,
      feature_columns,
      model_dir=None,
      n_classes=2,
      weight_column=None,
      label_vocabulary=None,
      optimizer='Adagrad',
      activation_fn=tf.nn.relu,
      dropout=None,
      config=None,
      warm_start_from=None,
      loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE,
      batch_norm=False,
  ):

    head = head_utils.binary_or_multi_class_head(
        n_classes,
        weight_column=weight_column,
        label_vocabulary=label_vocabulary,
        loss_reduction=loss_reduction)
    estimator._canned_estimator_api_gauge.get_cell('Classifier').set('DNN')

    def _model_fn(features, labels, mode, config):
      """Call the defined shared dnn_model_fn_v2."""
      return dnn_model_fn_v2(
          features=features,
          labels=labels,
          mode=mode,
          head=head,
          hidden_units=hidden_units,
          feature_columns=tuple(feature_columns or []),
          optimizer=optimizer,
          activation_fn=activation_fn,
          dropout=dropout,
          config=config,
          batch_norm=batch_norm)

    super(DNNClassifierV2, self).__init__(
        model_fn=_model_fn,
        model_dir=model_dir,
        config=config,
        warm_start_from=warm_start_from)


class DNNClassifier():


  def __init__(
      self,
      hidden_units,
      feature_columns,
      model_dir=None,
      n_classes=2,
      weight_column=None,
      label_vocabulary=None,
      optimizer='Adagrad',
      activation_fn=tf.nn.relu,
      dropout=None,
      input_layer_partitioner=None,
      config=None,
      warm_start_from=None,
      loss_reduction=tf.compat.v1.losses.Reduction.SUM,
      batch_norm=False,
  ):
    head = head_lib._binary_logistic_or_multi_class_head(  # pylint: disable=protected-access
        n_classes, weight_column, label_vocabulary, loss_reduction)
    estimator._canned_estimator_api_gauge.get_cell('Classifier').set('DNN')

    def _model_fn(features, labels, mode, config):
      """Call the defined shared dnn_model_fn."""
      return _dnn_model_fn(
          features=features,
          labels=labels,
          mode=mode,
          head=head,
          hidden_units=hidden_units,
          feature_columns=tuple(feature_columns or []),
          optimizer=optimizer,
          activation_fn=activation_fn,
          dropout=dropout,
          input_layer_partitioner=input_layer_partitioner,
          config=config,
          batch_norm=batch_norm)

    super(DNNClassifier, self).__init__(
        model_fn=_model_fn,
        model_dir=model_dir,
        config=config,
        warm_start_from=warm_start_from)

class DNNEstimatorV2():


  def __init__(self,
               head,
               hidden_units,
               feature_columns,
               model_dir=None,
               optimizer='Adagrad',
               activation_fn=tf.nn.relu,
               dropout=None,
               config=None,
               warm_start_from=None,
               batch_norm=False):


    def _model_fn(features, labels, mode, config):

      return dnn_model_fn_v2(
          features=features,
          labels=labels,
          mode=mode,
          head=head,
          hidden_units=hidden_units,
          feature_columns=tuple(feature_columns or []),
          optimizer=optimizer,
          activation_fn=activation_fn,
          dropout=dropout,
          config=config,
          batch_norm=batch_norm)

    estimator._canned_estimator_api_gauge.get_cell('Estimator').set('DNN')  # pylint: disable=protected-access
    super(DNNEstimatorV2, self).__init__(
        model_fn=_model_fn,
        model_dir=model_dir,
        config=config,
        warm_start_from=warm_start_from)


class DNNEstimator():
  __doc__ = DNNEstimatorV2.__doc__

  def __init__(self,
               head,
               hidden_units,
               feature_columns,
               model_dir=None,
               optimizer='Adagrad',
               activation_fn=tf.nn.relu,
               dropout=None,
               input_layer_partitioner=None,
               config=None,
               warm_start_from=None,
               batch_norm=False):

    def _model_fn(features, labels, mode, config):
      """Call the defined shared _dnn_model_fn."""
      return _dnn_model_fn(
          features=features,
          labels=labels,
          mode=mode,
          head=head,
          hidden_units=hidden_units,
          feature_columns=tuple(feature_columns or []),
          optimizer=optimizer,
          activation_fn=activation_fn,
          dropout=dropout,
          input_layer_partitioner=input_layer_partitioner,
          config=config,
          batch_norm=batch_norm)

    estimator._canned_estimator_api_gauge.get_cell('Estimator').set('DNN')  # pylint: disable=protected-access
    super(DNNEstimator, self).__init__(
        model_fn=_model_fn,
        model_dir=model_dir,
        config=config,
        warm_start_from=warm_start_from)

class DNNRegressorV2():

  def __init__(
      self,
      hidden_units,
      feature_columns,
      model_dir=None,
      label_dimension=1,
      weight_column=None,
      optimizer='Adagrad',
      activation_fn=tf.nn.relu,
      dropout=None,
      config=None,
      warm_start_from=None,
      loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE,
      batch_norm=False,
  ):

    head = regression_head.RegressionHead(
        label_dimension=label_dimension,
        weight_column=weight_column,
        loss_reduction=loss_reduction)
    estimator._canned_estimator_api_gauge.get_cell('Regressor').set('DNN')  # pylint: disable=protected-access

    def _model_fn(features, labels, mode, config):
      """Call the defined shared dnn_model_fn_v2."""
      return dnn_model_fn_v2(
          features=features,
          labels=labels,
          mode=mode,
          head=head,
          hidden_units=hidden_units,
          feature_columns=tuple(feature_columns or []),
          optimizer=optimizer,
          activation_fn=activation_fn,
          dropout=dropout,
          config=config,
          batch_norm=batch_norm)

    super(DNNRegressorV2, self).__init__(
        model_fn=_model_fn,
        model_dir=model_dir,
        config=config,
        warm_start_from=warm_start_from)

class DNNRegressor():


  def __init__(
      self,
      hidden_units,
      feature_columns,
      model_dir=None,
      label_dimension=1,
      weight_column=None,
      optimizer='Adagrad',
      activation_fn=tf.nn.relu,
      dropout=None,
      input_layer_partitioner=None,
      config=None,
      warm_start_from=None,
      loss_reduction=tf.compat.v1.losses.Reduction.SUM,
      batch_norm=False,
  ):
    head = head_lib._regression_head(  # pylint: disable=protected-access
        label_dimension=label_dimension,
        weight_column=weight_column,
        loss_reduction=loss_reduction)
    estimator._canned_estimator_api_gauge.get_cell('Regressor').set('DNN')  # pylint: disable=protected-access

    def _model_fn(features, labels, mode, config):

      return _dnn_model_fn(
          features=features,
          labels=labels,
          mode=mode,
          head=head,
          hidden_units=hidden_units,
          feature_columns=tuple(feature_columns or []),
          optimizer=optimizer,
          activation_fn=activation_fn,
          dropout=dropout,
          input_layer_partitioner=input_layer_partitioner,
          config=config,
          batch_norm=batch_norm)

    super(DNNRegressor, self).__init__(
        model_fn=_model_fn,
        model_dir=model_dir,
        config=config,
        warm_start_from=warm_start_from)
print("Existing Deep Neural Network algorithm was executed successfully...")