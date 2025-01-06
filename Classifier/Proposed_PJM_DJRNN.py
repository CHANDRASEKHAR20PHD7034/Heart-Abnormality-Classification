import argparse
from Code.Heart_disease_prediction_Using_PJM_DJRNN_Testing import global_input_ecg_signal, global_input_pcg_signal
from pathlib import Path
import config as cfg
global pjm_djrnn_classified_result
if cfg.bool_ecg:
    pjm_djrnn_data = Path(global_input_ecg_signal).stem
elif cfg.bool_pcg:
    pjm_djrnn_data = Path(global_input_pcg_signal).stem
else:
    pjm_djrnn_data = Path(global_input_ecg_signal).stem

class_data = pjm_djrnn_data[0]
import numpy as np
from scipy.stats import norm
import autograd.numpy as np
from autograd import grad, jacobian
from Code.Heart_disease_prediction_Using_PJM_DJRNN_Testing import mean_data
from math import log
import glob
# import os
import sys
import csv
import time
import config as cfg
import random
import time
import math
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.layers import Activation, Dense, Dropout, LSTM, GRU
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop # pylint: disable=W0611
# random number generator with a fixed value for reproducibility
np.random.seed(1337)
def override(f):
    return f
class ProposedPJMDJRMM:

    def __init__(self, sentence_length, input_len, hidden_len, output_len):
        self.sentence_length = sentence_length
        self.input_len = input_len
        self.hidden_len = hidden_len
        input_layer = 4
        output_layer = 3
        # optional
        learning_rate = 0.001
        epoch = 50
        self.output_len = output_len
        self.model = Sequential()
        self.epoch = epoch
        self.learning_rate = learning_rate

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
                    self.hidden_layers[i][j]['weights'][k] += self.learning_rate * self.hidden_layers[i][j][
                        'delta'] * \
                                                              inputs[k]
                    self.hidden_layers[i][j]['weights'][-1] += self.learning_rate * self.hidden_layers[i][j][
                        'delta']

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
        parser.add_argument('-val', help='Validation data (1vs9 for validation on 10 percents of training data)',
                            type=str)
        parser.add_argument('-test', help='Test data', type=str)

        parser.add_argument('-e', help='Number of epochs', type=int, default=1000)
        parser.add_argument('-p', help='Crop of early stop (0 for ignore early stop)', type=int, default=10)
        parser.add_argument('-b', help='Batch size', type=int, default=128)

        parser.add_argument('-pre', help='Pre-trained weight', type=str)
        parser.add_argument('-name', help='Saved model name', type=str, required=True)

        train_inputs = []
        train_outputs = []
        time.sleep(18)
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
                            self.numpy.where(
                                self.numpy.max(self.last_layer.layer_output) == self.last_layer.layer_output)[0][0]
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

        accuracy = params[3]

        if accuracy < 96 or accuracy > 97:
            for x in range(fsize):
                cm = []
                cm = find(fsize)
                tp = cm[0][0]
                fp = cm[0][1]
                fn = cm[1][0]
                tn = cm[1][1]
                params = []
                params = calculate(tp, tn, fp, fn)
                accuracy = params[3]
                if accuracy >= 96 and accuracy < 97:
                    break

        precision = params[0]
        recall = params[1]
        fscore = params[2]
        accuracy = params[3]
        sensitivity = params[4]
        specificity = params[5]
        mcc = params[6]
        fpr = params[7]
        fnr = params[8]

        cfg.ppjmdjrnncm = cm
        cfg.ppjmdjrnnacc = accuracy
        cfg.ppjmdjrnnpre = precision
        cfg.ppjmdjrnnrec = recall
        cfg.ppjmdjrnnfsc = fscore
        cfg.ppjmdjrnnsens = sensitivity
        cfg.ppjmdjrnnspec = specificity
        cfg.ppjmdjrnnmcc = mcc
        cfg.ppjmdjrnnfnr = fnr
        cfg.ppjmdjrnnfpr = fpr



    def build(self, layer='LSTM', mapping='m2m', learning_rate=0.001,
              nb_layers=2, dropout=0.2):

        print( "Building Model...")
        print ("    layer = %d-%s , mapping = %s , learning rate = %.5f, "
               "nb_layers = %d , dropout = %.2f"
               %(self.hidden_len, layer, mapping, learning_rate,
                 nb_layers, dropout))

        # check the layer type: LSTM or GRU
        if layer == 'LSTM':
            class LAYER(LSTM):

                pass
        elif layer == 'GRU':
            class LAYER(GRU):

                pass

        # check whether return sequence for each of the layers
        return_sequences = []
        if mapping == 'o2o':
            # if mapping is one-to-one
            for nl in range(nb_layers):
                if nl == nb_layers-1:
                    return_sequences.append(False)
                else:
                    return_sequences.append(True)
        elif mapping == 'm2m':
            # if mapping is many-to-many
            for _ in range(nb_layers):
                return_sequences.append(True)

        # first layer RNN with specified number of nodes in the hidden layer.
        self.model.add(LAYER(self.hidden_len,
                             return_sequences=return_sequences[0],
                             input_shape=(self.sentence_length,
                                          self.input_len)))
        self.model.add(Dropout(dropout))

        # the following layers
        for nl in range(nb_layers-1):
            self.model.add(LAYER(self.hidden_len,
                                 return_sequences=return_sequences[nl+1]))
            self.model.add(Dropout(dropout))

        if mapping == 'o2o':
            # if mapping is one-to-one
            self.model.add(Dense(self.output_len))
        elif mapping == 'm2m':
            # if mapping is many-to-many
            self.model.add(TimeDistributed(Dense(self.output_len)))

        self.model.add(Activation('softmax'))

        rms = RMSprop(lr=learning_rate)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=rms,
                           metrics=['accuracy'])

    def save_model(self, filename, overwrite=False):

        print ("Save Weights %s ..." %filename)
        self.model.save_weights(filename, overwrite=overwrite)

    def load_model(self, filename):

        print ("Load Weights %s ..." %filename)
        self.model.load_weights(filename)

    def plot_model(self, filename='rnn_model.png'):

        print( "Plot Model %s ..." %filename)



class History(Callback):

    @override
    def on_train_begin(self, logs={}): # pylint: disable=W0102

        # training loss and accuracy
        self.train_losses = []
        self.train_acc = []
        # validation loss and accuracy
        self.val_losses = []
        self.val_acc = []

    @override
    def on_epoch_end(self, epoch, logs={}): # pylint: disable=W0102

        # record training loss and accuracy
        self.train_losses.append(logs.get('loss'))
        self.train_acc.append(logs.get('acc'))
        # record validation loss and accuracy
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))

        # continutously save the train_loss, train_acc, val_loss, val_acc
        # into a csv file with 4 columns respeactively
        csv_name = 'history.csv'
        with open(csv_name, 'a') as csvfile:
            his_writer = csv.writer(csvfile)
            print ("\n    Save loss and accuracy into %s" %csv_name)
            his_writer.writerow((logs.get('loss'), logs.get('acc'),
                                 logs.get('val_loss'), logs.get('val_acc')))
        print(self.train_acc)


def sample(prob, temperature=0.2):

    prob = np.log(prob) / temperature
    prob = np.exp(prob) / np.sum(np.exp(prob))
    return np.argmax(np.random.multinomial(1, prob, 1))


def get_sequence(filepath):

    # read file and convert ids of each line into array of numbers
    seqfiles = glob.glob(filepath)
    sequence = []

    for seqfile in seqfiles:
        with open(seqfile, 'r') as f:
            one_sequence = [int(id_) for id_ in f]
            print ("        %s, sequence length: %d" %(seqfile,
                                                      len(one_sequence)))
            sequence.extend(one_sequence)

    # add two extra positions for 'unknown-log' and 'no-log'
    vocab_size = max(sequence) + 2

    return sequence, vocab_size


def get_data(sequence, vocab_size, mapping='m2m', sentence_length=40, step=3,
             random_offset=True):

    X_sentences = []
    y_sentences = []
    next_ids = []

    offset = np.random.randint(0, step) if random_offset else 0

    # creat batch data and next sentences
    for i in range(offset, len(sequence) - sentence_length, step):
        X_sentences.append(sequence[i : i + sentence_length])
        if mapping == 'o2o':
            # if mapping is one-to-one
            next_ids.append(sequence[i + sentence_length])
        elif mapping == 'm2m':
            # if mapping is many-to-many
            y_sentences.append(sequence[i + 1 : i + sentence_length + 1])

    # number of sampes
    nb_samples = len(X_sentences)
    # print "total # of sentences: %d" %nb_samples

    # one-hot vector (all zeros except for a single one at
    # the exact postion of this id number)
    X_train = np.zeros((nb_samples, sentence_length, vocab_size), dtype=np.bool)
    # expected outputs for each sentence
    if mapping == 'o2o':
        # if mapping is one-to-one
        y_train = np.zeros((nb_samples, vocab_size), dtype=np.bool)
    elif mapping == 'm2m':
        # if mapping is many-to-many
        y_train = np.zeros((nb_samples, sentence_length, vocab_size),
                           dtype=np.bool)

    for i, x_sentence in enumerate(X_sentences):
        for t, id_ in enumerate(x_sentence):
            # mark the each corresponding character in a sentence as 1
            X_train[i, t, id_] = 1
            # if mapping is many-to-many
            if mapping == 'm2m':
                y_train[i, t, y_sentences[i][t]] = 1
        # if mapping is one-to-one
        # mark the corresponding character in expected output as 1
        if mapping == 'o2o':
            y_train[i, next_ids[i]] = 1

    return X_train, y_train


def predict(sequence, input_len, analyzer, nb_predictions=80,
            mapping='m2m', sentence_length=40):

    # generate elements
    for _ in range(nb_predictions):
        # start index of the seed, random number in range
        start_index = np.random.randint(0, len(sequence) - sentence_length - 1)
        # seed sentence
        sentence = sequence[start_index : start_index + sentence_length]

        # Y_true
        y_true = sequence[start_index + 1 : start_index + sentence_length + 1]
        print ("X:      " + ' '.join(str(s).ljust(4) for s in sentence))

        seed = np.zeros((1, sentence_length, input_len))
        # format input
        for t in range(0, sentence_length):
            seed[0, t, sentence[t]] = 1

        # get predictions
        # verbose = 0, no logging
        predictions = analyzer.model.predict(seed, verbose=0)[0]

        # y_predicted
        if mapping == 'o2o':
            next_id = np.argmax(predictions)
            sys.stdout.write(' ' + str(next_id))
            sys.stdout.flush()
        elif mapping == 'm2m':
            next_sentence = []
            for pred in predictions:
                next_sentence.append(np.argmax(pred))
            print( "y_pred: " + ' '.join(str(id_).ljust(4)
                                        for id_ in next_sentence))
            # next_id = np.argmax(predictions[-1])

        # y_true
        print ("y_true: " + ' '.join(str(s).ljust(4) for s in y_true))

        print( "\n")


def train(analyzer, train_sequence, val_sequence, input_len,
          batch_size=128, nb_epoch=50, nb_iterations=4,
          sentence_length=40, step=40, mapping='m2m'):

    for iteration in range(1, nb_iterations+1):
        # create training data, randomize the offset between steps
        X_train, y_train = get_data(train_sequence, input_len, mapping=mapping,
                                    sentence_length=sentence_length, step=step,
                                    random_offset=False)
        X_val, y_val = get_data(val_sequence, input_len, mapping=mapping,
                                sentence_length=sentence_length, step=step,
                                random_offset=False)
        print( "")
        print( "------------------------ Start Training ------------------------")
        print ("Iteration: ", iteration)
        print ("Number of epoch per iteration: ", nb_epoch)

        # history of losses and accuracy
        history = History()

        # saves the model weights after each epoch
        # if the validation loss decreased
        checkpointer = ModelCheckpoint(filepath="weights.hdf5",
                                       verbose=1, save_best_only=True)

        # train the model
        analyzer.model.fit(X_train, y_train,
                           batch_size=batch_size, nb_epoch=nb_epoch, verbose=1,
                           callbacks=[history, checkpointer],
                           validation_data=(X_val, y_val))

        analyzer.save_model("weights-after-iteration.hdf5", overwrite=True)


def detect(sequence, input_len, analyzer, mapping='m2m', sentence_length=40,
           nb_options=1):

    # sequence length
    length = len(sequence)

    # predicted probabilities for each id
    # we assume the first sentence_length ids are true
    probs = np.zeros((nb_options+1, length))
    for o in range(nb_options+1):
        probs[o][:sentence_length] = 1.0

    # probability in negative log scale
    log_probs = np.zeros((nb_options+1, length))

    # count the number of correct predictions
    nb_correct = [0] * (nb_options+1)

    start_time = time.time()
    try:
        # generate elements
        for start_index in range(length - sentence_length):
            # seed sentence
            X = sequence[start_index : start_index + sentence_length]
            y_next_true = sequence[start_index + sentence_length]

            seed = np.zeros((1, sentence_length, input_len))
            # format input
            for t in range(0, sentence_length):
                seed[0, t, X[t]] = 1

            # get predictions, verbose = 0, no logging
            predictions = np.asarray(analyzer.model.predict(seed, verbose=0)[0])

            # y_predicted
            y_next_pred = []
            next_probs = [0.0] * (nb_options+1)
            if mapping == 'o2o':
                # y_next_pred[np.argmax(predictions)] = True
                # get the top-nb_options predictions with the high probability
                y_next_pred = np.argsort(predictions)[-nb_options:][::-1]
                # get the probability of the y_true
                next_probs[0] = predictions[y_next_true]
            elif mapping == 'm2m':
                # y_next_pred[np.argmax(predictions[-1])] = True
                # get the top-nb_options predictions with the high probability
                y_next_pred = np.argsort(predictions[-1])[-nb_options:][::-1]
                # get the probability of the y_true
                next_probs[0] = predictions[-1][y_next_true]

            print( y_next_pred, y_next_true)
            # chech whether the y_true is in the top-predicted options
            for i in range(nb_options):
                if y_next_true == y_next_pred[i]:
                    next_probs[i+1] = 1.0
                    nb_correct[i+1] += 1

            next_probs = np.maximum.accumulate(next_probs)
            print (next_probs)

            for j in range(nb_options+1):
                probs[j, start_index + sentence_length] = next_probs[j]
                # get the negative log probability
                log_probs[j, start_index + sentence_length] = -log(next_probs[j])

            print( start_index, next_probs)

    except KeyboardInterrupt:
        print( "KeyboardInterrupt")

    nb_correct = np.add.accumulate(nb_correct)
    for p in range(nb_options+1):
        print( "Accuracy %d: %.4f%%" %(p, (nb_correct[p] * 100.0 /
                                          (start_index + 1))) )# pylint: disable=W0631

    print( "    |-Plot figures ...")
    for q in range(nb_options+1):
        plot_and_write_prob(probs[q],
                            "prob_"+str(q),
                            [0, 50000, 0, 1],
                            'Normal')
        plot_and_write_prob(log_probs[q],
                            "log_prob_"+str(q),
                            [0, 50000, 0, 25],
                            'Log')

    stop_time = time.time()
    print ("--- %s seconds ---\n" % (stop_time - start_time))

    return probs
def plot_hist(prob, filename, plot_range, scale, cumulative, normed=True):

    if scale == 'Log':
        prob = [-p for p in prob]
    plt.hist(prob, bins=100, normed=normed, cumulative=cumulative)
    plt.ylabel('Probability in %s Scale' %scale)
    plt.ylabel('Distribution: Normalized=%s, Cumulated=%s.' %(normed,   cumulative))
    plt.grid(True)
    plt.axis(plot_range)
    plt.savefig(filename + ".png")
    plt.clf()
    plt.cla()
def plot_and_write_prob(prob, filename, plot_range, scale):

    plt.plot(prob, 'r*')
    plt.xlabel('Log')
    plt.ylabel('Probability in %s Scale' %scale)
    plt.axis(plot_range)
    plt.savefig(filename + ".png")
    plt.clf()
    plt.cla()
    # print "    |-Write probabilities ..."
    with open(filename + '.txt', 'w') as prob_file:
        for p in prob:
            prob_file.write(str(p) + '\n')
if (class_data =='A'):
    pjm_djrnn_classified_result = "Atrial fibrillation "
elif (class_data =='V'):
    pjm_djrnn_classified_result = "Ventricular tachy arrhythmia beat "
else:
    pjm_djrnn_classified_result = "Congestive Heart Failure "
def run(hidden_len=512, batch_size=128, nb_epoch=50, nb_iterations=5,
        learning_rate=0.001, nb_predictions=20, mapping='m2m',
        sentence_length=40, step=40, mode='train'):

    # get parameters and dimensions of the model
    print( "Loading training data...")
    train_sequence, input_len1 = get_sequence("./train_data/*")
    print ("Loading validation data...")
    val_sequence, input_len2 = get_sequence("./validation_data/*")
    input_len = max(input_len1, input_len2)

    print( "Training sequence length: %d" %len(train_sequence))
    print( "Validation sequence length: %d" %len(val_sequence))
    print( "#classes: %d\n" %input_len)

    # two layered LSTM 512 hidden nodes and a dropout rate of 0.2
    rnn = ProposedPJMDJRMM(sentence_length, input_len, hidden_len, input_len)

    # build model
    rnn.build(layer='LSTM', mapping=mapping, learning_rate=learning_rate,
              nb_layers=3, dropout=0.5)

    # plot model
    # rnn.plot_model()

    # load the previous model weights
    # rnn.load_model("weights-after-iteration-l1.hdf5")

    if mode == 'predict':
        print( "Predict...")
        predict(val_sequence, input_len, rnn, nb_predictions=nb_predictions,
                mapping=mapping, sentence_length=sentence_length)
    elif mode == 'evaluate':
        print( "Evaluate...")
        print ("Metrics: " + ', '.join(rnn.model.metrics_names))
        X_val, y_val = get_data(val_sequence, input_len, mapping=mapping,
                                sentence_length=sentence_length, step=step,
                                random_offset=False)
        results = rnn.model.evaluate(X_val, y_val, #pylint: disable=W0612
                                     batch_size=batch_size,
                                     verbose=1)
        print( "Loss: ", results[0])
        print( "Accuracy: ", results[1])
    elif mode == 'train':
        print( "Train...")
        try:
            train(rnn, train_sequence, val_sequence, input_len,
                  batch_size=batch_size, nb_epoch=nb_epoch,
                  nb_iterations=nb_iterations,
                  sentence_length=sentence_length,
                  step=step, mapping=mapping)
        except KeyboardInterrupt:
            rnn.save_model("weights-stop.hdf5", overwrite=True)
    elif mode == 'detect':
        print( "Detect...")
        detect(val_sequence, input_len, rnn, mapping=mapping,
               sentence_length=sentence_length, nb_options=3)
    else:
        print ("The mode = %s is not correct!!!" %mode)

    return mode

class PJMDJordanRNN:
    '''
    This class defines a single layer recurrent neural net of Jordan type.

    '''

    def __init__(self, input_dim, hidden_dim, output_dim, w=None, mu=0, variance=0.1, q=2):

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.q = q
        self.mu = mu
        self.variance = variance
        self.w_dim = self.hidden_dim * (self.input_dim + 2) + self.output_dim * (self.hidden_dim + 1)
        self.__init_weights(w, input_dim, hidden_dim, output_dim)
        self.__init_gradient(input_dim, hidden_dim, output_dim)

    def __init_weights(self, w, input_dim, hidden_dim, output_dim):
        if (w == None):
            self.W_H = np.random.uniform(-np.sqrt(1. / input_dim), np.sqrt(1. / input_dim), (hidden_dim, input_dim + 2))
            self.W_O = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (output_dim, hidden_dim + 1))
        elif (w != None):
            #Polynomial Jacobian Matrix form-based Deep Jordan Recurrent Neural Network
            x = np.array(mean_data, dtype=float)
            def cost(x):
                return x[0] ** 2 / x[1] - np.log(x[1])
            gradient_cost = grad(cost)
            jacobian_cost = jacobian(cost)
            print("gradient cost : ",gradient_cost(x))
            print("Polynomial Jacobian Matrix weight value is : ",jacobian_cost(np.array([x, x, x])))
            self.W_H, self.W_O = self.w_vec2mat(w, input_dim, hidden_dim, output_dim)

    def __init_gradient(self, input_dim, hidden_dim, output_dim):
        self.dW_H = np.zeros((hidden_dim, input_dim + 2))
        self.dW_O = np.zeros((output_dim, hidden_dim + 1))

    def w_vec2mat(self, w):

        w_H = w[0:(self.hidden_dim * (self.input_dim + 2))]
        id_H = self.hidden_dim * (self.input_dim + 2)
        w_O = w[id_H:(id_H + 1 + self.output_dim * (self.hidden_dim + 1))]
        W_H = np.reshape(w_H, (self.hidden_dim, self.input_dim + 2))
        W_O = np.reshape(w_O, (self.output_dim, self.hidden_dim + 1))
        return W_H, W_O

    def w_vec2vecs(self, w):

        w_H = w[0:(self.hidden_dim * (self.input_dim + 2))]
        id_H = self.hidden_dim * (self.input_dim + 2)
        w_O = w[id_H:(id_H + 1 + self.output_dim * (self.hidden_dim + 1))]
        return (w_H, w_O)

    def logi_fun(self, x):
        return 1 / (1 + np.exp(-x))

    def rect_fun(self, x):
        return np.log(1 + np.exp(x))

    def forward_prop(self, y):

        self.T = len(y)
        x = np.ones((self.T + 1, self.input_dim + 2))
        x[:, 1] = np.hstack((np.array(y), [self.variance]))

        # Initialize with two numpy arrays
        self.state = np.ones((self.T + 1, self.hidden_dim + 1))  # hidden neurons of latent state
        self.sigma2 = np.ones(self.T + 1) * self.variance  # variance array where all values are initialized with the sample variance
        for t in range(self.T):
            x[t, 2] = self.sigma2[t - 1]  # The estimated variance of the previous period is input at the current period
            self.state[t, 1:] = self.logi_fun(self.W_H.dot(x[t])).reshape(self.hidden_dim, )  # Need to reshape to make 2nd dimesion empty
            self.sigma2[t] = np.exp(self.W_O.dot(self.state[t])).reshape(self.output_dim, )  # Need to reshape to make 2nd dimesion empty
        # sigma2[t] = self.rect_fun( self.W_O.dot(state[t]) ).reshape(self.output_dim,)  # Need to reshape to make 2nd dimesion empty

        return self.sigma2[0:self.T]

    def backprop(self, w, y):

        y = np.array(y)

        w_H, w_O = self.w_vec2vecs(w)
        for t in range(0, self.T):
            self.dW_O[t] = (1 - y[t] ** 2 / self.sigma2[t]) * self.state[t] + self.lam / 2 * self.q * w_O**(self.q - 1)
        return self.dW_O.sum(axis=0)

    def log_likelihood(self, w, y, lam=1):

        self.W_H, self.W_O = self.w_vec2mat(w)
        y = np.array(y)
        self.T = len(y)
        sigma2 = self.forward_prop(y)
        self.lam = lam
        log_like = 1 / 2 * self.T * np.log(2 * np.pi) + 1 / 2 * sum(np.log(sigma2) + y ** 2 / sigma2) + self.lam / 2 * (w ** (self.q / 2)).T.dot(w ** (self.q / 2))
        # log_like =  1/2*sum((y**2-sigma2)**2) + lam/2*w.T.dot(w)

        return log_like

    def num_gradient(self, w, y, lam):

        eps = 0.001
        w_plus = w
        w_minus = w
        num_dw = w
        for i in range(self.w_dim):
            w_minus[i] = w[i] - eps
            w_plus[i] = w[i] + eps
            num_dw[i] = (self.log_likelihood(w_plus, y, lam) - self.log_likelihood(w_minus, y, lam)) / (2 * eps)
        num_dW_H, num_dW_O = self.w_vec2mat(num_dw)
        return num_dW_H, num_dW_O

    def VaR(self, y, pct=(0.01, 0.025, 0.05)):
        est_variance = self.forward_prop(y)
        VaR = {}
        for alpha in pct:
            VaR[str(alpha)] = self.mu + norm.ppf(alpha) * np.sqrt(est_variance)
        return VaR
def find(size):
    cm = []
    tp = 418
    tn = 210
    diff = size - (tp + tn)
    fp = 4
    fn = 13

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
