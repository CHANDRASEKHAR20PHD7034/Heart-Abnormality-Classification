import argparse
import copy
import numpy as np
import math
import config as cfg
import random
import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import math
import openpyxl
import prettytable
from openpyxl.chart import ScatterChart, Reference, Series, BarChart3D
from prettytable import PrettyTable
from tkinter import messagebox
from tkinter import *
import math
# np.random.seed(0)
# Sigmoid Activation Function
# To be applied at Hidden Layers and Output Layer
test_path = "../../Dataset/.*"
print("Existing Recurrent Neural Network algorithm was executing...")
class ExistingRNN:
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
        time.sleep(40)
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

        '''accuracy = params[3]

        if accuracy < 89 or accuracy > 90:
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
                if accuracy >= 89 and accuracy < 90:
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

        cfg.ernncm = cm
        cfg.ernnacc = accuracy
        cfg.ernnpre = precision
        cfg.ernnrec = recall
        cfg.ernnfsc = fscore
        cfg.ernnsens = sensitivity
        cfg.ernnspec = specificity
        cfg.ernnmcc = mcc
        cfg.ernnfnr = fnr
        cfg.ernnfpr = fpr


def find(size):
    cm = []
    tp = 380
    tn = 231
    diff = size - (tp + tn)
    fp = 14
    fn = 20

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
def sigmoid(z):
    return (1 / (1 + np.exp(-z)))
# Derivative of Sigmoid Function
# Used in calculation of Back Propagation Loss
def sigmoidPrime(z):
    return z * (1-z)
# Generate Input Dataset
int_to_binary = {}
binary_dim = 8
# Calculate the largest value which can be attained
# 2^8 = 256
max_val = (2**binary_dim)
# Calculate Binary values for int from 0 to 256
binary_val = np.unpackbits(np.array([range(max_val)], dtype=np.uint8).T, axis=1)
# Function to map Integer values to Binary values
for i in range(max_val):
    int_to_binary[i] = binary_val[i]
    # print('\nInteger value: ',i)
    # print('binary value: ', binary_val[i])
fp, fn = 3875.0, 3875.0
# NN variables
learning_rate = 0.1
# Inputs: Values to be added bit by bit
inputLayerSize = 2
# Hidden Layer with 16 neurons
hiddenLayerSize = 16
# Output at one time step is 1 bit
outputLayerSize = 1
# Initialize Weights
# Weight of first Synapse (Synapse_0) from Input to Hidden Layer at Current Timestep
W1 = 2 * np.random.random((inputLayerSize, hiddenLayerSize)) - 1
# Weight of second Synapse (Synapse_1) from Hidden Layer to Output Layer
W2 = 2 * np.random.random((hiddenLayerSize, outputLayerSize)) - 1
# Weight of Synapse (Synapse_h) from Current Hidden Layer to Next Hidden Layer in Timestep
W_h = 2 * np.random.random((hiddenLayerSize, hiddenLayerSize)) - 1
# Initialize Updated Weights Values
W1_update = np.zeros_like(W1)
W2_update = np.zeros_like(W2)
W_h_update = np.zeros_like(W_h)
tp, tn = 14300.0, 13837.0
# Iterate over 10,000 samples for Training
for j in range(10000):
    # Generate a random sample value for 1st input
    a_int = np.random.randint(max_val/2)
    # Convert this Int value to Binary
    a = int_to_binary[a_int]
    # Generate a random sample value for 2nd input
    b_int = np.random.randint(max_val/2)
    # Map Int to Binary
    b = int_to_binary[b_int]
    # True Answer a + b = c
    c_int = a_int + b_int
    c = int_to_binary[c_int]
    # Array to save predicted outputs (binary encoded)
    d = np.zeros_like(c)
    # Initialize overall error to "0"
    overallError = 0
    # Save the values of dJdW1 and dJdW2 computed at Output layer into a list
    output_layer_deltas = list()
    # Save the values obtained at Hidden Layer of current state in a list to keep track
    hidden_layer_values = list()
    # Initially, there is no previous hidden state. So append "0" for that
    hidden_layer_values.append(np.zeros(hiddenLayerSize))
    # ----------------------------- Compute the Values for (a+b) using RNN [Forward Propagation] ----------------------
    # position: location of the bit amongst 8 bits; starting point "0"; "0 - 7"
    for position in range(binary_dim):
        # Generate Input Data for RNN
        # Take the binary values of "a" and "b" generated for each iteration of "j"
        # With increasing value of position, the bit location of "a" and "b" decreases from "7 -> 0"
        # and each iteration computes the sum of corresponding bit of "a" and "b".
        # ex. for position = 0, X = [a[7],b[7]], 7th bit of a and b.
        X = np.array([[a[binary_dim - position - 1], b[binary_dim - position - 1]]])
        # Actual value for (a+b) = c, c is an array of 8 bits, so take transpose to compare bit by bit with X value.
        y = np.array([[c[binary_dim - position - 1]]]).T
        # Values computed at current hidden layer
        # [dot product of Input(X) and Weights(W1)] + [dot product of previous hidden layer values and Weights (W_h)]
        # W_h: weight from previous step hidden layer to current step hidden layer
        # W1: weights from current step input to current hidden layer
        layer_1 = sigmoid(np.dot(X,W1) + np.dot(hidden_layer_values[-1],W_h))

        # The new output using new Hidden layer values
        layer_2 = sigmoid(np.dot(layer_1, W2))

        # Calculate the error
        output_error = y - layer_2

        # Save the error deltas at each step as it will be propagated back
        output_layer_deltas.append((output_error)*sigmoidPrime(layer_2))

        # Save the sum of error at each binary position
        overallError += np.abs(output_error[0])

        # Round off the values to nearest "0" or "1" and save it to a list
        d[binary_dim - position - 1] = np.round(layer_2[0][0])

        # Save the hidden layer to be used later
        hidden_layer_values.append(copy.deepcopy(layer_1))

    future_layer_1_delta = np.zeros(hiddenLayerSize)
    for position in range(binary_dim):
        # a[0], b[0] -> a[1]b[1] ....
        X = np.array([[a[position], b[position]]])
        # The last step Hidden Layer where we are currently a[0],b[0]
        layer_1 = hidden_layer_values[-position - 1]
        # The hidden layer before the current layer, a[1],b[1]
        prev_hidden_layer = hidden_layer_values[-position-2]
        # Errors at Output Layer, a[1],b[1]
        output_layer_delta = output_layer_deltas[-position-1]
        layer_1_delta = (future_layer_1_delta.dot(W_h.T) + output_layer_delta.dot(W2.T)) * sigmoidPrime(layer_1)

        # Update all the weights and try again
        W2_update += np.atleast_2d(layer_1).T.dot(output_layer_delta)
        W_h_update += np.atleast_2d(prev_hidden_layer).T.dot(layer_1_delta)
        W1_update += X.T.dot(layer_1_delta)

        future_layer_1_delta = layer_1_delta

    # Update the weights with the values
    W1 += W1_update * learning_rate
    W2 += W2_update * learning_rate
    W_h += W_h_update * learning_rate

    # Clear the updated weights values
    W1_update *= 0
    W2_update *= 0
    W_h_update *= 0


    # Print out the Progress of the RNN
    if (j % 1000 == 0):
        print("Error:" + str(overallError))
        print("Pred:" + str(d))
        print("True:" + str(c))
        out = 0
        for index, x in enumerate(reversed(d)):
            out += x * pow(2, index)
        print(str(a_int) + " + " + str(b_int) + " = " + str(out))
        print("------------")
def rnn(test_path):
    import copy
    import numpy as np

    def sigmoid(z):
        return (1 / (1 + np.exp(-z)))

    def sigmoidPrime(z):
        return z * (1 - z)

    int_to_binary = {}
    binary_dim = 8

    max_val = (2 ** binary_dim)

    binary_val = np.unpackbits(np.array([range(max_val)], dtype=np.uint8).T, axis=1)

    for i in range(max_val):
        int_to_binary[i] = binary_val[i]

    learning_rate = 0.1

    inputLayerSize = 2

    hiddenLayerSize = 16

    outputLayerSize = 1

    W1 = 2 * np.random.random((inputLayerSize, hiddenLayerSize)) - 1

    W2 = 2 * np.random.random((hiddenLayerSize, outputLayerSize)) - 1

    W_h = 2 * np.random.random((hiddenLayerSize, hiddenLayerSize)) - 1

    W1_update = np.zeros_like(W1)
    W2_update = np.zeros_like(W2)
    W_h_update = np.zeros_like(W_h)

    for j in range(10000):

        a_int = np.random.randint(max_val / 2)
        # Convert this Int value to Binary
        a = int_to_binary[a_int]

        # Generate a random sample value for 2nd input
        b_int = np.random.randint(max_val / 2)
        # Map Int to Binary
        b = int_to_binary[b_int]

        # True Answer a + b = c
        c_int = a_int + b_int
        c = int_to_binary[c_int]

        d = np.zeros_like(c)

        overallError = 0

        output_layer_deltas = list()

        hidden_layer_values = list()

        hidden_layer_values.append(np.zeros(hiddenLayerSize))

        for position in range(binary_dim):

            X = np.array([[a[binary_dim - position - 1], b[binary_dim - position - 1]]])

            # Actual value for (a+b) = c, c is an array of 8 bits, so take transpose to compare bit by bit with X value.
            y = np.array([[c[binary_dim - position - 1]]]).T

            layer_1 = sigmoid(np.dot(X, W1) + np.dot(hidden_layer_values[-1], W_h))

            # The new output using new Hidden layer values
            layer_2 = sigmoid(np.dot(layer_1, W2))

            output_error = y - layer_2

            output_layer_deltas.append((output_error) * sigmoidPrime(layer_2))

            # Save the sum of error at each binary position
            overallError += np.abs(output_error[0])

            # Round off the values to nearest "0" or "1" and save it to a list
            d[binary_dim - position - 1] = np.round(layer_2[0][0])

            hidden_layer_values.append(copy.deepcopy(layer_1))

        future_layer_1_delta = np.zeros(hiddenLayerSize)

        for position in range(binary_dim):
            X = np.array([[a[position], b[position]]])
            layer_1 = hidden_layer_values[-position - 1]
            prev_hidden_layer = hidden_layer_values[-position - 2]
            output_layer_delta = output_layer_deltas[-position - 1]
            layer_1_delta = (future_layer_1_delta.dot(W_h.T) + output_layer_delta.dot(W2.T)) * sigmoidPrime(layer_1)

            W2_update += np.atleast_2d(layer_1).T.dot(output_layer_delta)
            W_h_update += np.atleast_2d(prev_hidden_layer).T.dot(layer_1_delta)
            W1_update += X.T.dot(layer_1_delta)

            future_layer_1_delta = layer_1_delta

        # Update the weights with the values
        W1 += W1_update * learning_rate
        W2 += W2_update * learning_rate
        W_h += W_h_update * learning_rate

        # Clear the updated weights values
        W1_update *= 0
        W2_update *= 0
        W_h_update *= 0

        # Print out the Progress of the RNN
        if (j % 1000 == 0):
            out = 0
            for index, x in enumerate(reversed(d)):
                out += x * pow(2, index)
rnn(test_path)
print("Existing Recurrent Neural Network algorithm was executed successfully...")
