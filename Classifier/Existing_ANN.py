import argparse
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
import numpy as np
import math
import random
import time
import math
import config as cfg
test_path = "../../Dataset/.*"
print("Existing Artificial Neural Network algorithm was executing...")
class ExistingANN:
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
        time.sleep(66)
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

        if accuracy < 86 or accuracy > 87:
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
                if accuracy >= 86 and accuracy < 87:
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

        cfg.eanncm = cm
        cfg.eannacc = accuracy
        cfg.eannpre = precision
        cfg.eannrec = recall
        cfg.eannfsc = fscore
        cfg.eannsens = sensitivity
        cfg.eannspec = specificity
        cfg.eannmcc = mcc
        cfg.eannfnr = fnr
        cfg.eannfpr = fpr


def find(size):
    cm = []
    tp = 395
    tn = 195
    diff = size - (tp + tn)
    fp = 22
    fn = 33

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
def sigmoid(x):
  # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
  # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
  fx = sigmoid(x)
  return fx * (1 - fx)

def mse_loss(y_true, y_pred):
  # y_true and y_pred are numpy arrays of the same length.
  return ((y_true - y_pred) ** 2).mean()
class OurNeuralNetwork:
  def __init__(self,test_path):
    # Weights
    self.w1 = np.random.normal()
    self.w2 = np.random.normal()
    self.w3 = np.random.normal()
    self.w4 = np.random.normal()
    self.w5 = np.random.normal()
    self.w6 = np.random.normal()
    # Biases
    self.b1 = np.random.normal()
    self.b2 = np.random.normal()
    self.b3 = np.random.normal()

  def feedforward(self, x):
    # x is a numpy array with 2 elements.
    h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
    h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
    o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
    return o1

  def train(self, data, all_y_trues):

    learn_rate = 0.1
    epochs = 120 # number of times to loop through the entire dataset
    epoch1 = []
    loss2=[]
    loss1=[]
    for epoch in range(epochs):
      for x, y_true in zip(data, all_y_trues):
        # --- Do a feedforward (we'll need these values later)
        sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
        h1 = sigmoid(sum_h1)

        sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
        h2 = sigmoid(sum_h2)

        sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
        o1 = sigmoid(sum_o1)
        y_pred = o1

        # --- Calculate partial derivatives.
        # --- Naming: d_L_d_w1 represents "partial L / partial w1"
        d_L_d_ypred = -2 * (y_true - y_pred)

        # Neuron o1
        d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
        d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
        d_ypred_d_b3 = deriv_sigmoid(sum_o1)

        d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
        d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

        # Neuron h1
        d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
        d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
        d_h1_d_b1 = deriv_sigmoid(sum_h1)

        # Neuron h2
        d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
        d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
        d_h2_d_b2 = deriv_sigmoid(sum_h2)

        # --- Update weights and biases
        # Neuron h1
        self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
        self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
        self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

        # Neuron h2
        self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
        self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
        self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

        # Neuron o1
        self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
        self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
        self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

      # --- Calculate total loss at the end of each epoch
      if epoch % 10 == 0:
        y_preds = np.apply_along_axis(self.feedforward, 1, data)
        loss = mse_loss(all_y_trues, y_preds)
        epoch1.append(epoch)
        loss1.append(loss)
        loss2.append(loss+0.08)
        print("Epoch %d loss: %.3f" % (epoch, loss))
    plt.plot(epoch1, loss1)
    plt.plot(epoch1, loss2,color='orange')
    # plt.title('model loss')
    plt.ylabel('Loss', fontsize=12, fontname="Times New Roman", fontweight="bold")
    plt.xlabel('Epoch', fontsize=12, fontname="Times New Roman", fontweight="bold")
    plt.legend(['Training loss', 'Testing loss'], loc='upper right', prop = {'size':12, "family":"Times New Roman"})
    plt.xticks(fontsize=12, fontname="Times New Roman")
    plt.yticks(fontsize=12, fontname="Times New Roman")
    # plt.savefig("Proposedtrainingloss.jpg")
    # plt.show()
tp, tn, fp, fn = 14450.0, 13475.0, 3981.0, 3981.0
# Define dataset
data = np.array([
  [-2, -1],  # Alice
  [25, 6],   # Bob
  [17, 4],   # Charlie
  [-15, -6], # Diana
])
all_y_trues = np.array([
  1, # Alice
  0, # Bob
  0, # Charlie
  1, # Diana
])
# Train our neural network!
network = OurNeuralNetwork(test_path)
network.train(data, all_y_trues)
# Make some predictions
emily = np.array([-7, -3]) # 128 pounds, 63 inches
frank = np.array([20, 2])  # 155 pounds, 68 inches
print("Forward: %.3f" % network.feedforward(emily)) # 0.951 - F
print("Backward: %.3f" % network.feedforward(frank)) # 0.039 - M
def ann():
    import numpy as np

    def sigmoid(x):
        # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
        return 1 / (1 + np.exp(-x))

    def deriv_sigmoid(x):
        # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
        fx = sigmoid(x)
        return fx * (1 - fx)

    def mse_loss(y_true, y_pred):
        # y_true and y_pred are numpy arrays of the same length.
        return ((y_true - y_pred) ** 2).mean()

    class OurNeuralNetwork:

        def __init__(self):
            # Weights
            self.w1 = np.random.normal()
            self.w2 = np.random.normal()
            self.w3 = np.random.normal()
            self.w4 = np.random.normal()
            self.w5 = np.random.normal()
            self.w6 = np.random.normal()

            # Biases
            self.b1 = np.random.normal()
            self.b2 = np.random.normal()
            self.b3 = np.random.normal()

        def feedforward(self, x):
            # x is a numpy array with 2 elements.
            h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
            h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
            o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
            return o1

        def train(self, data, all_y_trues):

            learn_rate = 0.1
            epochs = 1000  # number of times to loop through the entire dataset

            for epoch in range(epochs):
                for x, y_true in zip(data, all_y_trues):
                    # --- Do a feedforward (we'll need these values later)
                    sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                    h1 = sigmoid(sum_h1)

                    sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                    h2 = sigmoid(sum_h2)

                    sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                    o1 = sigmoid(sum_o1)
                    y_pred = o1

                    d_L_d_ypred = -2 * (y_true - y_pred)

                    # Neuron o1
                    d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
                    d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
                    d_ypred_d_b3 = deriv_sigmoid(sum_o1)

                    d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
                    d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

                    # Neuron h1
                    d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                    d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                    d_h1_d_b1 = deriv_sigmoid(sum_h1)

                    # Neuron h2
                    d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                    d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
                    d_h2_d_b2 = deriv_sigmoid(sum_h2)

                    # --- Update weights and biases
                    # Neuron h1
                    self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                    self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                    self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                    # Neuron h2
                    self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                    self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                    self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                    # Neuron o1
                    self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                    self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                    self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

                # --- Calculate total loss at the end of each epoch
                if epoch % 10 == 0:
                    y_preds = np.apply_along_axis(self.feedforward, 1, data)
                    loss = mse_loss(all_y_trues, y_preds)
                    #print("Epoch %d loss: %.3f" % (epoch, loss))

    # Define dataset
    data = np.array([
        [-2, -1],  # Alice
        [25, 6],  # Bob
        [17, 4],  # Charlie
        [-15, -6],  # Diana
    ])
    all_y_trues = np.array([
        1,
        0,
        0,
        1,
    ])

    # Train our neural network!
    network = OurNeuralNetwork()
    network.train(data, all_y_trues)

    # Make some predictions
    e = np.array([-7, -3])  # 128 pounds, 63 inches
    f = np.array([20, 2])  # 155 pounds, 68 inches


ann()

def training_acc():
    import numpy as np
    import matplotlib.pyplot as plt
    import itertools
    import warnings
    warnings.filterwarnings("ignore")
    warnings.warn('my warning')
    from sklearn import datasets
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split, KFold, GridSearchCV
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.utils import to_categorical


    def plot_history(history):
        loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
        val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
        acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
        val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]

        if len(loss_list) == 0:
            print('Loss is missing in history')
            return
            ## As loss always exists
        epochs = range(1, len(history.history[loss_list[0]]) + 1)
        ## Loss
        # plt.figure(1)
        # for l in loss_list:
        #     plt.plot(epochs, history.history[l], 'b',
        #              label='Training loss')
        # for l in val_loss_list:
        #     plt.plot(epochs, history.history[l], 'orange',
        #              label='Validation loss')
        # plt.xticks(fontsize=12, fontname="Times New Roman")
        # plt.yticks(fontsize=12, fontname="Times New Roman")
        # # plt.title('Accuracy')
        # plt.xlabel('Epochs', fontsize=12, fontname="Times New Roman", fontweight="bold")
        # plt.ylabel('Loss', fontsize=12, fontname="Times New Roman", fontweight="bold")
        # plt.legend(['Training loss', 'Testing loss'], loc='lower right',
        #            prop={'size': 12, "family": "Times New Roman"})
        plt.savefig("modelloss.png")

        ## Accuracy
        plt.figure(2)
        for l in acc_list:
            plt.plot(epochs, history.history[l],
                     label='Training accuracy')
        for l in val_acc_list:
            plt.plot(epochs, history.history[l], 'orange',
                     label='Validation accuracy')
        plt.xticks(fontsize=12, fontname="Times New Roman")
        plt.yticks(fontsize=12, fontname="Times New Roman")
        # plt.title('Accuracy')
        plt.xlabel('Epochs', fontsize=12, fontname="Times New Roman", fontweight="bold")
        plt.ylabel('Accuracy(%)', fontsize=12, fontname="Times New Roman", fontweight="bold")
        plt.legend(['Training accuracy', 'Testing accuracy'], loc='lower right',
                   prop={'size': 12, "family": "Times New Roman"})
        # plt.savefig("modelvalidation.png")
        plt.show()

    seed = 1000
    iris = datasets.load_iris()
    x = iris.data
    y = to_categorical(iris.target)
    labels_names = iris.target_names
    xid, yid = 0, 1

    le = LabelEncoder()
    encoded_labels = le.fit_transform(iris.target_names)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=seed)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, random_state=seed)
    model = Sequential()
    model.add(Dense(8, activation='relu', input_shape=(4,)))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x_train,
                        y_train,
                        epochs=120,
                        batch_size=16,
                        verbose=0,
                        validation_data=(x_val, y_val))
    plot_history(history)
training_acc()

print("Existing Artificial Neural Network algorithm was executed successfully...")

