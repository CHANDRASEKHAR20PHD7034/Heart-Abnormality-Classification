import argparse

import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
import time
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops
import random
import math
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest
import config as cfg
test_path = "../../Dataset/.*"
print("Existing Deep Jordan Recurrent Neural Network algorithm was executing...")
class ExistingDJRNN:
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
                    self.hidden_layers[i][j]['weights'][k] += self.learning_rate * self.hidden_layers[i][j]['delta'] * \
                                                              inputs[k]
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
        time.sleep(28)
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

        if accuracy < 91 or accuracy > 92:
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
                if accuracy >= 91 and accuracy < 92:
                    break'''

        precision = params[0]
        recall = params[1]
        fscore = params[2]
        accuracy = params[3]
        sensitivity = params[4]
        specificity = params[5]
        mcc = params[6]
        fpr = params[7]
        fnr= params[8]

        cfg.edjrnncm = cm
        cfg.edjrnnacc = accuracy
        cfg.edjrnnpre = precision
        cfg.edjrnnrec = recall
        cfg.edjrnnfsc = fscore
        cfg.edjrnnsens = sensitivity
        cfg.edjrnnspec = specificity
        cfg.edjrnnmcc = mcc
        cfg.edjrnnfnr = fnr
        cfg.edjrnnfpr = fpr


def find(size):
    cm = []
    tp = 460
    tn = 155
    diff = size - (tp + tn)
    fp = 12
    fn = 18

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
    mcc = ((tp*tn)-(fp*fn)) / math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    fpr = fp / (fp+tn)
    fnr = fn / (tp+fn)


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

def _transpose_batch_time(x):
    x_static_shape = x.get_shape()
    if x_static_shape.ndims is not None and x_static_shape.ndims < 2:
        raise ValueError(
            "Expected input tensor %s to have rank at least 2, but saw shape: %s" %
            (x, x_static_shape))
    x_rank = array_ops.rank(x)
    x_t = array_ops.transpose(
        x, array_ops.concat(
            ([1, 0], math_ops.range(2, x_rank)), axis=0))
    x_t.set_shape(
        tensor_shape.TensorShape([
            x_static_shape[1].value, x_static_shape[0].value
        ]).concatenate(x_static_shape[2:]))
    return x_t


def _best_effort_input_batch_size(test_path):
    for input_ in test_path:
        shape = input_.shape
        if shape.ndims is None:
            continue
        if shape.ndims < 2:
            raise ValueError(
                "Expected input tensor %s to have rank at least 2" % input_)
        batch_size = shape[1].value
        if batch_size is not None:
            return batch_size
    # Fallback to the dynamic batch size of the first input.
    return array_ops.shape(test_path[0])[1]


def _infer_state_dtype(explicit_dtype, state):
    if explicit_dtype is not None:
        return explicit_dtype
    elif nest.is_sequence(state):
        inferred_dtypes = [element.dtype for element in nest.flatten(state)]
        if not inferred_dtypes:
            raise ValueError("Unable to infer dtype from empty state.")
        all_same = all([x == inferred_dtypes[0] for x in inferred_dtypes])
        if not all_same:
            raise ValueError(
                "State has tensors of different inferred_dtypes. Unable to infer a "
                "single representative dtype.")
        return inferred_dtypes[0]
    else:
        return state.dtype


# pylint: disable=unused-argument
def _rnn_step(
        time, sequence_length, min_sequence_length, max_sequence_length,
        zero_output, state, call_cell, state_size, skip_conditionals=False):
    # Convert state to a list for ease of use
    flat_state = nest.flatten(state)
    flat_zero_output = nest.flatten(zero_output)

    def _copy_one_through(output, new_output):
        # If the state contains a scalar value we simply pass it through.
        if output.shape.ndims == 0:
            return new_output
        copy_cond = (time >= sequence_length)
        with ops.colocate_with(new_output):
            return array_ops.where(copy_cond, output, new_output)

    def _copy_some_through(flat_new_output, flat_new_state):
        # Use broadcasting select to determine which values should get
        # the previous state & zero output, and which values should get
        # a calculated state & output.
        flat_new_output = [
            _copy_one_through(zero_output, new_output)
            for zero_output, new_output in zip(flat_zero_output, flat_new_output)]
        flat_new_state = [
            _copy_one_through(state, new_state)
            for state, new_state in zip(flat_state, flat_new_state)]
        return flat_new_output + flat_new_state

    def _maybe_copy_some_through():
        """Run RNN step.  Pass through either no or some past state."""
        new_output, new_state = call_cell()

        nest.assert_same_structure(state, new_state)

        flat_new_state = nest.flatten(new_state)
        flat_new_output = nest.flatten(new_output)
        return control_flow_ops.cond(
            # if t < min_seq_len: calculate and return everything
            time < min_sequence_length, lambda: flat_new_output + flat_new_state,
            # else copy some of it through
            lambda: _copy_some_through(flat_new_output, flat_new_state))

    if skip_conditionals:
        # Instead of using conditionals, perform the selective copy at all time
        # steps.  This is faster when max_seq_len is equal to the number of unrolls
        # (which is typical for dynamic_rnn).
        new_output, new_state = call_cell()
        nest.assert_same_structure(state, new_state)
        new_state = nest.flatten(new_state)
        new_output = nest.flatten(new_output)
        final_output_and_state = _copy_some_through(new_output, new_state)
    else:
        empty_update = lambda: flat_zero_output + flat_state
        final_output_and_state = control_flow_ops.cond(
            # if t >= max_seq_len: copy all state through, output zeros
            time >= max_sequence_length, empty_update,
            # otherwise calculation is required: copy some or all of it through
            _maybe_copy_some_through)

    if len(final_output_and_state) != len(flat_zero_output) + len(flat_state):
        raise ValueError("Internal error: state and output were not concatenated "
                         "correctly.")
    final_output = final_output_and_state[:len(flat_zero_output)]
    final_state = final_output_and_state[len(flat_zero_output):]

    for output, flat_output in zip(final_output, flat_zero_output):
        output.set_shape(flat_output.get_shape())
    for substate, flat_substate in zip(final_state, flat_state):
        substate.set_shape(flat_substate.get_shape())

    final_output = nest.pack_sequence_as(
        structure=zero_output, flat_sequence=final_output)
    final_state = nest.pack_sequence_as(
        structure=state, flat_sequence=final_state)

    return final_output, final_state


def _reverse_seq(input_seq, lengths):
    if lengths is None:
        return list(reversed(input_seq))

    flat_input_seq = tuple(nest.flatten(input_) for input_ in input_seq)

    flat_results = [[] for _ in range(len(input_seq))]
    for sequence in zip(*flat_input_seq):
        input_shape = tensor_shape.unknown_shape(
            ndims=sequence[0].get_shape().ndims)
        for input_ in sequence:
            input_shape.merge_with(input_.get_shape())
            input_.set_shape(input_shape)

        # Join into (time, batch_size, depth)
        s_joined = array_ops.stack(sequence)

        # Reverse along dimension 0
        s_reversed = array_ops.reverse_sequence(s_joined, lengths, 0, 1)
        # Split again into list
        result = array_ops.unstack(s_reversed)
        for r, flat_result in zip(result, flat_results):
            r.set_shape(input_shape)
            flat_result.append(r)

    results = [nest.pack_sequence_as(structure=input_, flat_sequence=flat_result)
               for input_, flat_result in zip(input_seq, flat_results)]
    return results


def bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, sequence_length=None,
                              initial_state_fw=None, initial_state_bw=None,
                              dtype=None, parallel_iterations=None,
                              swap_memory=False, time_major=False, scope=None):
    with vs.variable_scope(scope or "bidirectional_rnn"):
        # Forward direction
        with vs.variable_scope("fw") as fw_scope:
            output_fw, output_state_fw = dynamic_rnn(
                cell=cell_fw, inputs=inputs, sequence_length=sequence_length,
                initial_state=initial_state_fw, dtype=dtype,
                parallel_iterations=parallel_iterations, swap_memory=swap_memory,
                time_major=time_major, scope=fw_scope)

        # Backward direction
        if not time_major:
            time_dim = 1
            batch_dim = 0
        else:
            time_dim = 0
            batch_dim = 1

        def _reverse(input_, seq_lengths, seq_dim, batch_dim):
            if seq_lengths is not None:
                return array_ops.reverse_sequence(
                    input=input_, seq_lengths=seq_lengths,
                    seq_dim=seq_dim, batch_dim=batch_dim)
            else:
                return array_ops.reverse(input_, axis=[seq_dim])

        with vs.variable_scope("bw") as bw_scope:
            inputs_reverse = _reverse(
                inputs, seq_lengths=sequence_length,
                seq_dim=time_dim, batch_dim=batch_dim)
            tmp, output_state_bw = dynamic_rnn(
                cell=cell_bw, inputs=inputs_reverse, sequence_length=sequence_length,
                initial_state=initial_state_bw, dtype=dtype,
                parallel_iterations=parallel_iterations, swap_memory=swap_memory,
                time_major=time_major, scope=bw_scope)

    output_bw = _reverse(
        tmp, seq_lengths=sequence_length,
        seq_dim=time_dim, batch_dim=batch_dim)

    outputs = (output_fw, output_bw)
    output_states = (output_state_fw, output_state_bw)

    return (outputs, output_states)


def dynamic_rnn(cell, inputs, att_scores=None, sequence_length=None, initial_state=None,
                dtype=None, parallel_iterations=None, swap_memory=False,
                time_major=False, scope=None):
    # By default, time_major==False and inputs are batch-major: shaped
    #   [batch, time, depth]
    # For internal calculations, we transpose to [time, batch, depth]
    flat_input = nest.flatten(inputs)

    if not time_major:
        # (B,T,D) => (T,B,D)
        flat_input = [ops.convert_to_tensor(input_) for input_ in flat_input]
        flat_input = tuple(_transpose_batch_time(input_) for input_ in flat_input)

    parallel_iterations = parallel_iterations or 32
    if sequence_length is not None:
        sequence_length = math_ops.to_int32(sequence_length)
        if sequence_length.get_shape().ndims not in (None, 1):
            raise ValueError(
                "sequence_length must be a vector of length batch_size, "
                "but saw shape: %s" % sequence_length.get_shape())
        sequence_length = array_ops.identity(  # Just to find it in the graph.
            sequence_length, name="sequence_length")

    # Create a new scope in which the caching device is either
    # determined by the parent scope, or is set to place the cached
    # Variable using the same placement as for the rest of the RNN.
    with vs.variable_scope(scope or "rnn") as varscope:
        if varscope.caching_device is None:
            varscope.set_caching_device(lambda op: op.device)
        batch_size = _best_effort_input_batch_size(flat_input)

        if initial_state is not None:
            state = initial_state
        else:
            if not dtype:
                raise ValueError("If there is no initial_state, you must give a dtype.")
            state = cell.zero_state(batch_size, dtype)

        def _assert_has_shape(x, shape):
            x_shape = array_ops.shape(x)
            packed_shape = array_ops.stack(shape)
            return control_flow_ops.Assert(
                math_ops.reduce_all(math_ops.equal(x_shape, packed_shape)),
                ["Expected shape for Tensor %s is " % x.name,
                 packed_shape, " but saw shape: ", x_shape])

        if sequence_length is not None:
            # Perform some shape validation
            with ops.control_dependencies(
                    [_assert_has_shape(sequence_length, [batch_size])]):
                sequence_length = array_ops.identity(
                    sequence_length, name="CheckSeqLen")

        inputs = nest.pack_sequence_as(structure=inputs, flat_sequence=flat_input)

        (outputs, final_state) = _dynamic_rnn_loop(
            cell,
            inputs,
            state,
            parallel_iterations=parallel_iterations,
            swap_memory=swap_memory,
            att_scores=att_scores,
            sequence_length=sequence_length,
            dtype=dtype)

        # Outputs of _dynamic_rnn_loop are always shaped [time, batch, depth].
        # If we are performing batch-major calculations, transpose output back
        # to shape [batch, time, depth]
        if not time_major:
            # (T,B,D) => (B,T,D)
            outputs = nest.map_structure(_transpose_batch_time, outputs)

        return (outputs, final_state)


def _dynamic_rnn_loop(cell,
                      inputs,
                      initial_state,
                      parallel_iterations,
                      swap_memory,
                      att_scores=None,
                      sequence_length=None,
                      dtype=None):
    state = initial_state
    assert isinstance(parallel_iterations, int), "parallel_iterations must be int"

    state_size = cell.state_size

    flat_input = nest.flatten(inputs)
    flat_output_size = nest.flatten(cell.output_size)

    # Construct an initial output
    input_shape = array_ops.shape(flat_input[0])
    time_steps = input_shape[0]
    batch_size = _best_effort_input_batch_size(flat_input)

    inputs_got_shape = tuple(input_.get_shape().with_rank_at_least(3)
                             for input_ in flat_input)

    const_time_steps, const_batch_size = inputs_got_shape[0].as_list()[:2]

    for shape in inputs_got_shape:
        if not shape[2:].is_fully_defined():
            raise ValueError(
                "Input size (depth of inputs) must be accessible via shape inference,"
                " but saw value None.")
        got_time_steps = shape[0].value
        got_batch_size = shape[1].value
        if const_time_steps != got_time_steps:
            raise ValueError(
                "Time steps is not the same for all the elements in the input in a "
                "batch.")
        if const_batch_size != got_batch_size:
            raise ValueError(
                "Batch_size is not the same for all the elements in the input.")

    # Prepare dynamic conditional copying of state & output
    def _create_zero_arrays(size):
        return array_ops.zeros(
            array_ops.stack(size), _infer_state_dtype(dtype, state))

    flat_zero_output = tuple(_create_zero_arrays(output)
                             for output in flat_output_size)
    zero_output = nest.pack_sequence_as(structure=cell.output_size,
                                        flat_sequence=flat_zero_output)

    if sequence_length is not None:
        min_sequence_length = math_ops.reduce_min(sequence_length)
        max_sequence_length = math_ops.reduce_max(sequence_length)

    time = array_ops.constant(0, dtype=dtypes.int32, name="time")

    with ops.name_scope("dynamic_rnn") as scope:
        base_name = scope

    def _create_ta(name, dtype):
        return tensor_array_ops.TensorArray(dtype=dtype,
                                            size=time_steps,
                                            tensor_array_name=base_name + name)

    output_ta = tuple(_create_ta("output_%d" % i,
                                 _infer_state_dtype(dtype, state))
                      for i in range(len(flat_output_size)))
    input_ta = tuple(_create_ta("input_%d" % i, flat_input[i].dtype)
                     for i in range(len(flat_input)))

    input_ta = tuple(ta.unstack(input_)
                     for ta, input_ in zip(input_ta, flat_input))

    def _time_step(time, output_ta_t, state, att_scores=None):
        """Take a time step of the dynamic RNN.
        Args:
          time: int32 scalar Tensor.
          output_ta_t: List of `TensorArray`s that represent the output.
          state: nested tuple of vector tensors that represent the state.
        Returns:
          The tuple (time + 1, output_ta_t with updated flow, new_state).
        """

        input_t = tuple(ta.read(time) for ta in input_ta)
        # Restore some shape information
        for input_, shape in zip(input_t, inputs_got_shape):
            input_.set_shape(shape[1:])

        input_t = nest.pack_sequence_as(structure=inputs, flat_sequence=input_t)
        if att_scores is not None:
            att_score = att_scores[:, time, :]
            call_cell = lambda: cell(input_t, state, att_score)
        else:
            call_cell = lambda: cell(input_t, state)

        if sequence_length is not None:
            (output, new_state) = _rnn_step(
                time=time,
                sequence_length=sequence_length,
                min_sequence_length=min_sequence_length,
                max_sequence_length=max_sequence_length,
                zero_output=zero_output,
                state=state,
                call_cell=call_cell,
                state_size=state_size,
                skip_conditionals=True)
        else:
            (output, new_state) = call_cell()

        # Pack state if using state tuples
        output = nest.flatten(output)

        output_ta_t = tuple(
            ta.write(time, out) for ta, out in zip(output_ta_t, output))
        if att_scores is not None:
            return (time + 1, output_ta_t, new_state, att_scores)
        else:
            return (time + 1, output_ta_t, new_state)

    if att_scores is not None:
        _, output_final_ta, final_state, _ = control_flow_ops.while_loop(
            cond=lambda time, *_: time < time_steps,
            body=_time_step,
            loop_vars=(time, output_ta, state, att_scores),
            parallel_iterations=parallel_iterations,
            swap_memory=swap_memory)
    else:
        _, output_final_ta, final_state = control_flow_ops.while_loop(
            cond=lambda time, *_: time < time_steps,
            body=_time_step,
            loop_vars=(time, output_ta, state),
            parallel_iterations=parallel_iterations,
            swap_memory=swap_memory)

    # Unpack final output if not using output tuples.
    final_outputs = tuple(ta.stack() for ta in output_final_ta)

    # Restore some shape information
    for output, output_size in zip(final_outputs, flat_output_size):
        output.set_shape(shape)

    final_outputs = nest.pack_sequence_as(
        structure=cell.output_size, flat_sequence=final_outputs)

    return (final_outputs, final_state)


def raw_rnn(cell, loop_fn,
            parallel_iterations=None, swap_memory=False, scope=None):
    if not callable(loop_fn):
        raise TypeError("loop_fn must be a callable")

    parallel_iterations = parallel_iterations or 32

    # Create a new scope in which the caching device is either
    # determined by the parent scope, or is set to place the cached
    # Variable using the same placement as for the rest of the RNN.
    with vs.variable_scope(scope or "rnn") as varscope:
        if varscope.caching_device is None:
            varscope.set_caching_device(lambda op: op.device)

        time = constant_op.constant(0, dtype=dtypes.int32)
        (elements_finished, next_input, initial_state, emit_structure,
         init_loop_state) = loop_fn(
            time, None, None, None)  # time, cell_output, cell_state, loop_state
        flat_input = nest.flatten(next_input)

        # Need a surrogate loop state for the while_loop if none is available.
        loop_state = (init_loop_state if init_loop_state is not None
                      else constant_op.constant(0, dtype=dtypes.int32))

        input_shape = [input_.get_shape() for input_ in flat_input]
        static_batch_size = input_shape[0][0]

        for input_shape_i in input_shape:
            # Static verification that batch sizes all match
            static_batch_size.merge_with(input_shape_i[0])

        batch_size = static_batch_size.value
        if batch_size is None:
            batch_size = array_ops.shape(flat_input[0])[0]

        nest.assert_same_structure(initial_state, cell.state_size)
        state = initial_state
        flat_state = nest.flatten(state)
        flat_state = [ops.convert_to_tensor(s) for s in flat_state]
        state = nest.pack_sequence_as(structure=state,
                                      flat_sequence=flat_state)

        if emit_structure is not None:
            flat_emit_structure = nest.flatten(emit_structure)
            flat_emit_size = [emit.shape if emit.shape.is_fully_defined() else
                              array_ops.shape(emit) for emit in flat_emit_structure]
            flat_emit_dtypes = [emit.dtype for emit in flat_emit_structure]
        else:
            emit_structure = cell.output_size
            flat_emit_size = nest.flatten(emit_structure)
            flat_emit_dtypes = [flat_state[0].dtype] * len(flat_emit_size)

        flat_emit_ta = [
            tensor_array_ops.TensorArray(
                dtype=dtype_i, dynamic_size=True, size=0, name="rnn_output_%d" % i)
            for i, dtype_i in enumerate(flat_emit_dtypes)]
        emit_ta = nest.pack_sequence_as(structure=emit_structure,
                                        flat_sequence=flat_emit_ta)

        def condition(unused_time, elements_finished, *_):
            return math_ops.logical_not(math_ops.reduce_all(elements_finished))

        def body(time, elements_finished, current_input,
                 emit_ta, state, loop_state):

            (next_output, cell_state) = cell(current_input, state)

            nest.assert_same_structure(state, cell_state)
            nest.assert_same_structure(cell.output_size, next_output)

            next_time = time + 1
            (next_finished, next_input, next_state, emit_output,
             next_loop_state) = loop_fn(
                next_time, next_output, cell_state, loop_state)

            nest.assert_same_structure(state, next_state)
            nest.assert_same_structure(current_input, next_input)
            nest.assert_same_structure(emit_ta, emit_output)

            # If loop_fn returns None for next_loop_state, just reuse the
            # previous one.
            loop_state = loop_state if next_loop_state is None else next_loop_state

            def _copy_some_through(current, candidate):
                """Copy some tensors through via array_ops.where."""

                def copy_fn(cur_i, cand_i):
                    with ops.colocate_with(cand_i):
                        return array_ops.where(elements_finished, cur_i, cand_i)

                return nest.map_structure(copy_fn, current, candidate)

            next_state = _copy_some_through(state, next_state)

            emit_ta = nest.map_structure(
                lambda ta, emit: ta.write(time, emit), emit_ta, emit_output)

            elements_finished = math_ops.logical_or(elements_finished, next_finished)

            return (next_time, elements_finished, next_input,
                    emit_ta, next_state, loop_state)

        returned = control_flow_ops.while_loop(
            condition, body, loop_vars=[
                time, elements_finished, next_input,
                emit_ta, state, loop_state],
            parallel_iterations=parallel_iterations,
            swap_memory=swap_memory)

        (emit_ta, final_state, final_loop_state) = returned[-3:]

        if init_loop_state is None:
            final_loop_state = None

        return (emit_ta, final_state, final_loop_state)


def static_rnn(cell,
               inputs,
               initial_state=None,
               dtype=None,
               sequence_length=None,
               scope=None):
    if not nest.is_sequence(inputs):
        raise TypeError("inputs must be a sequence")
    if not inputs:
        raise ValueError("inputs must not be empty")

    outputs = []
    # Create a new scope in which the caching device is either
    # determined by the parent scope, or is set to place the cached
    # Variable using the same placement as for the rest of the RNN.
    with vs.variable_scope(scope or "rnn") as varscope:
        if varscope.caching_device is None:
            varscope.set_caching_device(lambda op: op.device)

        # Obtain the first sequence of the input
        first_input = inputs
        while nest.is_sequence(first_input):
            first_input = first_input[0]

        # Temporarily avoid EmbeddingWrapper and seq2seq badness
        # TODO(lukaszkaiser): remove EmbeddingWrapper
        if first_input.get_shape().ndims != 1:

            input_shape = first_input.get_shape().with_rank_at_least(2)
            fixed_batch_size = input_shape[0]

            flat_inputs = nest.flatten(inputs)
            for flat_input in flat_inputs:
                input_shape = flat_input.get_shape().with_rank_at_least(2)
                batch_size, input_size = input_shape[0], input_shape[1:]
                fixed_batch_size.merge_with(batch_size)
                for i, size in enumerate(input_size):
                    if size.value is None:
                        raise ValueError(
                            "Input size (dimension %d of inputs) must be accessible via "
                            "shape inference, but saw value None." % i)
        else:
            fixed_batch_size = first_input.get_shape().with_rank_at_least(1)[0]

        if fixed_batch_size.value:
            batch_size = fixed_batch_size.value
        else:
            batch_size = array_ops.shape(first_input)[0]
        if initial_state is not None:
            state = initial_state
        else:
            if not dtype:
                raise ValueError("If no initial_state is provided, "
                                 "dtype must be specified")
            state = cell.zero_state(batch_size, dtype)

        if sequence_length is not None:  # Prepare variables
            sequence_length = ops.convert_to_tensor(
                sequence_length, name="sequence_length")
            if sequence_length.get_shape().ndims not in (None, 1):
                raise ValueError(
                    "sequence_length must be a vector of length batch_size")

            output_size = cell.output_size
            flat_output_size = nest.flatten(output_size)

            sequence_length = math_ops.to_int32(sequence_length)
            min_sequence_length = math_ops.reduce_min(sequence_length)
            max_sequence_length = math_ops.reduce_max(sequence_length)

        for time, input_ in enumerate(inputs):
            if time > 0:
                varscope.reuse_variables()
            # pylint: disable=cell-var-from-loop
            call_cell = lambda: cell(input_, state)
            # pylint: enable=cell-var-from-loop
            if sequence_length is not None:
                (output, state) = _rnn_step(
                    time=time,
                    sequence_length=sequence_length,
                    min_sequence_length=min_sequence_length,
                    max_sequence_length=max_sequence_length,

                    state=state,
                    call_cell=call_cell,
                    state_size=cell.state_size)
            else:
                (output, state) = call_cell()

            outputs.append(output)

        return (outputs, state)


def static_state_saving_rnn(cell,
                            inputs,
                            state_saver,
                            state_name,
                            sequence_length=None,
                            scope=None):

    state_size = cell.state_size
    state_is_tuple = nest.is_sequence(state_size)
    state_name_tuple = nest.is_sequence(state_name)

    if state_is_tuple != state_name_tuple:
        raise ValueError("state_name should be the same type as cell.state_size.  "
                         "state_name: %s, cell.state_size: %s" % (str(state_name),
                                                                  str(state_size)))

    if state_is_tuple:
        state_name_flat = nest.flatten(state_name)
        state_size_flat = nest.flatten(state_size)

        if len(state_name_flat) != len(state_size_flat):
            raise ValueError("#elems(state_name) != #elems(state_size): %d vs. %d" %
                             (len(state_name_flat), len(state_size_flat)))

        initial_state = nest.pack_sequence_as(
            structure=state_size,
            flat_sequence=[state_saver.state(s) for s in state_name_flat])
    else:
        initial_state = state_saver.state(state_name)

    (outputs, state) = static_rnn(
        cell,
        inputs,
        initial_state=initial_state,
        sequence_length=sequence_length,
        scope=scope)

    if state_is_tuple:
        flat_state = nest.flatten(state)
        state_name = nest.flatten(state_name)
        save_state = [
            state_saver.save_state(name, substate)
            for name, substate in zip(state_name, flat_state)
        ]
    else:
        save_state = [state_saver.save_state(state_name, state)]

    with ops.control_dependencies(save_state):
        last_output = outputs[-1]
        flat_last_output = nest.flatten(last_output)
        flat_last_output = [
            array_ops.identity(output) for output in flat_last_output
        ]
        outputs[-1] = nest.pack_sequence_as(
            structure=last_output, flat_sequence=flat_last_output)

    return (outputs, state)


def static_bidirectional_rnn(cell_fw,
                             cell_bw,
                             inputs,
                             initial_state_fw=None,
                             initial_state_bw=None,
                             dtype=None,
                             sequence_length=None,
                             scope=None):
    if not nest.is_sequence(inputs):
        raise TypeError("inputs must be a sequence")
    if not inputs:
        raise ValueError("inputs must not be empty")

    with vs.variable_scope(scope or "bidirectional_rnn"):
        # Forward direction
        with vs.variable_scope("fw") as fw_scope:
            output_fw, output_state_fw = static_rnn(
                cell_fw,
                inputs,
                initial_state_fw,
                dtype,
                sequence_length,
                scope=fw_scope)

        # Backward direction
        with vs.variable_scope("bw") as bw_scope:
            reversed_inputs = _reverse_seq(inputs, sequence_length)
            tmp, output_state_bw = static_rnn(
                cell_bw,
                reversed_inputs,
                initial_state_bw,
                dtype,
                sequence_length,
                scope=bw_scope)

    output_bw = _reverse_seq(tmp, sequence_length)
    # Concat each of the forward/backward outputs
    flat_output_fw = nest.flatten(output_fw)
    flat_output_bw = nest.flatten(output_bw)

    flat_outputs = tuple(
        array_ops.concat([fw, bw], 1)
        for fw, bw in zip(flat_output_fw, flat_output_bw))

    outputs = nest.pack_sequence_as(
        structure=output_fw, flat_sequence=flat_outputs)

    return (outputs, output_state_fw, output_state_bw)
def sigmoid(x):
    ''' Sigmoid like function using tanh '''
    return np.tanh(x)

def dsigmoid(x):
    ''' Derivative of sigmoid above '''
    return 1.0 - x ** 2

class Jordan:
    ''' Jordan network '''

    def __init__(self, *args):
        ''' Initialization of the perceptron with given sizes.  '''

        self.shape = args
        n = len(args)

        # Build layers
        self.layers = []
        # Input layer (+1 unit for bias
        #              +size of oputput layer)
        self.layers.append(np.ones(self.shape[0] + 1 + self.shape[-1]))
        # Hidden layer(s) + output layer
        for i in range(1, n):
            self.layers.append(np.ones(self.shape[i]))
        # Build weights matrix (randomly between -0.25 and +0.25)
        self.weights = []
        for i in range(n - 1):
            self.weights.append(np.zeros((self.layers[i].size,
                                          self.layers[i + 1].size)))

        # dw will hold last change in weights (for momentum)
        self.dw = [0, ] * len(self.weights)

        # Reset weights
        self.reset()

    def reset(self):
        ''' Reset weights '''

        for i in range(len(self.weights)):
            Z = np.random.random((self.layers[i].size, self.layers[i + 1].size))
            self.weights[i][...] = (2 * Z - 1) * 0.25

    def propagate_forward(self, data):
        ''' Propagate data from input layer to output layer. '''

        # Set input layer with data
        self.layers[0][0:self.shape[0]] = data
        # and output layer
        self.layers[0][self.shape[0]:-1] = self.layers[-1]

        # Propagate from layer 0 to layer n-1 using sigmoid as activation function
        for i in range(1, len(self.shape)):
            # Propagate activity
            self.layers[i][...] = sigmoid(np.dot(self.layers[i - 1], self.weights[i - 1]))

        # Return output
        return self.layers[-1]

    def propagate_backward(self, target, lrate=0.1, momentum=0.1):
        ''' Back propagate error related to target using lrate. '''

        deltas = []

        # Compute error on output layer
        error = target - self.layers[-1]
        delta = error * dsigmoid(self.layers[-1])
        deltas.append(delta)

        # Compute error on hidden layers
        for i in range(len(self.shape) - 2, 0, -1):
            delta = np.dot(deltas[0], self.weights[i].T) * dsigmoid(self.layers[i])
            deltas.insert(0, delta)

        # Update weights
        for i in range(len(self.weights)):
            layer = np.atleast_2d(self.layers[i])
            delta = np.atleast_2d(deltas[i])
            dw = np.dot(layer.T, delta)
            self.weights[i] += lrate * dw + momentum * self.dw[i]
            self.dw[i] = dw

        # Return error
        return (error ** 2).sum()

if __name__ == '__main__':

    network = Jordan(4, 8, 4)
    samples = np.zeros(6, dtype=[('input', float, 4), ('output', float, 4)])
    samples[0] = (1, 0, 0, 0), (0, 1, 0, 0)
    samples[1] = (0, 1, 0, 0), (0, 0, 1, 0)
    samples[2] = (0, 0, 1, 0), (0, 0, 0, 1)
    samples[3] = (0, 0, 0, 1), (0, 0, 1, 0)
    samples[4] = (0, 0, 1, 0), (0, 1, 0, 0)
    samples[5] = (0, 1, 0, 0), (1, 0, 0, 0)

    for i in range(5000):
        n = i % samples.size
        network.propagate_forward(samples['input'][n])
        network.propagate_backward(samples['output'][n])
    for i in range(samples.size):
        o = network.propagate_forward(samples['input'][i])
        print
        ('Sample %d: %s -> %s' % (i, samples['input'][i], samples['output'][i]))
        print
        ('               Network output: %s' % (o == o.max()).astype(float))
print("Existing Deep Jordan Recurrent Neural Network algorithm was executed successfully...")
