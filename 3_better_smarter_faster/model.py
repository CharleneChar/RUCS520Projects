import matplotlib.pyplot as plt
import numpy as np
from copy import *


class NeuralNetwork:
    def __init__(self, kwargs):
        if kwargs['train_mode']:
            self.params = {}
            self.learning_rate = kwargs['learning_rate']
            self.mse_threshold = kwargs['mse_threshold']
            self.mse = float('Inf')
            # used for plotting
            self.stats = {'train_loss_value': [], 'val_loss_value': []}
        else:
            self.params = deepcopy(kwargs['params'])
        self.ith_layer_nodes = deepcopy(kwargs['ith_layer_nodes'])
        self.input_x = None
        self.output_y = None

    def init_params(self):
        np.random.seed(1)
        for i in range(1, len(self.ith_layer_nodes)):
            weight = f'w{i}'
            bias = f'b{i}'
            self.params[weight] = np.random.rand(self.ith_layer_nodes[i - 1], self.ith_layer_nodes[i]) - 0.5
            self.params[bias] = np.random.rand(self.ith_layer_nodes[i], 1) - 0.5

    @staticmethod
    def activation_func(input_x, func_name='relu'):
        if func_name == 'relu':
            return np.maximum(0, input_x)

    @staticmethod
    def loss_func(predicted_y, y, sample_size, func_type='mse'):
        if func_type == 'mse':
            return np.sum(np.square(predicted_y - y)) / sample_size

    def forward_propagation(self, input_x, is_val_mode=False):
        a = input_x
        for i in range(1, len(self.ith_layer_nodes)):
            if i == 1:
                k = np.dot(self.params[f'w{i}'].T, input_x) + \
                                       self.params[f'b{i}']
            else:
                k = np.dot(self.params[f'w{i}'].T, a) + \
                                       self.params[f'b{i}']
            a = self.activation_func(k, func_name='relu')
            if not is_val_mode:
                self.params[f'k{i}'] = k
                self.params[f'a{i}'] = a
        return a

    def backward_propagation(self, predicted_y,  y, sample_size):
        temp_params = {}
        for i in range(len(self.ith_layer_nodes) - 1, 0, -1):
            if i == len(self.ith_layer_nodes) - 1:
                y = y.values.reshape((1, y.shape[0]))
                temp_params[f'da{i}'] = (2 / sample_size) * (predicted_y - y)
            else:
                temp_params[f'da{i}'] = np.dot(self.params[f'w{i + 1}'], temp_params[f'dk{i + 1}'])
            temp_params[f'dk{i}'] = temp_params[f'da{i}'] * \
                                    np.where(self.params[f'k{i}'] > 0, 1, 0)
            if i != 1:
                temp_params[f'dw{i}'] = np.dot(self.params[f'a{i - 1}'], temp_params[f'dk{i}'].T)
            else:
                temp_params[f'dw{i}'] = np.dot(self.input_x, temp_params[f'dk{i}'].T)
            temp_params[f'db{i}'] = np.sum(temp_params[f'dk{i}'], axis=1, keepdims=True)
        # update weights and biases
        for i in range(len(self.ith_layer_nodes) - 1, 0, -1):
            self.params[f'w{i}'] -= self.learning_rate * temp_params[f'dw{i}']
            self.params[f'b{i}'] -= self.learning_rate * temp_params[f'db{i}']

    def fit(self, train_input_x, train_output_y, val_input_x, val_output_y):
        self.input_x = train_input_x
        self.output_y = train_output_y
        train_sample_size = train_input_x.shape[1]
        self.init_params()
        iter_count = 0
        while self.mse > self.mse_threshold:
            iter_count += 1
            train_predicted_y = self.forward_propagation(self.input_x)
            train_loss = self.loss_func(train_predicted_y.flatten(), self.output_y, train_sample_size)
            self.mse = train_loss
            self.backward_propagation(train_predicted_y, self.output_y, train_sample_size)
            if val_input_x is not None and val_output_y is not None:
                val_sample_size = val_input_x.shape[1]
                val_predicted_y = self.forward_propagation(val_input_x, is_val_mode=True)
                val_loss = self.loss_func(val_predicted_y.flatten(), val_output_y, val_sample_size)
                if iter_count % 100 == 0:
                    self.stats['val_loss_value'].append(val_loss)
            if iter_count % 100 == 0:
                self.stats['train_loss_value'].append(train_loss)
                print(f'{iter_count}th iterate, with mse: {self.mse}')

    def predict(self, input_x):
        predicted_y = self.forward_propagation(input_x)
        return predicted_y

    def plot_loss(self):
        # plot graph to display statistics
        plt.figure(figsize=(8, 6), dpi=100)
        colors = ['palevioletred', 'steelblue', 'teal']
        for i, loss_stats in enumerate(self.stats):
            plt.plot(self.stats[loss_stats],
                     label=f'{loss_stats}', color=colors[i])
        plt.xlabel('(i x 100)th iter')
        # plt.ylim((-5, 5))
        # plt.yticks(np.arange(-5, 6, 1))
        plt.ylabel('loss')
        plt.title('Loss History under NN')
        plt.legend()
        plt.savefig('loss_value_neural_network', bbox_inches='tight')
        # plt.show()


class LinearRegression:
    def __init__(self, kwargs):
        if kwargs['train_mode']:
            self.params = {}
            self.learning_rate = kwargs['learning_rate']
            self.mse_threshold = kwargs['mse_threshold']
            self.mse = float('Inf')
            # used for plotting
            self.stats = {'train_loss_value': [], 'val_loss_value': []}
        else:
            self.params = deepcopy(kwargs['params'])
        self.input_x = None
        self.output_y = None

    def init_params(self):
        np.random.seed(1)
        self.params[f'w'] = np.random.rand(self.input_x.shape[0], 1)
        self.params[f'b'] = np.random.rand(1, 1)

    @staticmethod
    def loss_func(predicted_y, y, sample_size, func_type='mse'):
        if func_type == 'mse':
            return np.sum(np.square(predicted_y - y)) / sample_size

    def forward_propagation(self, input_x):
        predicted_y = self.params[f'k'] = np.dot(self.params[f'w'].T, input_x) + self.params[f'b']
        return predicted_y

    def backward_propagation(self, predicted_y, y, sample_size):
        y = y.values.reshape((1, y.shape[0]))
        temp_params = {f'dk': (2 / sample_size) * (predicted_y - y)}
        temp_params[f'dw'] = np.dot(self.input_x, temp_params[f'dk'].T)
        temp_params[f'db'] = np.sum(temp_params[f'dk'], axis=1, keepdims=True)
        # update weights and biases
        self.params[f'w'] -= self.learning_rate * temp_params[f'dw']
        self.params[f'b'] -= self.learning_rate * temp_params[f'db']

    def fit(self, train_input_x, train_output_y, val_input_x, val_output_y):
        self.input_x = train_input_x
        self.output_y = train_output_y
        train_sample_size = train_input_x.shape[1]
        self.init_params()
        iter_count = 0
        while self.mse > self.mse_threshold:
            iter_count += 1
            train_predicted_y = self.forward_propagation(self.input_x)
            train_loss = self.loss_func(train_predicted_y.flatten(), self.output_y, train_sample_size)
            self.mse = train_loss
            self.backward_propagation(train_predicted_y, self.output_y, train_sample_size)
            if val_input_x is not None and val_output_y is not None:
                val_sample_size = val_input_x.shape[1]
                val_predicted_y = self.forward_propagation(val_input_x)
                val_loss = self.loss_func(val_predicted_y.flatten(), val_output_y, val_sample_size)
                if iter_count % 100 == 0:
                    self.stats['val_loss_value'].append(val_loss)
            if iter_count % 100 == 0:
                self.stats['train_loss_value'].append(train_loss)
                print(f'{iter_count}th iterate, with mse: {self.mse}')

    def predict(self, input_x):
        predicted_y = self.forward_propagation(input_x)
        return predicted_y

    def plot_loss(self):
        # plot graph to display statistics
        plt.figure(figsize=(8, 6), dpi=100)
        colors = ['palevioletred', 'steelblue', 'teal']
        for i, loss_stats in enumerate(self.stats):
            plt.plot(self.stats[loss_stats],
                     label=f'{loss_stats}', color=colors[i])
        plt.xlabel('(i x 100)th iter')
        # plt.ylim((-5, 5))
        # plt.yticks(np.arange(-5, 6, 1))
        plt.ylabel('loss')
        plt.title('Loss History under LinearReg')
        plt.legend()
        plt.savefig('loss_value_linear_reg', bbox_inches='tight')
        # plt.show()
