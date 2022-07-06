"""Pytorch Multi-layer Perceptron
"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


def make_confusion_matrix(
    cf,
    group_names=None,
    categories="auto",
    count=True,
    percent=True,
    cbar=True,
    xyticks=True,
    xyplotlabels=True,
    sum_stats=True,
    figsize=None,
    cmap="Blues",
    title=None,
):
    """
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html

    title:         Title for the heatmap. Default is None.
    """

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ["" for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = [
            "{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)
        ]
    else:
        group_percentages = blanks

    box_labels = [
        f"{v1}{v2}{v3}".strip()
        for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)
    ]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))

        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score
            )
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize is None:
        # Get default figure size if not set
        figsize = plt.rcParams.get("figure.figsize")

    if xyticks == False:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(
        cf,
        annot=box_labels,
        fmt="",
        cmap=cmap,
        cbar=cbar,
        xticklabels=categories,
        yticklabels=categories,
    )

    if xyplotlabels:
        plt.ylabel("True label")
        plt.xlabel("Predicted label" + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)


def load_model(
    model_class,
    path="",
    param_file_name="param",
):

    with open(f"{path}/{param_file_name}.pickle", "rb") as handle:
        param = pickle.load(handle)
    model = model_class.from_config(param["_init_param"])
    model.set_parameter(param)
    model.load_model(path)
    return model


class PytorchMLPClassifier:

    """Pytorch Multi-layer Perceptron Classifier.
    Parameters
    ----------
    standard
        input_dim : int
        classes : list or array of class
        hidden_layer_sizes : tuple
        activation_function : list of string with shape (n_hidden_layers) or {None, 'sigmoid', 'tanh', 'relu', 'prelu'}, default='relu'
        learning_rate_init : float, default=0.01
        epochs : int, default=100

    optional
        progress_bar : bool, default=True
        validation_split : float, default=0
        class_weight : {None, 'balanced'}, default=None

    improvement
        use_gpu : bool, default=False
        solver : {'sgd', 'adam'}, default='sgd'
        batch_size : int, default=n_samples
        shuffle : bool, default=True
        beta_1 : float, default=0.9
        beta_2 : float, default=0.999
        l1_lambda : list of float with shape (n_hidden_layers + 1) or float, default=0
        l2_lambda : list of float with shape (n_hidden_layers + 1) or float, default=0
        dropout_rate : list of float with shape (n_hidden_layers) or float, default=0

    Methods
    ----------
    get_parameter()
    get_weights()
    fit(X,y)
    predict(X)
    predict_proba(X)

    """

    def __init__(
        self,
        input_dim,
        classes,
        hidden_layer_sizes,
        activation_function="relu",
        learning_rate_init=0.01,
        epochs=100,
        progress_bar=True,
        validation_split=0,
        class_weight=None,
        use_gpu=False,
        solver="sgd",
        batch_size=None,
        shuffle=True,
        beta_1=0.9,
        beta_2=0.999,
        l1_lambda=0,
        l2_lambda=0,
        dropout_rate=0,
    ):
        self._init_param = locals()
        del self._init_param["self"]

        if type(use_gpu) != bool:
            raise Exception("Invalid use_gpu")
        self._use_gpu = use_gpu

        # setting optimizer
        self._setting_gpu()

        # parameter validation
        if input_dim == 0:
            raise Exception("Invalid input_dim")
        self._input_dim = input_dim

        if type(classes) != np.ndarray and type(classes) != list:
            raise Exception("Invalid classes")
        classes = np.array(classes)
        self._classes = classes
        self._number_of_class = classes.shape[0]

        if hidden_layer_sizes == 0:
            raise Exception("Invalid hidden_layer_sizes")
        self._hidden_layer_sizes = hidden_layer_sizes
        self._number_of_hidden_layer = len(hidden_layer_sizes)

        if activation_function is None or type(activation_function) == str:
            activation_function = [
                activation_function for i in range(self._number_of_hidden_layer)
            ]
        if self._number_of_hidden_layer != len(activation_function):
            raise Exception("Invalid activation_function")
        use_activation_function = [True for i in activation_function]
        for i, function in enumerate(activation_function):
            if function not in {None, "sigmoid", "tanh", "relu", "prelu"}:
                raise Exception("Invalid solver")
            if function is None:
                use_activation_function[i] = False
            elif function == "relu":
                activation_function[i] = nn.ReLU()
            elif function == "prelu":
                activation_function[i] = nn.PReLU()
            elif function == "sigmoid":
                activation_function[i] = nn.Sigmoid()
            elif function == "tanh":
                activation_function[i] = nn.Tanh()
            else:
                raise Exception("Invalid Activation Function")
        self._use_activation_function = use_activation_function
        self._activation_function = activation_function

        if self._number_of_hidden_layer != len(activation_function):
            raise Exception("Invalid activation_function")

        if type(learning_rate_init) != int and type(learning_rate_init) != float:
            raise Exception("Invalid learning_rate_init")
        self._learning_rate_init = learning_rate_init

        if type(epochs) != int and type(epochs) != float:
            raise Exception("Invalid epochs")
        self._epochs = epochs

        if type(progress_bar) != bool:
            raise Exception("Invalid progress_bar")
        self._progress_bar = progress_bar

        if (
            type(validation_split) != int
            and type(validation_split) != float
            and validation_split < 0
            and validation_split >= 1
        ):
            raise Exception("Invalid validation_split")
        self._validation_split = validation_split

        if class_weight is None:
            self._use_class_weight = False
        elif class_weight == "balanced":
            self._use_class_weight = True
        else:
            raise Exception("Invalid class_weight")

        if solver not in {"sgd", "adam"}:
            raise Exception("Invalid solver")
        if solver == "adam":
            if type(beta_1) != int and type(beta_1) != float:
                raise Exception("Invalid beta_1")
            if type(beta_2) != int and type(beta_2) != float:
                raise Exception("Invalid beta_2")
        self._solver = solver
        self._beta_1 = beta_1
        self._beta_2 = beta_2

        if (
            batch_size is not None
            and type(batch_size) != int
            and type(batch_size) != float
        ):
            raise Exception("Invalid batch_size")
        self._batch_size = batch_size

        if type(shuffle) != bool:
            raise Exception("Invalid shuffle")
        self._shuffle = shuffle

        if type(l1_lambda) == int or type(l1_lambda) == float:
            l1_lambda = [l1_lambda for i in range(self._number_of_hidden_layer + 1)]
        if type(l1_lambda) == tuple:
            l1_lambda = list(l1_lambda)
        if type(l1_lambda) == list:
            if self._number_of_hidden_layer + 1 != len(l1_lambda):
                raise Exception("Invalid l1_lambda")
            for lambda_ in l1_lambda:
                if lambda_ < 0:
                    raise Exception("l1_lambda must be greater than 0")
        else:
            raise Exception("Invalid l1_lambda")
        self._l1_lambda = l1_lambda

        if type(l2_lambda) == int or type(l2_lambda) == float:
            l2_lambda = [l2_lambda for i in range(self._number_of_hidden_layer + 1)]
        if type(l2_lambda) == tuple:
            l2_lambda = list(l2_lambda)
        if type(l2_lambda) == list:
            if self._number_of_hidden_layer + 1 != len(l2_lambda):
                raise Exception("Invalid l2_lambda")
            for lambda_ in l2_lambda:
                if lambda_ < 0:
                    raise Exception("l2_lambda must be greater than 0")
        else:
            raise Exception("Invalid l2_lambda")
        self._l2_lambda = l2_lambda

        if type(dropout_rate) == int or type(dropout_rate) == float:
            dropout_rate = [dropout_rate for i in range(self._number_of_hidden_layer)]
        if self._number_of_hidden_layer != len(dropout_rate):
            raise Exception("Invalid dropout rate")
        self._dropout_rate = dropout_rate

        # create model
        self._create_model()

    def _create_model(
        self,
    ):

        layers = []

        # first layer
        layers.append(nn.Linear(self._input_dim, self._hidden_layer_sizes[0]))
        if self._use_activation_function[0]:
            layers.append(self._activation_function[0])

        # hidden layer
        for i, number_of_node in enumerate(self._hidden_layer_sizes[1:], start=1):
            if self._dropout_rate[i - i] != 0:
                layers.append(nn.Dropout(p=self._dropout_rate[i - 1]))
            layers.append(nn.Linear(self._hidden_layer_sizes[i - 1], number_of_node))
            if self._use_activation_function[i]:
                layers.append(self._activation_function[i])

        # output layer
        if self._dropout_rate[-1] != 0:
            layers.append(nn.Dropout(p=self._dropout_rate[-1]))
        layers.append(nn.Linear(self._hidden_layer_sizes[-1], self._number_of_class))

        self._linear_layers = [
            layer for layer in layers if isinstance(layer, nn.Linear)
        ]

        model = nn.Sequential(*layers).to(self._device)

        self._model = model

    def _setting_gpu(
        self,
    ):
        self._device = torch.device(
            "cuda:0" if self._use_gpu and torch.cuda.is_available() else "cpu"
        )

    def _create_loss(
        self,
    ):
        self._loss_function = nn.CrossEntropyLoss(weight=self._class_weight)

    def _create_optimizer(
        self,
    ):
        if self._solver == "sgd":
            self._optimizer = optim.SGD(
                self._model.parameters(),
                lr=self._learning_rate_init,
            )
        elif self._solver == "adam":
            self._optimizer = optim.Adam(
                self._model.parameters(),
                lr=self._learning_rate_init,
                betas=(self._beta_1, self._beta_2),
            )

    def _torch_data(self, X, y):
        return (self._torch_data_X(X), self._torch_data_y(y))

    def _torch_data_X(self, X):
        return torch.from_numpy(np.array(X)).float().to(self._device)

    def _torch_data_y(self, y):
        return torch.from_numpy(np.array(y)).long().to(self._device)

    def fit(self, X, y):

        if self._use_class_weight:
            class_weight = compute_class_weight("balanced", self._classes, y)
            class_weight = torch.from_numpy(class_weight).float().to(self._device)
        else:
            class_weight = None

        self._class_weight = class_weight

        self._target_label_encoder = LabelEncoder()

        y = self._target_label_encoder.fit_transform(np.array(y).reshape(-1, 1))

        if self._validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self._validation_split, shuffle=True
            )
            X_val_torch, y_val_torch = self._torch_data(X_val, y_val)
        else:
            X_train = X
            y_train = y

        X_train_torch, y_train_torch = self._torch_data(X_train, y_train)

        if self._batch_size is None:
            batch_size = X_train_torch.shape[0]
        else:
            batch_size = self._batch_size

        # create loss
        self._create_loss()

        # create loss
        self._create_optimizer()

        model = self._model
        optimizer = self._optimizer
        loss_function = self._loss_function
        l1_lambda = self._l1_lambda
        l2_lambda = self._l2_lambda
        linear_layers = self._linear_layers

        loss_history = []
        val_loss_history = []
        batch_range = range(0, X_train_torch.shape[0], batch_size)

        for epoch in tqdm(range(self._epochs)):

            # TRAINING
            model.train()
            train_epoch_loss = 0
            validation_epoch_loss = 0

            data_train = TensorDataset(X_train_torch, y_train_torch)
            data_train_batch = DataLoader(
                data_train, batch_size=batch_size, shuffle=self._shuffle
            )

            for batch_x, batch_y in data_train_batch:

                optimizer.zero_grad()

                batch_y_pred = model(batch_x)

                loss = loss_function(batch_y_pred, batch_y)

                if np.array(l1_lambda).sum() == 0 and np.array(l2_lambda).sum() == 0:
                    loss_with_reg = loss
                else:
                    l1_norm = 0
                    l2_norm = 0
                    for i, layer in enumerate(linear_layers):
                        for params in layer.parameters():
                            l1_norm += l1_lambda[i] * sum(p.abs().sum() for p in params)
                            l2_norm += l2_lambda[i] * sum(
                                p.pow(2).sum() for p in params
                            )
                    loss_with_reg = loss + l1_norm + l2_norm

                loss_with_reg.backward()
                optimizer.step()

                training_loss = loss.item()
                train_epoch_loss += training_loss

            mean_loss = train_epoch_loss / len(batch_range)
            loss_history.append(mean_loss)

            # VALIDATION
            if self._validation_split > 0:
                with torch.no_grad():

                    model.eval()

                    y_val_pred = model(X_val_torch)

                    val_loss = loss_function(y_val_pred, y_val_torch)

                    val_epoch_loss = val_loss.item()
                    val_loss_history.append(val_epoch_loss)

        self._model = model
        self._optimizer = optimizer
        self._loss_function = loss_function

        self.loss_curve_ = loss_history
        if self._validation_split > 0:
            self.val_loss_curve_ = val_loss_history

        self.coefs_ = [
            parameter.cpu().detach().numpy().T
            for name, parameter in self._model.named_parameters()
            if "weight" in name
        ]
        self.intercepts_ = [
            parameter.cpu().detach().numpy()
            for name, parameter in self._model.named_parameters()
            if "bias" in name
        ]

        return {
            "loss": loss_history,
            "val_loss": val_loss_history,
        }

    def get_weights(
        self,
    ):
        return [param for name, param in self._model.named_parameters()]

    def predict(self, X):
        X_torch = self._torch_data_X(X)
        model = self._model
        with torch.no_grad():
            model.eval()
            y_pred_prob = model(X_torch)
            y_pred_softmax = torch.log_softmax(y_pred_prob, dim=1)
            _, y_pred = torch.max(y_pred_softmax, dim=1)
            y_pred = y_pred.cpu()
            y_pred = self._target_label_encoder.inverse_transform(y_pred)
            return y_pred

    def predict_proba(self, X):
        X_torch = self._torch_data_X(X)
        model = self._model
        with torch.no_grad():
            model.eval()
            y_pred_prob = model(X_torch)
            y_pred_softmax = torch.log_softmax(y_pred_prob, dim=1)
            return y_pred_softmax.cpu()

    def get_parameter(
        self,
    ):
        return vars(self)

    def set_parameter(self, initial_data):
        for key in initial_data:
            setattr(self, key, initial_data[key])

    def save_model(
        self,
        path="",
        param_file_name="param",
    ):
        param = self.get_parameter().copy()
        del param["_model"]
        del param["_optimizer"]
        filename = f"{path}/{param_file_name}.pickle"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as handle:
            pickle.dump(param, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return torch.save(
            {
                "epoch": self._epochs,
                "model_state_dict": self._model.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "loss": self._loss_function,
            },
            os.path.join(path, "model.pth"),
        )

    def load_model(
        self,
        path="",
    ):
        # create loss
        self._create_loss()

        # create loss
        self._create_optimizer()

        checkpoint = torch.load(os.path.join(path, "model.pth"))
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._epochs = checkpoint["epoch"]
        self._loss_function = checkpoint["loss"]
        self._model.eval()

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class PytorchMLPRegressor:

    """Pytorch Multi-layer Perceptron Regressor.
    Parameters
    ----------
    standard
        input_dim : int
        hidden_layer_sizes : tuple
        activation_function : list of string with shape (n_hidden_layers) or {None, 'sigmoid', 'tanh', 'relu', 'prelu'}, default='relu'
        learning_rate_init : float, default=0.01
        epochs : int, default=100

    optional
        progress_bar : bool, default=True
        validation_split : float, default=0

    improvement
        use_gpu : bool, default=False
        solver : {'sgd', 'adam'}, default='sgd'
        batch_size : int, default=n_samples
        shuffle : bool, default=True
        beta_1 : float, default=0.9
        beta_2 : float, default=0.999
        l1_lambda : list of float with shape (n_hidden_layers + 1) or float, default=0
        l2_lambda : list of float with shape (n_hidden_layers + 1) or float, default=0
        dropout_rate : list of float with shape (n_hidden_layers) or float, default=0

    Methods
    ----------
    get_parameter()
    get_weights()
    fit(X,y)
    predict(X)

    """

    def __init__(
        self,
        input_dim,
        hidden_layer_sizes,
        activation_function="relu",
        learning_rate_init=0.01,
        epochs=100,
        progress_bar=True,
        validation_split=0,
        use_gpu=False,
        solver="sgd",
        batch_size=None,
        shuffle=True,
        beta_1=0.9,
        beta_2=0.999,
        l1_lambda=0,
        l2_lambda=0,
        dropout_rate=0,
    ):
        self._init_param = locals()
        del self._init_param["self"]

        if type(use_gpu) != bool:
            raise Exception("Invalid use_gpu")
        self._use_gpu = use_gpu

        # setting optimizer
        self._setting_gpu()

        # parameter validation
        if input_dim == 0:
            raise Exception("Invalid input_dim")
        self._input_dim = input_dim

        if hidden_layer_sizes == 0:
            raise Exception("Invalid hidden_layer_sizes")
        self._hidden_layer_sizes = hidden_layer_sizes
        self._number_of_hidden_layer = len(hidden_layer_sizes)

        if activation_function is None or type(activation_function) == str:
            activation_function = [
                activation_function for i in range(self._number_of_hidden_layer)
            ]
        if self._number_of_hidden_layer != len(activation_function):
            raise Exception("Invalid activation_function")
        use_activation_function = [True for i in activation_function]
        for i, function in enumerate(activation_function):
            if function not in {None, "sigmoid", "tanh", "relu", "prelu"}:
                raise Exception("Invalid solver")
            if function is None:
                use_activation_function[i] = False
            elif function == "relu":
                activation_function[i] = nn.ReLU()
            elif function == "prelu":
                activation_function[i] = nn.PReLU()
            elif function == "sigmoid":
                activation_function[i] = nn.Sigmoid()
            elif function == "tanh":
                activation_function[i] = nn.Tanh()
            else:
                raise Exception("Invalid Activation Function")
        self._use_activation_function = use_activation_function
        self._activation_function = activation_function

        if self._number_of_hidden_layer != len(activation_function):
            raise Exception("Invalid activation_function")

        if type(learning_rate_init) != int and type(learning_rate_init) != float:
            raise Exception("Invalid learning_rate_init")
        self._learning_rate_init = learning_rate_init

        if type(epochs) != int and type(epochs) != float:
            raise Exception("Invalid epochs")
        self._epochs = epochs

        if type(progress_bar) != bool:
            raise Exception("Invalid progress_bar")
        self._progress_bar = progress_bar

        if (
            type(validation_split) != int
            and type(validation_split) != float
            and validation_split < 0
            and validation_split >= 1
        ):
            raise Exception("Invalid validation_split")
        self._validation_split = validation_split

        if solver not in {"sgd", "adam"}:
            raise Exception("Invalid solver")
        if solver == "adam":
            if type(beta_1) != int and type(beta_1) != float:
                raise Exception("Invalid beta_1")
            if type(beta_2) != int and type(beta_2) != float:
                raise Exception("Invalid beta_2")
        self._solver = solver
        self._beta_1 = beta_1
        self._beta_2 = beta_2

        if (
            batch_size is not None
            and type(batch_size) != int
            and type(batch_size) != float
        ):
            raise Exception("Invalid batch_size")
        self._batch_size = batch_size

        if type(shuffle) != bool:
            raise Exception("Invalid shuffle")
        self._shuffle = shuffle

        if type(l1_lambda) == int or type(l1_lambda) == float:
            l1_lambda = [l1_lambda for i in range(self._number_of_hidden_layer + 1)]
        if type(l1_lambda) == tuple:
            l1_lambda = list(l1_lambda)
        if type(l1_lambda) == list:
            if self._number_of_hidden_layer + 1 != len(l1_lambda):
                raise Exception("Invalid l1_lambda")
            for lambda_ in l1_lambda:
                if lambda_ < 0:
                    raise Exception("l1_lambda must be greater than 0")
        else:
            raise Exception("Invalid l1_lambda")
        self._l1_lambda = l1_lambda

        if type(l2_lambda) == int or type(l2_lambda) == float:
            l2_lambda = [l2_lambda for i in range(self._number_of_hidden_layer + 1)]
        if type(l2_lambda) == tuple:
            l2_lambda = list(l2_lambda)
        if type(l2_lambda) == list:
            if self._number_of_hidden_layer + 1 != len(l2_lambda):
                raise Exception("Invalid l2_lambda")
            for lambda_ in l2_lambda:
                if lambda_ < 0:
                    raise Exception("l2_lambda must be greater than 0")
        else:
            raise Exception("Invalid l2_lambda")
        self._l2_lambda = l2_lambda

        if type(dropout_rate) == int or type(dropout_rate) == float:
            dropout_rate = [dropout_rate for i in range(self._number_of_hidden_layer)]
        if self._number_of_hidden_layer != len(dropout_rate):
            raise Exception("Invalid dropout rate")
        self._dropout_rate = dropout_rate

        # create model
        self._create_model()

    def _create_model(
        self,
    ):

        layers = []

        # first layer
        layers.append(nn.Linear(self._input_dim, self._hidden_layer_sizes[0]))
        if self._use_activation_function[0]:
            layers.append(self._activation_function[0])

        # hidden layer
        for i, number_of_node in enumerate(self._hidden_layer_sizes[1:], start=1):
            if self._dropout_rate[i - i] != 0:
                layers.append(nn.Dropout(p=self._dropout_rate[i - 1]))
            layers.append(nn.Linear(self._hidden_layer_sizes[i - 1], number_of_node))
            if self._use_activation_function[i]:
                layers.append(self._activation_function[i])

        # output layer
        if self._dropout_rate[-1] != 0:
            layers.append(nn.Dropout(p=self._dropout_rate[-1]))
        layers.append(nn.Linear(self._hidden_layer_sizes[-1], 1))

        self._linear_layers = [
            layer for layer in layers if isinstance(layer, nn.Linear)
        ]

        model = nn.Sequential(*layers).to(self._device)

        self._model = model

    def _setting_gpu(
        self,
    ):
        self._device = torch.device(
            "cuda:0" if self._use_gpu and torch.cuda.is_available() else "cpu"
        )

    def _create_loss(
        self,
    ):
        self._loss_function = nn.MSELoss()

    def _create_optimizer(
        self,
    ):
        if self._solver == "sgd":
            self._optimizer = optim.SGD(
                self._model.parameters(),
                lr=self._learning_rate_init,
            )
        elif self._solver == "adam":
            self._optimizer = optim.Adam(
                self._model.parameters(),
                lr=self._learning_rate_init,
                betas=(self._beta_1, self._beta_2),
            )

    def _torch_data(self, X, y):
        return (self._torch_data_X(X), self._torch_data_y(y))

    def _torch_data_X(self, X):
        return torch.from_numpy(np.array(X)).float().to(self._device)

    def _torch_data_y(self, y):
        return torch.from_numpy(np.array(y)).float().to(self._device)

    def fit(self, X, y):
        # create loss
        self._create_loss()

        # create loss
        self._create_optimizer()

        if self._validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self._validation_split, shuffle=True
            )
            X_val_torch, y_val_torch = self._torch_data(X_val, y_val)
        else:
            X_train = X
            y_train = y

        X_train_torch, y_train_torch = self._torch_data(X_train, y_train)

        if self._batch_size is None:
            batch_size = X_train_torch.shape[0]
        else:
            batch_size = self._batch_size

        model = self._model
        optimizer = self._optimizer
        loss_function = self._loss_function
        l1_lambda = self._l1_lambda
        l2_lambda = self._l2_lambda
        linear_layers = self._linear_layers

        loss_history = []
        val_loss_history = []
        batch_range = range(0, X_train_torch.shape[0], batch_size)

        for epoch in tqdm(range(self._epochs)):

            # TRAINING
            model.train()
            train_epoch_loss = 0
            validation_epoch_loss = 0

            data_train = TensorDataset(X_train_torch, y_train_torch)
            data_train_batch = DataLoader(
                data_train, batch_size=batch_size, shuffle=self._shuffle
            )

            for batch_x, batch_y in data_train_batch:

                optimizer.zero_grad()

                batch_y_pred = model(batch_x)
                batch_y_pred = batch_y_pred.reshape(-1)

                loss = loss_function(batch_y_pred, batch_y)

                if np.array(l1_lambda).sum() == 0 and np.array(l2_lambda).sum() == 0:
                    loss_with_reg = loss
                else:
                    l1_norm = 0
                    l2_norm = 0
                    for i, layer in enumerate(linear_layers):
                        for params in layer.parameters():
                            l1_norm += l1_lambda[i] * sum(p.abs().sum() for p in params)
                            l2_norm += l2_lambda[i] * sum(
                                p.pow(2).sum() for p in params
                            )
                    loss_with_reg = loss + l1_norm + l2_norm

                loss_with_reg.backward()
                optimizer.step()

                training_loss = loss.item()
                train_epoch_loss += training_loss

            mean_loss = train_epoch_loss / len(batch_range)
            loss_history.append(mean_loss)

            # VALIDATION
            if self._validation_split > 0:
                with torch.no_grad():

                    model.eval()

                    y_val_pred = model(X_val_torch)
                    y_val_pred = y_val_pred.reshape(-1)

                    val_loss = loss_function(y_val_pred, y_val_torch)

                    val_epoch_loss = val_loss.item()
                    val_loss_history.append(val_epoch_loss)

        self._model = model
        self._optimizer = optimizer
        self._loss_function = loss_function

        self.loss_curve_ = loss_history
        if self._validation_split > 0:
            self.val_loss_curve_ = val_loss_history

        self.coefs_ = [
            parameter.cpu().detach().numpy().T
            for name, parameter in self._model.named_parameters()
            if "weight" in name
        ]
        self.intercepts_ = [
            parameter.cpu().detach().numpy()
            for name, parameter in self._model.named_parameters()
            if "bias" in name
        ]

        return {
            "loss": loss_history,
            "val_loss": val_loss_history,
        }

    def get_weights(
        self,
    ):
        return [param for name, param in self._model.named_parameters()]

    def predict(self, X):
        X_torch = self._torch_data_X(X)
        model = self._model
        with torch.no_grad():
            model.eval()
            y_pred = model(X_torch)
            return y_pred.cpu()

    def get_parameter(
        self,
    ):
        return vars(self)

    def set_parameter(self, initial_data):
        for key in initial_data:
            setattr(self, key, initial_data[key])

    def save_model(
        self,
        path="",
        param_file_name="param",
    ):
        param = self.get_parameter().copy()
        del param["_model"]
        del param["_optimizer"]
        filename = f"{path}/{param_file_name}.pickle"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as handle:
            pickle.dump(param, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return torch.save(
            {
                "epoch": self._epochs,
                "model_state_dict": self._model.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "loss": self._loss_function,
            },
            os.path.join(path, "model.pth"),
        )

    def load_model(
        self,
        path="",
    ):
        # create loss
        self._create_loss()

        # create loss
        self._create_optimizer()

        checkpoint = torch.load(os.path.join(path, "model.pth"))
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._epochs = checkpoint["epoch"]
        self._loss_function = checkpoint["loss"]
        self._model.eval()

    @classmethod
    def from_config(cls, config):
        return cls(**config)