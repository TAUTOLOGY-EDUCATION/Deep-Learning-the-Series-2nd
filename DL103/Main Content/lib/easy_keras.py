"""Keras Multi-layer Perceptron
"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Dropout, PReLU
from tensorflow.keras.losses import CategoricalCrossentropy, MeanSquaredError
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow_addons.callbacks import TQDMProgressBar


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


class KerasMLPClassifier:

    """Keras Multi-layer Perceptron Classifier.
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
        for i, function in enumerate(activation_function):
            if function not in {None, "sigmoid", "tanh", "relu", "prelu"}:
                raise Exception("Invalid solver")
            if function == "prelu":
                activation_function[i] = PReLU()
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

        if type(use_gpu) != bool:
            raise Exception("Invalid use_gpu")
        self._use_gpu = use_gpu

        if solver not in {"sgd", "adam"}:
            raise Exception("Invalid solver")

        if solver == "sgd":
            self._optimizer = SGD(
                learning_rate=self._learning_rate_init,
            )
        if solver == "adam":
            if type(beta_1) != int and type(beta_1) != float:
                raise Exception("Invalid beta_1")
            if type(beta_2) != int and type(beta_2) != float:
                raise Exception("Invalid beta_2")
            self._optimizer = Adam(
                beta_1=beta_1,
                beta_2=beta_2,
                learning_rate=self._learning_rate_init,
            )

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

        # setting optimizer
        self._setting_gpu()

    def _create_model(
        self,
    ):

        model = Sequential()

        # first hidden layer
        model.add(
            Dense(
                self._hidden_layer_sizes[0],
                input_dim=self._input_dim,
                activation=self._activation_function[0],
                kernel_regularizer=regularizers.l1_l2(
                    l1=self._l1_lambda[0], l2=self._l2_lambda[0]
                ),
                bias_regularizer=regularizers.l1_l2(
                    l1=self._l1_lambda[0], l2=self._l2_lambda[0]
                ),
            )
        )

        # hidden layer
        for i, number_of_node in enumerate(self._hidden_layer_sizes[1:], start=1):
            if self._dropout_rate[i - i] != 0:
                model.add(Dropout(self._dropout_rate[i - 1]))
            model.add(
                Dense(
                    number_of_node,
                    activation=self._activation_function[i],
                    kernel_regularizer=regularizers.l1_l2(
                        l1=self._l1_lambda[i], l2=self._l2_lambda[i]
                    ),
                    bias_regularizer=regularizers.l1_l2(
                        l1=self._l1_lambda[i], l2=self._l2_lambda[i]
                    ),
                )
            )

        # output layer
        if self._dropout_rate[-1] != 0:
            model.add(Dropout(self._dropout_rate[-1]))
        model.add(
            Dense(
                self._number_of_class,
                activation="softmax",
                kernel_regularizer=regularizers.l1_l2(
                    l1=self._l1_lambda[-1], l2=self._l2_lambda[-1]
                ),
                bias_regularizer=regularizers.l1_l2(
                    l1=self._l1_lambda[-1], l2=self._l2_lambda[-1]
                ),
            )
        )

        model.compile(
            loss=CategoricalCrossentropy(),
            optimizer=self._optimizer,
        )

        self._model = model

    def _setting_gpu(
        self,
    ):
        device_type = [
            device.device_type for device in tf.config.list_physical_devices()
        ]

        if self._use_gpu and "GPU" in device_type:
            self._device_name = "/device:GPU:0"
        else:
            self._device_name = "/device:CPU:0"

    def fit(self, X, y):
        # initialize tqdm callback with default parameters
        if self._progress_bar:
            callback = [TQDMProgressBar(show_epoch_progress=False)]
        else:
            callback = None

        if self._use_class_weight:
            class_weight = compute_class_weight("balanced", self._classes, y)
            class_weight = dict(enumerate(class_weight.flatten()))
        else:
            class_weight = None

        self._target_onehot_encoder = OneHotEncoder(
            sparse=False, handle_unknown="ignore"
        )

        y = self._target_onehot_encoder.fit_transform(np.array(y).reshape(-1, 1))

        if self._batch_size is None:
            batch_size = X.shape[0]
        else:
            batch_size = self._batch_size

        # fit the keras model on the dataset
        with tf.device(self._device_name):
            history = self._model.fit(
                X,
                y,
                shuffle=self._shuffle,
                batch_size=batch_size,
                epochs=self._epochs,
                verbose=0,
                callbacks=callback,
                validation_split=self._validation_split,
            ).history
            self.loss_curve_ = history["loss"]
            if self._validation_split > 0:
                self.val_loss_curve_ = history["val_loss"]

            weights = self.get_weights()
            self.coefs_ = [weight for weight in weights if len(weight.shape) == 2]
            self.intercepts_ = [weight for weight in weights if len(weight.shape) == 1]

            return history

    def get_weights(
        self,
    ):
        return self._model.get_weights()

    def predict(self, X):
        y_pred = self._model.predict(X)
        y_pred = self._target_onehot_encoder.inverse_transform(y_pred)
        return y_pred

    def predict_proba(self, X):
        y_pred_proba = self._model.predict(X)
        return y_pred_proba

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
        return self._model.save(path)

    def load_model(
        self,
        path="",
    ):
        self._model = keras.models.load_model(path)

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class KerasMLPRegressor:

    """Keras Multi-layer Perceptron Regressor.
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
        for i, function in enumerate(activation_function):
            if function not in {None, "sigmoid", "tanh", "relu", "prelu"}:
                raise Exception("Invalid solver")
            if function == "prelu":
                activation_function[i] = PReLU()
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

        if type(use_gpu) != bool:
            raise Exception("Invalid use_gpu")
        self._use_gpu = use_gpu

        if solver not in {"sgd", "adam"}:
            raise Exception("Invalid solver")

        if solver == "sgd":
            self._optimizer = SGD(
                learning_rate=self._learning_rate_init,
            )
        if solver == "adam":
            if type(beta_1) != int and type(beta_1) != float:
                raise Exception("Invalid beta_1")
            if type(beta_2) != int and type(beta_2) != float:
                raise Exception("Invalid beta_2")
            self._optimizer = Adam(
                beta_1=beta_1,
                beta_2=beta_2,
                learning_rate=self._learning_rate_init,
            )

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

        # setting optimizer
        self._setting_gpu()

    def _create_model(
        self,
    ):

        model = Sequential()

        # first hidden layer
        model.add(
            Dense(
                self._hidden_layer_sizes[0],
                input_dim=self._input_dim,
                activation=self._activation_function[0],
                kernel_regularizer=regularizers.l1_l2(
                    l1=self._l1_lambda[0], l2=self._l2_lambda[0]
                ),
                bias_regularizer=regularizers.l1_l2(
                    l1=self._l1_lambda[0], l2=self._l2_lambda[0]
                ),
            )
        )

        # hidden layer
        for i, number_of_node in enumerate(self._hidden_layer_sizes[1:], start=1):
            if self._dropout_rate[i] != 0:
                model.add(Dropout(self._dropout_rate[i - 1]))
            model.add(
                Dense(
                    number_of_node,
                    activation=self._activation_function[i],
                    kernel_regularizer=regularizers.l1_l2(
                        l1=self._l1_lambda[i], l2=self._l2_lambda[i]
                    ),
                    bias_regularizer=regularizers.l1_l2(
                        l1=self._l1_lambda[i], l2=self._l2_lambda[i]
                    ),
                )
            )

        # output layer
        if self._dropout_rate[-1] != 0:
            model.add(Dropout(self._dropout_rate[-1]))
        model.add(
            Dense(
                1,
                kernel_regularizer=regularizers.l1_l2(
                    l1=self._l1_lambda[-1], l2=self._l2_lambda[-1]
                ),
                bias_regularizer=regularizers.l1_l2(
                    l1=self._l1_lambda[-1], l2=self._l2_lambda[-1]
                ),
            )
        )

        model.compile(
            loss=MeanSquaredError(),
            optimizer=self._optimizer,
        )

        self._model = model

    def _setting_gpu(
        self,
    ):
        device_type = [
            device.device_type for device in tf.config.list_physical_devices()
        ]

        if self._use_gpu and "GPU" in device_type:
            self._device_name = "/device:GPU:0"
        else:
            self._device_name = "/device:CPU:0"

    def fit(self, X, y):
        # initialize tqdm callback with default parameters
        if self._progress_bar:
            callback = [TQDMProgressBar(show_epoch_progress=False)]
        else:
            callback = None

        if self._batch_size is None:
            batch_size = X.shape[0]
        else:
            batch_size = self._batch_size

        # fit the keras model on the dataset
        with tf.device(self._device_name):
            history = self._model.fit(
                X,
                y,
                shuffle=self._shuffle,
                batch_size=batch_size,
                epochs=self._epochs,
                verbose=0,
                callbacks=callback,
                validation_split=self._validation_split,
            ).history
            self.loss_curve_ = history["loss"]
            if self._validation_split > 0:
                self.val_loss_curve_ = history["val_loss"]

            weights = self.get_weights()
            self.coefs_ = [weight for weight in weights if len(weight.shape) == 2]
            self.intercepts_ = [weight for weight in weights if len(weight.shape) == 1]

            return history

    def get_weights(
        self,
    ):
        return self._model.get_weights()

    def predict(self, X):
        return self._model.predict(X)

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
        return self._model.save(path)

    def load_model(
        self,
        path="",
    ):
        self._model = keras.models.load_model(path)

    @classmethod
    def from_config(cls, config):
        return cls(**config)