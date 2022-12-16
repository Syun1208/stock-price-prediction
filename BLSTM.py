import numpy as np
import pandas as pd
import tensorflow as tf
import glob
import os
import csv
import argparse
import tqdm
from sklearn.preprocessing import MinMaxScaler


# import keras
# from tensorflow.keras.models import Model

# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession


def merge_csi_label(csifile, labelfile, win_len=500, thrshd=0.6, step=50):
    """
    Merge CSV files into a Numperrory Array  X,  csi amplitude feature
    Returns Numpy Array X, Shape(Num, Win_Len, 90)
    Args:
        csifile  :  str, csv file containing CSI data
        labelfile:  str, csv fiel with activity label 
        win_len  :  integer, window length
        thrshd   :  float,  determine if an activity is strong enough inside a window
        step     :  integer, sliding window by step
    """
    activity = []
    with open(labelfile, 'r') as labelf:
        reader = csv.reader(labelf)
        for line in reader:
            label = line[0]
            if label == 'NoActivity':
                activity.append(0)
            else:
                activity.append(1)
    activity = np.array(activity)
    csi = []
    with open(csifile, 'r') as csif:
        reader = csv.reader(csif)
        for line in reader:
            line_array = np.array([float(v) for v in line])
            # extract the amplitude only
            line_array = line_array[0:1]
            csi.append(line_array[np.newaxis, ...])
    csi = np.concatenate(csi, axis=0)
    assert (csi.shape[0] == activity.shape[0])
    # screen the data with a window
    index = 0
    feature = []
    while index + win_len <= csi.shape[0]:
        cur_activity = activity[index:index + win_len]
        if np.sum(cur_activity) < thrshd * win_len:
            index += step
            continue
        cur_feature = np.zeros((1, win_len, 1))
        cur_feature[0] = csi[index:index + win_len, :]
        feature.append(cur_feature)
        index += step
    return np.concatenate(feature, axis=0)


def extract_csi_by_label(raw_folder, label, labels, save=False, win_len=500, thrshd=0.6, step=50):
    """
    Returns all the samples (X,y) of "label" in the entire dataset
    Args:
        raw_folder: The path of Dataset folder
        label    : str, could be one of labels
        labels   : list of str, ['lie down', 'fall', 'bend', 'run', 'sitdown', 'standup', 'walk']
        save     : boolean, choose whether save the numpy array 
        win_len  :  integer, window length
        thrshd   :  float,  determine if an activity is strong enough inside a window
        step     :  integer, sliding window by step
    """
    print('Starting Extract CSI for Label {}'.format(label))
    label = label.lower()
    if not label in labels:
        raise ValueError(
            "The label {} should be among 'lie down','fall','bend','run','sitdown','standup','walk'".format(labels))

    data_path_pattern = os.path.join(raw_folder, label, 'user_*' + label + '*.csv')
    input_csv_files = sorted(glob.glob(data_path_pattern))
    # annot_csv_files = [os.path.basename(fname).replace('user_', 'annotation_user') for fname in input_csv_files]
    # annot_csv_files = [os.path.join(raw_folder, label, fname) for fname in annot_csv_files]
    annot_csv_files = os.path.join(raw_folder, label, 'Annotation_user_*' + label + '*.csv')
    annot_csv_files = sorted(glob.glob(annot_csv_files))
    feature = []
    index = 0
    for csi_file, label_file in tqdm.tqdm(zip(input_csv_files, annot_csv_files)):
        index += 1
        if not os.path.exists(label_file):
            print('Warning! Label File {} doesn\'t exist.'.format(label_file))
            continue
        feature.append(merge_csi_label(csi_file, label_file, win_len=win_len, thrshd=thrshd, step=step))
        print('Finished {:.2f}% for Label {}'.format(index / len(input_csv_files) * 100, label))

    feat_arr = np.concatenate(feature, axis=0)
    if save:
        np.savez_compressed("X_{}_win_{}_thrshd_{}percent_step_{}.npz".format(
            label, win_len, int(thrshd * 100), step), feat_arr)
    # one hot
    feat_label = np.zeros((feat_arr.shape[0], len(labels)))
    feat_label[:, labels.index(label)] = 1
    return feat_arr, feat_label


def train_valid_split(numpy_tuple, train_portion=0.75, seed=379):
    """
    Returns Train and Valid Datset with the format of (x_train, y_train, x_valid, y_valid),
    where x_train and y_train are shuffled randomly.

    Args:
        numpy_tuple  : tuple of numpy array: (x_lie_down, x_fall, x_bend, x_run, x_sitdown, x_standup, x_walk)
        train_portion: float, range (0,1)
        seed         : random seed
    """
    np.random.seed(seed=seed)
    x_train = []
    x_valid = []
    y_valid = []
    y_train = []

    for i, x_arr in enumerate(numpy_tuple):
        index = np.random.permutation([i for i in range(x_arr.shape[0])])
        split_len = int(train_portion * x_arr.shape[0])
        x_train.append(x_arr[index[:split_len], ...])
        tmpy = np.zeros((split_len, 1))
        tmpy[:, i] = 1
        y_train.append(tmpy)
        x_valid.append(x_arr[index[split_len:], ...])
        tmpy = np.zeros((x_arr.shape[0] - split_len, 1))
        tmpy[:, i] = 1
        y_valid.append(tmpy)

    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    x_valid = np.concatenate(x_valid, axis=0)
    y_valid = np.concatenate(y_valid, axis=0)

    index_train = np.random.permutation([i for i in range(x_train.shape[0])])
    x_train = x_train[index_train, ...]
    y_train = y_train[index_train, ...]
    index_valid = np.random.permutation([i for i in range(x_valid.shape[0])])
    x_valid = x_valid[index_valid, ...]
    y_valid = y_valid[index_valid, ...]
    return x_train, y_train, x_valid, y_valid


def split_dataset(df, numpy_tuple, train_proportion, num_steps):
    split_len = int(train_proportion * numpy_tuple.shape[0])
    training_set = df.iloc[:split_len, 4:5].values
    validation_set = df.iloc[split_len:, 4:5].values
    sc = MinMaxScaler(feature_range=(-1, 2))
    training_set_scaled = sc.fit_transform(training_set)
    validation_set_scaled = sc.fit_transform(validation_set)
    print(training_set_scaled.shape)
    print(validation_set_scaled.shape)
    X_train = []
    y_train = []
    X_val = []
    y_val = []
    for i in range(num_steps, split_len):
        X_train.append(numpy_tuple[i - num_steps:i, 0])
        y_train.append(numpy_tuple[i, 0])
    for j in range(num_steps, numpy_tuple.shape[0] - split_len):
        X_val.append(numpy_tuple[j - num_steps:j, 0])
        y_val.append(numpy_tuple[j, 0])
    X_train, y_train = np.array(X_train), np.array(y_train).reshape(len(y_train), 1)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_val, y_val = np.array(X_val), np.array(y_val).reshape(len(y_val), 1)
    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))
    return X_train, y_train, X_val, y_val


def extract_csi(raw_folder, labels, save=False, win_len=500, thrshd=0.6, step=50):
    """
    Return List of Array in the format of [X_label1, y_label1, X_label2, y_label2, .... X_Label7, y_label7]
    Args:
        raw_folder: the folder path of raw CSI csv files
        labels    : all the labels existing in the folder
        save      : boolean, choose whether save the numpy array 
        win_len   :  integer, window length
        thrshd    :  float,  determine if an activity is strong enough inside a window
        step      :  integer, sliding window by step
    """
    ans = []
    for label in labels:
        feature_arr, label_arr = extract_csi_by_label(raw_folder, label, labels, save, win_len, thrshd, step)
        ans.append(feature_arr)
        ans.append(label_arr)
    return tuple(ans)


class AttenLayer(tf.keras.layers.Layer):
    """
    Attention Layers used to Compute Weighted Features along Time axis
    Args:
        num_state :  number of hidden Attention state

    edited code provided on https://github.com/ludlows
    """

    def __init__(self, num_state, **kw):
        super(AttenLayer, self).__init__(**kw)
        self.prob_kernel = None
        self.bias = None
        self.kernel = None
        self.num_state = num_state

    def build(self, input_shape):
        self.kernel = self.add_weight('kernel', shape=[input_shape[-1], self.num_state])
        self.bias = self.add_weight('bias', shape=[self.num_state])
        self.prob_kernel = self.add_weight('prob_kernel', shape=[self.num_state])

    def call(self, input_tensor, **kwargs):
        atten_state = tf.tanh(tf.tensordot(input_tensor, self.kernel, axes=1) + self.bias)
        logits = tf.tensordot(atten_state, self.prob_kernel, axes=1)
        prob = tf.nn.softmax(logits)
        weighted_feature = tf.reduce_sum(tf.multiply(input_tensor, tf.expand_dims(prob, -1)), axis=1)
        return weighted_feature

    # for saving the model
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_state': self.num_state, })
        return config


class CSIModelConfig:
    """
    class for Human Activity Recognition ("lie down", "fall", "bend", "run", "sitdown", "standup", "walk")
    Using CSI (Channel State Information)

    Args:
        win_len   :  integer (500 default) window length for batching sequence
        step      :  integer (200  default) sliding window by this step
        thrshd    :  float   (0.6  default) used to check if the activity is intensive inside a window
        downsample:  integer >=1 (2 default) downsample along the time axis
    """

    def __init__(self, win_len=500, step=50, thrshd=0.6, downsample=1):
        self._win_len = win_len
        self._step = step
        self._thrshd = thrshd
        self._labels = ["money"]
        self._downsample = downsample

    def preprocessing(self, raw_folder, save=False):
        """
        Returns the Numpy Array for training within the format of (X_lable1, y_label1, ...., X_label7, y_label7)
        Args:
            raw_folder: the folder containing raw CSI
            save      : choose if save the numpy array
        """
        numpy_tuple = extract_csi(raw_folder, self._labels, save, self._win_len, self._thrshd, self._step)
        if self._downsample > 1:
            return tuple([v[:, ::self._downsample, ...] if i % 2 == 0 else v for i, v in enumerate(numpy_tuple)])
        return numpy_tuple

    def load_csi_data_from_files(self, np_files):
        """
        Returns the Numpy Array for training within the format of (X_lable1, y_label1, ...., X_label7, y_label7)
        Args:
            np_files: ('x_lie_down.npz', 'x_fall.npz', 'x_bend.npz', 'x_run.npz', 'x_sitdown.npz', 'x_standup.npz', 'x_walk.npz')
        """
        if len(np_files) != 1:
            raise ValueError('There should be 7 numpy files for money')
        x = [np.load(f)['arr_0'] for f in np_files]
        if self._downsample > 1:
            x = [arr[:, ::self._downsample, :] for arr in x]
        y = [np.zeros((arr.shape[0], len(self._labels))) for arr in x]
        numpy_list = []
        for i in range(len(self._labels)):
            y[i][:, i] = 1
            numpy_list.append(x[i])
            numpy_list.append(y[i])
        return tuple(numpy_list)

    def build_model(self, n_unit_lstm=200, n_unit_atten=400):
        """
        Returns the Tensorflow Model which uses AttenLayer
        """
        if self._downsample > 1:
            length = len(np.ones((self._win_len,))[::self._downsample])
            x_in = tf.keras.Input(shape=(length, 1))
        else:
            x_in = tf.keras.Input(shape=(self._win_len, 1))
        x_tensor = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=n_unit_lstm, return_sequences=True))(x_in)
        x_tensor = tf.keras.layers.Dropout(0.3)(x_tensor)
        x_tensor = AttenLayer(n_unit_atten)(x_tensor)
        # x_tensor = tf.keras.layers.Dense(512, activation='leaky_relu')(x_tensor)
        # x_tensor = tf.keras.layers.Dropout(0.5)(x_tensor)
        # x_tensor = tf.keras.layers.Dense(256, activation='leaky_relu')(x_tensor)
        # x_tensor = tf.keras.layers.Dropout(0.5)(x_tensor)
        # x_tensor = tf.keras.layers.Dense(128, activation='leaky_relu')(x_tensor)
        # x_tensor = tf.keras.layers.Dropout(0.2)(x_tensor)
        # x_tensor = tf.keras.layers.Dense(64, activation='leaky_relu')(x_tensor)
        x_tensor = tf.keras.layers.Dropout(0.2)(x_tensor)
        pred = tf.keras.layers.Dense(len(self._labels), activation='leaky_relu')(x_tensor)
        # pred = tf.keras.layers.Dropout(0.5)(pred)
        model = tf.keras.Model(inputs=x_in, outputs=pred)
        return model

    @staticmethod
    def load_model(hdf5path):
        """
        Returns the Tensorflow Model for AttenLayer
        Args:
            hdf5path: str, the model file path
        """
        model = tf.keras.models.load_model(hdf5path, custom_objects={'AttenLayer': AttenLayer})
        return model


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-train', type=str, help='Input your train', default='datasets/')
    parser.add_argument('--path-test', type=str, help='Input your test', default='test/')
    return parser.parse_args()


def read_test_datasets(df_test, df, numpy_tuple, train_proportion, num_step):
    num_train = int(train_proportion * numpy_tuple.shape[0])
    testing_set = df_test.filter(['Close'])
    # sc = MinMaxScaler(feature_range=(-1, 2))
    # testing_set = sc.fit_transform(testing_set)
    training_set = df.iloc[:num_train, 4:5]
    print(testing_set.shape)
    print(training_set.shape)
    total_dataset = pd.concat((training_set['Close'], testing_set['Close']), axis=0)
    inputs = total_dataset[len(total_dataset) - len(training_set) - num_step:].values
    inputs = inputs.reshape(-1, 1)
    # sc = MinMaxScaler(feature_range=(-1, 2))
    # inputs = sc.fit_transform(inputs)
    X_test = []
    for i in range(num_step, inputs.shape[0]):
        X_test.append(inputs[i - num_step:i, :])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    return X_test, testing_set


def train_val_split(data, train_proportion, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps + 1):
        X.append(data[i:i + n_steps, :-1])
        y.append(data[i + n_steps - 1, -1])
    split_inx = int(np.ceil(len(np.array(X) * train_proportion)))
    X_train, X_test = X[:split_inx], X[split_inx:]
    y_train, y_test = y[:split_inx], y[split_inx:]
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)


def main():
    import sys

    if len(sys.argv) != 2:
        SystemError("Error! Correct Command: python3 csimodel.py Dataset_folder_path")
    raw_data_folder = sys.argv[0]
    args = parser_args()
    # preprocessing
    cfg = CSIModelConfig(win_len=2, step=1, thrshd=0.6, downsample=1)
    x_money, y_money = cfg.preprocessing(args.path_train, save=True)
    print(np.array(x_money).shape)
    # load previous saved numpy files, ignore this if you haven't saved numpy array to files before
    # numpy_tuple = cfg.load_csi_data_from_files(('x_lie_down.npz', 'x_fall.npz', 'x_bend.npz', 'x_run.npz', 'x_sitdown.npz', 'x_standup.npz', 'x_walk.npz'))
    # x_train, y_train, x_valid, y_valid = train_valid_split([x_money], train_portion=0.75, seed=379)
    # x_train, y_train, x_valid, y_valid = train_val_split(x_money, train_proportion=0.75, n_steps=2)
    df = pd.read_csv('datasets/money/BTC-USD.csv')
    df_test = pd.read_csv('test/money/BTC-USD-test.csv')
    # Feature Scaling
    x_train, y_train, x_valid, y_valid = split_dataset(df, x_money, train_proportion=0.75, num_steps=2)
    print(x_train.shape, y_train.shape, x_valid.shape, y_valid.shape)
    x_test, testing_set = read_test_datasets(df_test, df, x_money, train_proportion=0.75, num_step=2)
    # parameters for Deep Learning Model
    model = cfg.build_model(n_unit_lstm=10, n_unit_atten=10)
    # train
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='mse',
        metrics=['accuracy'])
    model.summary()
    history = model.fit(
        x_train,
        y_train,
        batch_size=64, epochs=200,
        validation_data=(x_valid, y_valid),
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint('best_attend.hdf5',
                                               monitor='val_accuracy',
                                               save_best_only=True,
                                               save_weights_only=False)
        ])
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    # print(np.array(y_pred).shape)
    # print(np.array(y_valid).shape)
    '''---------------------------------------------------------------------------------------------------------------'''
    # cm = confusion_matrix(np.argmax(y_valid, axis=1), np.argmax(y_pred, axis=1))
    # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # cmd = ConfusionMatrixDisplay(cm, display_labels=["money"])
    plt.figure(figsize=(40, 40))
    # # print(cm)
    # cmd.plot()
    # plt.title('confusion matrix')
    # plt.ylabel('y_valid')
    # plt.xlabel('y_pred')
    # plt.savefig('confusion_matrix.png')
    # plt.show()

    # plot curves
    import matplotlib.pyplot as plt

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('model_accuracy.png')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('model_loss.png')
    plt.show()
    '''---------------------------------------------------------------------------------------------------------------'''
    # load testing datasets
    # from sklearn.preprocessing import MinMaxScaler
    # df_test = pd.read_csv('test/money/BTC-USD-test.csv')
    # df_test = df_test.to_dict('list')
    # scale = MinMaxScaler(feature_range=(0, 1))
    # df_test = scale.fit_transform(np.array(df_test['Close']).reshape(-1, 1))
    # x_test_money, y_test_money = cfg.preprocessing(args.path_test, save=True)
    # x_train_test, y_train_test, x_valid_test, y_valid_test = train_valid_split([x_test_money], train_portion=0.75,
    #                                                                            seed=379)
    # y_pred_test = model.predict(x_valid_test)
    # print('X valid: ', np.array(x_valid).shape)
    # print('X valid test: ', np.array(x_valid_test).shape)
    # print('Y test: ', np.array(y_test).shape)
    # Test
    model = cfg.load_model('best_attend.hdf5')
    # y_pred = model.predict(x_valid)
    y_test = model.predict(x_test)
    '''---------------------------------------------------------------------------------------------------------------'''
    # plt.plot([i for i, _ in enumerate(y_pred)], y_pred)
    # plt.plot([i for i, _ in enumerate(y_valid)], y_valid)
    # # plt.plot([i for i, _ in enumerate(y_test.transpose(2, 0, 1).reshape(-1, y_test.shape[1]))],
    # #          x_valid.transpose(2, 0, 1).reshape(-1, x_valid.shape[1]))
    # plt.title('Correlation training datasets')
    # plt.ylabel('value')
    # plt.xlabel('index')
    # plt.legend(['y_pred', 'y_val'], loc='upper left')
    # plt.savefig('model_prediction_train.png')
    # plt.show()
    #
    # plt.plot([i for i, _ in enumerate(y_pred_test)], y_pred_test)
    # plt.plot([i for i, _ in enumerate(y_valid_test)], y_valid_test)
    # # plt.plot([i for i, _ in enumerate(y_test.transpose(2, 0, 1).reshape(-1, y_test.shape[1]))],
    # #          x_valid.transpose(2, 0, 1).reshape(-1, x_valid.shape[1]))
    # plt.title('Correlation training datasets')
    # plt.ylabel('value')
    # plt.xlabel('index')
    # plt.legend(['y_pred', 'y_test'], loc='upper left')
    # plt.savefig('model_prediction_test.png')
    # plt.show()
    plt.plot(testing_set, 'b-', label='Real Stock Price Close')
    plt.plot(y_test, 'r-', label='Predicted Stock Price')
    plt.title('Doge Price Prediction')
    plt.savefig('model_prediction.png')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
