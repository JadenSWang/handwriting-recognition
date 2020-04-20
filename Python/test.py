import json
import tensorflow as tf
import numpy as np
from LogisticRegression import LogisticRegression
from PIL import Image

with open('weights.json') as json_file:
    data = json.load(json_file)
    weights = data['weights']

weights = tf.cast(tf.Variable(weights), tf.dtypes.float32)
features = LogisticRegression.matrify_image('images/image_1.png')
features = [features]
features = LogisticRegression.flatten_matrix(features)


def load_data():
    (train_features, train_labels), (test_features, test_labels) = tf.keras.datasets.mnist.load_data()

    # flatten 3 dimensions into 2
    train_features = LogisticRegression.flatten_matrix(train_features)
    test_features = LogisticRegression.flatten_matrix(test_features)

    # encode labels
    def encode(label):
        encoded_label = [0] * 10
        encoded_label[label] = 1
        return encoded_label
    train_labels = list(map(encode, train_labels))
    test_labels = list(map(encode, test_labels))

    return train_features, train_labels, test_features, test_labels


# load_data returns a numpy array
training_data, training_labels, testing_features, testing_labels = load_data()

regression = LogisticRegression(training_data, training_labels, learning_rate=1, iterations=80, batchsize=500)
regression.weights = weights

accuracy = regression.test(features, [testing_labels[0]]).numpy().tolist()
