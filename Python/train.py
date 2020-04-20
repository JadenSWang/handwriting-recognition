import tensorflow as tf
import numpy as np
from LogisticRegression import LogisticRegression
import json


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
regression.train()

accuracy = regression.test(testing_features, testing_labels).numpy().tolist()
accuracy = regression.test([testing_features[0]], [testing_labels[0]]).numpy().tolist()

# store data as json
data = {}
data["accuracy"] = accuracy
data["weights"] = regression.weights.numpy().tolist()

with open('weights.json', 'w') as outfile:
    json.dump(data, outfile)
