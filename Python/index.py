import tensorflow as tf
import numpy as np
from LogisticRegression import LogisticRegression

def load_data():
    (train_features, train_labels), (test_features, test_labels) = tf.keras.datasets.mnist.load_data()

    #flatten 3 dimensions into 2
    train_features = list(map(lambda matrix: np.array(matrix).flatten().tolist(), train_features))

    #encode labels
    def encode(label):
        encoded_label = [0] * 10
        encoded_label[label] = 1
        return encoded_label
    train_labels = list(map(encode, train_labels))

    return train_features, train_labels, test_features, test_labels

#load_data returns a numpy array
training_data, training_labels, testing_features, testing_labels = load_data()

training_data = tf.convert_to_tensor(training_data, dtype=tf.float32)
training_labels = tf.convert_to_tensor(training_labels, dtype=tf.float32)

regression = LogisticRegression(training_data, training_labels, learning_rate=1, iterations=80, batchsize=500)
regression.train()

accuracy = regression.test(testing_features, testing_labels)
print(accuracy)
