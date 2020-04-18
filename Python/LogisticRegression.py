import tensorflow as tf
import numpy as np

class LogisticRegression:
    def __init__(self, features, labels, learning_rate=0.1, iterations=50, batchsize=500):
        if (not isinstance(features, tf.Tensor) or not isinstance(labels, tf.Tensor)):
            raise TypeError("Both Features and Labels must be tensors")

        self.features = features
        self.labels = labels

        #options
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.batchsize = batchsize

        self.weights = tf.zeros((np.shape(self.features)[1], np.shape(self.labels)[1]), tf.dtypes.float32)
        self.cost_history = []

    def gradient_descent(self, features, labels):
        current_guesses = tf.linalg.matmul(features, self.weights);
        current_guesses = tf.nn.softmax(current_guesses)

        differences = tf.math.subtract(current_guesses, labels)

        slopes = tf.transpose(features)
        slopes = tf.linalg.matmul(slopes, differences)
        slopes = tf.math.divide(slopes, tf.cast(tf.shape(features)[0], tf.dtypes.float32))

        return tf.math.subtract(self.weights, tf.math.multiply(slopes, self.learning_rate))

    def train(self):
        num_batches = np.floor(np.shape(self.features)[0] / self.batchsize)

        i = 0
        while i < self.batchsize:
            batch_index= 0

            while batch_index < num_batches:
                start_index = batch_index * self.batchsize

                features_slice = tf.slice(self.features, [start_index, 0], [self.batchsize, -1])
                labels_slice = tf.slice(self.labels, [start_index, 0], [self.batchsize, -1])

                self.weights = self.gradient_descent(features_slice, labels_slice)
                batch_index += 1

            self.record_cost();
            self.update_learning_rate()
            i += 1

    def test(testFeatures, testLabels):
        

    def record_cost(self):
        guesses = tf.linalg.matmul(self.features, self.weights)
        guesses = tf.nn.softmax(guesses)

        term_one = tf.transpose(self.labels)
        term_one = tf.linalg.matmul(term_one, tf.math.log(tf.math.add(guesses, 1e-7)))

        term_two = tf.math.multiply(self.labels, -1)
        term_two = tf.math.add(term_two, 1)
        term_two = tf.transpose(term_two)
        # muyltiply by -1, add 1, add 1e-7 (to avoid log(0)), then log
        guesses = tf.math.multiply(guesses, -1)
        guesses = tf.math.add(guesses, 1 + 1e-7)
        guesses = tf.math.log(guesses)
        term_two = tf.linalg.matmul(term_two, guesses)

        cross_entropy = tf.math.add(term_one, term_two)
        cross_entropy = tf.math.divide(cross_entropy, tf.cast(tf.shape(self.features)[0], tf.dtypes.float32))
        cross_entropy = tf.math.multiply(cross_entropy, -1)

        self.cost_history = [] + self.cost_history

    def update_learning_rate(self):
        if(len(self.cost_history) < 2):
            return

        if(self.cost_history[0] > self.cost_history[1]):
            self.learning_rate /= 2
        else:
            self.learning_rate *= 1.05
