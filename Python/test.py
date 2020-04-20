import json
import tensorflow as tf
import numpy as np
from LogisticRegression import LogisticRegression
from PIL import Image

with open('weights.json') as json_file:
    data = json.load(json_file)
    weights = data['weights']

weights = tf.cast(tf.Variable(weights), tf.dtypes.float32)
features = LogisticRegression.matrify_image('images/image_6.png')
(train_features, train_labels), (test_features, test_labels) = tf.keras.datasets.mnist.load_data()

print(features)
print('[', end='')
i = 0
while i < len(test_features[0]):
    j = 0
    print('[', end='')
    while j < len(test_features[0][i]):
        print(test_features[0][i][j], end=',')
        j += 1
    print(']')
    i += 1
print(']', end='')

print(np.shape(features))
print(np.shape(test_features[0]))
features = LogisticRegression.flatten_matrix(features)

print(LogisticRegression.static_predict(weights, features).numpy().tolist())
