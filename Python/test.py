# import json
# from LogisticRegression import LogisticRegression
# import tensorflow as tf
#
# with open('weights.json') as json_file:
#     data = json.load(json_file)
#     weights = data['weights']
#
# weights = tf.cast(tf.Variable(weights), tf.dtypes.float32)
#
# features = LogisticRegression.matrify_image('./images/image_2')
# print(features)
# features = LogisticRegression.flatten_matrix(features)
# print(features)
# print(LogisticRegression.static_predict(weights, features))

from PIL import Image

img = Image.open('/images/image_2.png').convert('LA')
img.save('greyscale.png')
