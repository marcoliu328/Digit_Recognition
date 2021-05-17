import cv2
import os
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

BATCH_SIZE=32
path = r'/Users/marcoliu/Desktop/Github/DIGIT_RECOGNITION/dataset'
model = load_model("digits.model")

mnist = tf.keras.datasets.mnist


for img in os.listdir(path):
    image = cv2.imread(path+ "/" + img)
    imgcopy = image
    image = img_to_array(image)
    image = image[:,:,0]
    image = image.reshape(1, 28, 28, 1)
    #image = cv2.resize(image, (28, 28 ))
    #image = cv2.bitwise_not(image)
    image = image.astype('float32')
    image = 255 - image
    image = image / 255
    prediction = model.predict(image)
    #prediction = model.predict(tf.expand_dims(image, axis=0))
    print(f'The result is probably: {np.argmax(prediction)}')
    plt.imshow(image.reshape(28,28), cmap='Greys_r')
    plt.show()