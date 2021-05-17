import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

BATCH_SIZE = 32
EPOCHS = 10

#grab the mnist digit dataset
mnist = tf.keras.datasets.mnist

#load in data (automa `tically split with load_data)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#normalize the training data
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#build model
model = tf.keras.models.Sequential()
#data is in the shape 28x28, change to fc
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(units=128, activation="relu"))
model.add(tf.keras.layers.Dense(units=10, activation="softmax")) #0-9

#compile model
model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])


#train model
H = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(x_test, y_test), validation_steps=len(x_test)//BATCH_SIZE)

#evaluate
loss, accuracy = model.evaluate(x_test, y_test)
print("Accuracy: {}".format(accuracy))
print("Loss: {}".format(loss))

#save the model
model.save('digits.model')

#plot the loss/accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, EPOCHS), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")






