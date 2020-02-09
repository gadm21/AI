#this DeepNeuralNetwork trains on mnist dataset

import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.datasets import mnist
from keras.optimizers import RMSprop


batch_size= 128
epochs= 20
num_classes= 10
(x_train, y_train), (x_test, y_test)= mnist.load_data()


#convert x_train & x_test from (60000, 28, 28) to (60000, 784) of type float32
# then normalize them
x_train= (x_train.reshape(60000, 784).astype('float32'))/255
x_test= (x_test.reshape(10000, 784).astype('float32'))/255

#convert y_train & y_test to binary class matricies (hot-one encoded)
y_train= keras.utils.to_categorical(y_train, num_classes)
y_test= keras.utils.to_categorical(y_test, num_classes)


#create an instance of the sequential model
model= Sequential()

#1st layer denses neurons from 784 to 512
model.add(Dense(400, input_shape= (784,)))

#2nd layer connects the 400 neurons in the previous layer
#to a relu activation function, the output still 100 neurons
model.add(Activation('relu'))

#3rd layer is a dropout_layer with keep probability of 20%
model.add(Dropout(0.2))

#4th layer denses the 400 neurons to 60 neurons
model.add(Dense(60))

#5th layer connects the 60 neurons to a relu activation function
model.add(Activation('relu'))

#6th layer. Another Dropout layer
model.add(Dropout(0.2))

#Final layer. Denses 60 neurons to num_classes and an activation function
#softmax is added.
model.add(Dense(num_classes, activation='softmax'))


'''NOTE
we've only set the input shape and keras will infer all 
the shapes after the 1st layer. This is arguably the biggest
advantage to use keras framework
'''


#prints out a summary of the model architecture
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])




