from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten

#create an instance of the sequential model
model= Sequential()

#1st layer is designed to take an image and flatten it.
#(32, 32, 3) to (3072=32 x 32 x 3)
model.add(Flatten(input_shape=(32, 32, 3)))

#2nd layer takes the (32*32*3) neurons in the previous layer
#and connect them to 100 neurons as outputs of this layer
 model.add(Dense(100))

 #3rd layer connects the 100 neurons in the previous layer
 #to a relu activation function, the output still 100 neurons
 model.add(Activation('relu'))

#4th layer connects the 100 neurons to 60 neurons
 model.add(Dense(60))

 #5th layer connects the 60 neurons to a relu activation function
 model.add(Activation('relu'))

 '''NOTE
 we've only set the input shape and keras will infer all 
 the shapes after the 1st layer. This is arguably the biggest
 advantage to use keras framework
 '''

