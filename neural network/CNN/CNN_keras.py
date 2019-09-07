#this CNN trains on mnist dataset

import keras
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten

batchsize= 128
num_classes= 10
epochs= 12

#input size
img_size= (28, 28)

(x_train, y_train), (x_test, y_test)= keras.datasets.mnist.load_data()

'''
we check for image_data_format() because different keras backend, such
as "tensorflow", "cntk", "theano", etc. ,will have different format
: either "channels_first" or "channels_last", in which it expects the 
input images in mutli-dim conv layers.
'''
if keras.backend.image_data_format()== "channels_first":
    x_train= x_train.reshape(x_train.shape[0], 1, img_size[0], img_size[1]).asytpe('float32')/255
    x_test= x_test.reshape(x_test.shape[0], img_size[0], img_size[1], 1).astype('float32')/255
    input_shape= (1, img_size[0], img_size[1])
else:
    x_train= x_train.reshape(x_train.shape[0], img_size[0], img_size[1], 1)
    x_test= x_test.reshape(x_test.shape[0], img_size[0], img_size[1], 1)
    input_shape= (img_size[0], img_size[1], 1)



#convert class vectors to binary class matricies (one_hot encoded)
y_train= keras.utils.to_categorical(y_train, num_classes)
y_test= keras.utils.to_categorical(y_test, num_classes)

model= keras.Sequential()

'''
conv layer with input_shape set as the img shape and the output depth dimension (num of filters)
set to 32, filter size 3. height and width of the 
output is, for height for example,  (input_h- kernel_h +1/stride)
which is (28-3+ 1)/1= 26
we also add a relu activation function on the fly
Note that strides are by default: (1, 1) 
and padding is by default: "valid"
'''
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape))

'''conv layer with input_shape infered automagically by keras from 
the previous layer, and output depth is increased from 32 to 64
we also add a relu activation function on the fly
'''
model.add(Conv2D(64, (3,3), activation='relu'))


'''
pooling layer. DUH
strides is by default: the same as pool_size
'''
model.add(MaxPooling2D(pool_size=(2,2)))

'''
a dropout layer with keeping probability= 25%
'''
model.add(Dropout(0.25))

'''
a flatten layer to, well... flatten !
'''
model.add(Flatten())

'''
now we start the fully connected network part
this layer denses the flattened array of neurons from the previous 
layer to 128 neurons
Also, an activation function is added right after it
'''
model.add(Dense(128, activation='relu'))

'''
a dropout layer with keeping propability of 50%
'''
model.add(Dropout(0.5))

'''
another dense layer that will finally dense the 128 neurons form
the previous layer to the 10 neurons corresponding to the 10 classes
we have. Also, we'll add a softmax activation function to change
the values in these 10 neurons to actual probabilities (that sums to 1 :D )
'''
model.add(Dense(num_classes, activation= 'softmax'))


'''
from here:: https://datascience.stackexchange.com/questions/46124/what-do-compile-fit-and-predict-do-in-keras-sequential-models
"This will create a Python object which will build the CNN. This is done by 
building the computation graph in the correct format based on the Keras backend used"
The compilation steps also asks you to define the loss function and kind of optimizer you want to use.
These options depend on the problem you are trying to solve, 
you can find the best techniques usually reading the literature in the field.
 For a classification task categorical cross-entropy works very well. "
 '''
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
             
             
# this will save the model architecture (the network) not the weights
model_json = model.to_json()
with open("weights/model.json", "w") as json_file:
    json_file.write(model_json)
    

# Save the weights using a checkpoint. when we get better results
filepath="weights/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


'''
Also from here:: https://datascience.stackexchange.com/questions/46124/what-do-compile-fit-and-predict-do-in-keras-sequential-models
"This will fit the model parameters to the data.
'''
model.fit(x_train, y_train,
          batch_size=batchsize,
          epochs=epochs,
          verbose=1,
          callbacks= callbacks_list,
          validation_data=(x_test, y_test))
         
         
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])