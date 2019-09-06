from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
from helper import get_batches

learning_rate= 0.001

n_features= 784     #image_shape= 28*28
n_classes= 10       #0-9 digits

#import MNIST data
mnist_data= input_data.read_data_sets('/datasets/ud730/mnist', one_hot=True)

#images are already flattened and data is shuffled
train_features= mnist_data.train.images
test_features= mnist_data.test.images
train_labels= mnist_data.train.labels.astype(np.float32)
test_labels= mnist_data.test.labels.astype(np.float32)

'''features and labels placeholders just like train_features &
train_labels, but the number of samples is not 50,000
instead, its size will be reset at each run to be the batch_size.
We use placeholder instread of constant because the value of both
tensors (features & labels) will be set at network running and will
be reset the next running by the next batch size. '''
features= tf.placeholder(tf.float32, [None, n_features])
labels= tf.placeholder(tf.float32, [None, n_classes])

'''weights and bias are variables so that they can be changed and 
optimized througthout training the network. the size is set to be
suitable the matrix multiplication and addition rules  i.e. 
(features*weights)+ bias  
Also, note that weights is initialized as a random matrix from
a random destribution bool to make the model initially unbiased'''
weights= tf.Variable(tf.random_normal([n_features, n_classes]))
bias= tf.Variable(tf.random_normal([n_classes]))

'''the actual network strucure'''
logits= tf.add(tf.matmul(features, weights), bias)

'''this function applies softmax function to logits then cross_entropy function between softmax(logits) and labels '''
cost= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= logits, labels= labels))

'''this tensor calculates the gradiant descent for each trainable variable in the network (all variables are
trainables by default) and returns an op that, when run, updates all trainable variables according to their 
graidiant descent and learning rate.'''
optimizer= tf.train.GradientDescentOptimizer(learning_rate= learning_rate).minimize(cost)

'''tf.argmax returns the index of the biggest value in each batch. 
argmax dimension equals logits or labels dimensions - 1 
i.e. if logits is a matrix, argmax is an array, if logits is an array, argmax is a number
   tf.equal: returns a datastructure of the same size as its arguments of 0s and 1s based
on whether each index contains equal value in both arguments (result in that index'll be 1) or
not (result in that index'll be 0)'''
'''then tf.reduce_mean will return the mean of datastructure returned by tf.equal. If this is a good
model, logits correspond to the right label will be high thus tf.argmax of labels & logits will yeild
the same index and tf.equal will put 1 in the resulting datastructure (correct_prediction). Otherwise,
tf.equal will put 0. Then tf.reduce_mean count the result (1s and 0s) dividing by the size of the
datastrucure thus, the higher the better.'''
correct_prediction= tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

''' Set batch size. This divides the total_size (50,000) into chunks'''
batch_size = 500
assert batch_size is not None, 'You must set the batch size'


with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    for i in range(500):
        '''loop on all batches and train the optimizer on them all'''
        '''each sess.run(optimizer) updates the values of the variables'''
        for batch_features, batch_labels in get_batches(batch_size, train_features, train_labels):
            sess.run(optimizer, 
                     feed_dict= {features: batch_features, labels:batch_labels})
        
        '''calculate accuracy for test dataset'''
        test_accuracy= sess.run(accuracy, 
                                feed_dict= {features:test_features, labels: test_labels})
        
        print("test accuracy: {}", format(test_accuracy))
        
        
        