
import numpy as np
import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()
tf.reset_default_graph()

def reshape_and_onehot(train_images, train_labels):
    train_images= np.reshape(train_images, [train_images.shape[0], train_images.shape[1]* train_images.shape[2]])
    train_labels_onehot= np.zeros((train_labels.shape[0], num_classes))
    train_labels_onehot[np.arange(train_labels.shape[0]), train_labels]= 1

    return train_images, train_labels_onehot

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
num_classes= 10 #compute?

train_images, train_labels= reshape_and_onehot(train_images, train_labels)
test_images, test_labels= reshape_and_onehot(test_images, test_labels)



X= tf.placeholder(name= 'input', shape= [None, train_images.shape[1]], dtype= tf.float32)
Y= tf.placeholder(name= 'label', shape= [None, num_classes], dtype= tf.float32)

var_init= tf.truncated_normal_initializer(mean= 0, stddev= 0.1)
W= tf.get_variable(name= 'weight', dtype= tf.float32, shape= [train_images.shape[1], num_classes], initializer= var_init)
b= tf.get_variable(name= 'bias', dtype= tf.float32, initializer= tf.zeros([num_classes]))

logits= tf.matmul(X, W) + b
Y_hat= tf.nn.softmax(name= 'y_hat', logits= logits)

loss= tf.nn.softmax_cross_entropy_with_logits_v2(name= 'loss', labels= Y, logits= Y_hat)
loss_summary= tf.summary.scalar(name= 'loss_summary', tensor= tf.reduce_mean(loss))

optimizer= tf.train.AdamOptimizer(learning_rate= 0.0001, name= 'adam_opt').minimize(loss)

correct_prediction= tf.equal(tf.argmax(Y_hat, axis= 1), tf.argmax(Y, axis= 1), name= 'correct_prediction')
accuracy= tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name= 'accuracy')
accuracy_summary= tf.summary.scalar(name= 'accuracy_summary', tensor= accuracy)



saver= tf.train.Saver()

epochs= 150
batch_size= 6
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer= tf.summary.FileWriter(graph= sess.graph,logdir='./graphs')
    writer_index= 0
    total_training_images= train_images.shape[0]
    batch_training_images= int(total_training_images// batch_size)

    for i in range(epochs):

        for j in range(0, total_training_images, batch_training_images):
            x_batch, y_batch= train_images[j:j+batch_training_images], train_labels[j:j+batch_training_images]
            feed_dict= {X: x_batch, Y: y_batch}
            _, loss_out, accuracy_out, ac= sess.run([optimizer, loss_summary, accuracy_summary, accuracy], feed_dict= feed_dict)
        print("epoch {:d} with accuracy:".format(i), ac*100)
        #loss_summary_out, accuracy_summary_out= sess.run([loss_summary, accuracy_summary])
        writer.add_summary(loss_out, writer_index)
        writer.add_summary(accuracy_out, writer_index)
        writer_index+=1

print("done")