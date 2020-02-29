import config
import cnn_model
import helper

import numpy as np 
np.random.seed(10)
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.reset_default_graph()


(train_images, train_labels), (test_images, test_labels)= tf.keras.datasets.mnist.load_data() 
train_images, train_labels= helper.reshape_and_onehot(train_images, train_labels)
test_images, test_labels= helper.reshape_and_onehot(test_images, test_labels)

with tf.variable_scope('Input'):
    X= tf.placeholder(name= 'x', shape= [None, train_images.shape[1], train_images.shape[2], train_images.shape[3]], dtype= tf.float32)
    Y= tf.placeholder(name= 'y', shape= [None, config.num_classes], dtype= tf.float32)
    output_logits= cnn_model.model(X)

with tf.variable_scope('Train'):

    with tf.variable_scope('Loss'):
        loss= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels= Y, logits= output_logits), name= 'loss')
    tf.summary.scalar('loss_summary', loss)
    
    with tf.variable_scope('Optimizer'):
        optimizer= tf.train.AdamOptimizer(learning_rate= config.learning_rate, name= 'AdamOpt').minimize(loss)

    with tf.variable_scope('Accuracy'):
        correct_predictions= tf.equal(tf.argmax(output_logits, 1), tf.argmax(Y, 1), name= 'correct_prediction')
        accuracy= (tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name= 'accuracy'))
    tf.summary.scalar('accuracy_summary', accuracy)

    with tf.variable_scope('Prediction'):
        cls_prediction= tf.argmax(output_logits, axis= 1, name= 'predictions')
    

    variables_init= tf.global_variables_initializer()
    merged_summaries= tf.summary.merge_all()


    sess= tf.InteractiveSession()
    sess.run(variables_init)

    global_step= 0
    summary_writer= tf.summary.FileWriter(config.logs_dir, sess.graph)
    iterations= int(len(train_labels)// config.batch_size)

    for epoch in range(config.epochs):
        train_images, train_labels= helper.randomize(train_images, train_labels)
        for iteration in range(iterations):
            global_step+= 1
            start= iteration * config.batch_size
            end= (iteration+1) * config.batch_size
            x_batch, y_batch= helper.get_next(train_images, train_labels, start, end)
            
            feed_dict= {X: x_batch, Y:y_batch}
            sess.run(optimizer, feed_dict= feed_dict)
            
            if iteration % config.display_frequency == 0:
                batch_loss, batch_accuracy, summaries_out= sess.run([loss, accuracy, merged_summaries], feed_dict= feed_dict)
                summary_writer.add_summary(summaries_out, global_step)
                print("iter {0:3d}:\t Loss={1:.2f},\tTraining Accuracy={2:.01%}".
                  format(iteration, batch_loss, batch_accuracy))
        
        valid_feed_dict= {X:test_images, Y:test_labels}
        validation_loss, validation_accuracy= sess.run([loss, accuracy], feed_dict= valid_feed_dict)
        print('---------------------------------------------------------')
        print("Epoch: {0}, validation loss: {1:.2f}, validation accuracy: {2:.01%}".format(epoch, validation_loss, validation_accuracy))
        print('---------------------------------------------------------')    


sess.close()