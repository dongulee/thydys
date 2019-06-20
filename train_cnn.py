
# coding: utf-8

# # Training of Thyroid dysfunction prediction model w/ LSTM network
# 
# * Using pre-processed data from dat files (X.dat, Y.dat)
# * 

# In[1]:


from os import listdir
from itertools import combinations
import pandas as pd
import numpy as np
import subprocess
import tensorflow as tf
import utils as utl
#from collections import Counter

import sys
from optparse import OptionParser

def RNN(x, weights, biases, timesteps, num_hidden):

    
    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

    
def eval_thydys(x):
    return np.array(list(tf.map_fn(lambda x: 1 if x>=1.8 else 0,x)))


def main():
    # Option Parser
    if (len(sys.argv) <= 1):
        print("memopad.py -h or --help to get guideline of input options")
        exit()
    use = "Usage: %prog [options] filename"
    parser = OptionParser(usage = use)
    parser.add_option("-d", "--input-dir", dest="input_dir", action="store", type="string", help="input data dir")
    parser.add_option("-t", "--timesteps", dest="timesteps", action="store", type="int", help="timesteps")
    parser.add_option("-n", "--num-input", dest="num_input", action="store", type="int", help="number of input (input vector's width)")

    (options, args) = parser.parse_args()
    input_dir = options.input_dir
    timesteps = options.timesteps
    num_input = options.num_input
   
    X = np.fromfile(input_dir + '/X.dat', dtype=float)
    cardinality = int(X.shape[0]/(timesteps * num_input))
    X = X.reshape([cardinality, timesteps, num_input])
    Y = np.fromfile(input_dir + '/Y.dat', dtype=float)
    train_x, val_x, test_x, train_y, val_y, test_y = utl.train_val_test_split(X, Y, split_frac=0.80)
    #print("Data Set Size")
    #print("Train set: \t\t{}".format(train_x.shape), 
    #      "\nValidation set: \t{}".format(val_x.shape),
    #      "\nTest set: \t\t{}".format(test_x.shape))
    
    
    # In[ ]:
    
    
    # Training Parameters
    learning_rate = 0.001
    epochs = 30
    batch_size = 20
    #display_step = 200
    
    # Network Parameters
    #num_input = 2 
    #timesteps = 480 
    #num_hidden = 1024
    num_classes = 1
   
    print("### Network Parameters ###")
    print("Learning Rate: {}".format(learning_rate))
    print("Batch Size: {}".format(batch_size))
    #print("Size of Hidden Layer: {}".format(num_hidden))
    print("Timestep: {}".format(timesteps)) 
    print("------------------")
    X_ = tf.placeholder("float", [None, timesteps, num_input])
    Y_ = tf.placeholder("float", [None, num_classes])
    lr = tf.placeholder("float")
    keep_prob_ = tf.placeholder(tf.float32, name = 'keep')
    # (batch, 480, 2) -> (batch, 240, 18)
    conv1 = tf.layers.conv1d(inputs=X_, filters=18, kernel_size = 4, strides=1, padding='same', activation=tf.nn.relu)
    max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2, padding='same')
    # (batch, 240, 18) -> (batch, 120, 36)
    conv2 = tf.layers.conv1d(inputs=max_pool_1, filters=36, kernel_size = 2, strides=1, padding='same', activation=tf.nn.relu)
    max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2, padding='same')
    # (batch, 120, 36) -> (batch, 60, 24)
    conv3 = tf.layers.conv1d(inputs=max_pool_2, filters=24, kernel_size = 2, strides=1, padding='same', activation=tf.nn.relu)
    max_pool_3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2, padding='same')

    # Flatten and dropout
    flat = tf.reshape(max_pool_3, (-1, 60*24))
    flat = tf.nn.dropout(flat, keep_prob=keep_prob_)

    # Prediction
    prediction = tf.layers.dense(flat, num_classes) 
    loss_op = tf.losses.mean_squared_error(Y_, prediction)
    #optimizer = tf.train.AdadeltaOptimizer(lr).minimize(loss_op)
    optimizer = tf.train.AdamOptimizer(lr).minimize(loss_op)
     
    correct_pred = tf.equal(tf.cast( (prediction/1.8) - tf.round(prediction/1.8), tf.float32), tf.cast( (prediction/1.8)-tf.round(Y_/1.8), tf.float32))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
    
        # Run the initializer
        sess.run(init)
        
        n_batches = len(train_x)//batch_size
        
        for e in range(epochs):
            if epochs%10==0:
                learning_rate = learning_rate*0.9
            train_acc = []
            for ii, (x, y) in enumerate(utl.get_batches(train_x, train_y, batch_size), 1):
                x = x.reshape((batch_size, timesteps, num_input))
                feed = {X_: x, Y_: y[:, None], lr:learning_rate, keep_prob_:0.75}
                loss, acc = sess.run([loss_op, accuracy], feed_dict=feed)
                train_acc.append(acc)
    
                if (ii+1) % n_batches == 0:
                    val_acc = []
                    for xx, yy in utl.get_batches(val_x, val_y, batch_size):
                        xx = xx.reshape((batch_size, timesteps, num_input))
                        feed = {X_:xx,Y_:yy[:,None], lr:learning_rate, keep_prob_:1}
                        val_batch_acc = sess.run([accuracy], feed_dict=feed)
                        val_acc.append(val_batch_acc)
    
                    print("Epoch: {}/{}...".format(e+1, epochs),
                          "Batch: {}/{}...".format(ii+1, n_batches),
                          "Train Loss: {:.3f}...".format(loss),
                          "Train Accruacy: {:.3f}...".format(np.mean(train_acc)),
                          "Val Accuracy: {:.3f}".format(np.mean(val_acc)))
        
       
    
        # Calculate accuracy for 128 mnist test images
        #test_len = 128
        test_data = test_x.reshape((-1, timesteps, num_input))
        test_label = test_y
        print("Testing Accuracy:", sess.run(accuracy, 
            feed_dict={
                X_: test_data, 
                Y_: test_label[:, None], 
                lr:learning_rate,
                keep_prob_:1}))
    
if __name__ == "__main__": 
    main()
