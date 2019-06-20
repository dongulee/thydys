
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
#sys.stdin.encoding
#tf.disable_v2_behavior()

# In[2]:


X = np.fromfile('X.dat', dtype=float).reshape([96,14400])
Y = np.fromfile('Y.dat', dtype=float)
train_x, val_x, test_x, train_y, val_y, test_y = utl.train_val_test_split(X, Y, split_frac=0.80)
#print("Data Set Size")
#print("Train set: \t\t{}".format(train_x.shape), 
#      "\nValidation set: \t{}".format(val_x.shape),
#      "\nTest set: \t\t{}".format(test_x.shape))


# In[ ]:


# Training Parameters
learning_rate = 0.1
epochs = 30
batch_size = 10
display_step = 200

# Network Parameters
num_input = 2 
timesteps = 480 
num_hidden = 128 
num_classes = 1

input_dim = 960 
h1_dim = 5000
h2_dim = 1000
out_dim = 1


X_ = tf.placeholder("float", [None, input_dim])
Y_ = tf.placeholder("float", [None, out_dim])

weights = {
    'W1':tf.Variable(tf.random_normal([input_dim, h1_dim])),
    'W2':tf.Variable(tf.random_normal([h1_dim, h2_dim])),
    'out':tf.Variable(tf.random_normal([h2_dim, out_dim]))
}
biases = {
    'W1':tf.Variable(tf.random_normal([h1_dim])),
    'W2':tf.Variable(tf.random_normal([h2_dim])),
    'out':tf.Variable(tf.random_normal([out_dim]))
}
def FCNET(x, weights, biases):
    h1 = tf.matmul(x, weights['W1']) + biases['W1']
    h2 = tf.matmul(h1, weights['W2']) + biases['W2']
    pred = tf.matmul(h2, weights['out']) + biases['out']
    return pred

def RNN(x, weights, biases):

    
    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

    

#prediction = RNN(X_, weights, biases)
prediction = FCNET(X_, weights, biases)

loss_op = tf.losses.mean_squared_error(Y_, prediction)
optimizer = tf.train.AdadeltaOptimizer(learning_rate).minimize(loss_op)



correct_pred = tf.equal(tf.cast(tf.round(prediction), tf.float32), Y_)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    
    n_batches = len(train_x)//batch_size
    
    for e in range(epochs):
        train_acc = []
        for ii, (x, y) in enumerate(utl.get_batches(train_x, train_y, batch_size), 1):
            x = x.reshape((batch_size, input_dim))
            feed = {X_: x, Y_: y[:, None]}
            loss, acc = sess.run([loss_op, accuracy], feed_dict=feed)
            train_acc.append(acc)

            if (ii+1) % n_batches == 0:
                val_acc = []
                for xx, yy in utl.get_batches(val_x, val_y, batch_size):
                    xx = xx.reshape((batch_size, input_dim))
                    feed = {X_:xx,Y_:yy[:,None]}
                    val_batch_acc = sess.run([accuracy], feed_dict=feed)
                    val_acc.append(val_batch_acc)

                print("Epoch: {}/{}...".format(e+1, epochs),
                      "Batch: {}/{}...".format(ii+1, n_batches),
                      "Train Loss: {:.3f}...".format(loss),
                      "Train Accruacy: {:.3f}...".format(np.mean(train_acc)),
                      "Val Accuracy: {:.3f}".format(np.mean(val_acc)))
    
   

    # Calculate accuracy for 128 mnist test images
    #test_len = 128
    test_data = test_x.reshape((-1, input_dim))
    test_label = test_y
    print("Testing Accuracy:",         sess.run(accuracy, feed_dict={X_: test_data, Y_: test_label[:, None]}))

