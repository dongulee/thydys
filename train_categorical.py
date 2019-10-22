# coding: utf-8

'''
# # Training of Thyroid dysfunction prediction model w/ LSTM network
# 
# * Using pre-processed data from dat files (X.dat, Y.dat)
data description
X: 
- sequence of HR and acitvity
- categorical data
  + age
  + gender
  + Ht (height)

Y:
- free T4

'''


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
    # Y (freeT4) = W_x * o_T (final output from LSTM) + W_status * S + biases
    return tf.matmul(outputs[-1], weights['out']) +  biases['out']

def eval_thydys(x):
    return np.array(list(tf.map_fn(lambda x: 1 if x>=1.8 else 0,x)))


def main():
    # Option Parser
    if (len(sys.argv) <= 1):
        print("train.py -h or --help to get guideline of input options")
        exit()
    use = "Usage: %prog [options] filename"
    parser = OptionParser(usage = use)
    parser.add_option("-d", "--input-dir", dest="input_dir", action="store", type="string", help="input data dir")
    parser.add_option("-o", "--output-dir", dest="ckpt_dir", action="store", type="string", help="ckpt data dir")
    parser.add_option("-t", "--timesteps", dest="timesteps", action="store", type="int", help="timesteps")
    parser.add_option("-n", "--num-input", dest="num_input", action="store", type="int", help="number of input (input vector's width)")

    (options, args) = parser.parse_args()
    input_dir = options.input_dir
    timesteps = options.timesteps
    num_input = options.num_input
    ckpt_dir = options.ckpt_dir
    len_status = 3

    X = np.fromfile(input_dir + '/X.dat', dtype=float)
    cardinality = int(X.shape[0]/(timesteps * num_input + len_status+1))
    X = X.reshape([cardinality, timesteps*num_input + len_status+1])
    Y = np.fromfile(input_dir + '/Y.dat', dtype=float)
    train_x, val_x, test_x, train_y, val_y, test_y = utl.train_val_test_split(X, Y, split_frac=0.80)
    #print("Data Set Size")
    #print("Train set: \t\t{}".format(train_x.shape), 
    #      "\nValidation set: \t{}".format(val_x.shape),
    #      "\nTest set: \t\t{}".format(test_x.shape))
    
    
    # In[ ]:
    
    
    # Training Parameters
    learning_rate = 0.001
    epochs =500 
    batch_size = 60
    #display_step = 200
    
    # Network Parameters
    #num_input = 2 
    #timesteps = 480 
    num_hidden = 4096
    num_classes = 1
   
    print("### Network Parameters ###")
    print("Learning Rate: {}".format(learning_rate))
    print("Batch Size: {}".format(batch_size))
    print("Size of Hidden Layer: {}".format(num_hidden))
    print("Timestep: {}".format(timesteps)) 
    print("------------------")
    X_ = tf.placeholder("float", [None, timesteps, num_input])
    X_status = tf.placeholder("float", [None, len_status])

    Y_ = tf.placeholder("float", [None, num_classes])
    lr = tf.placeholder("float")
    
    weights = {
        'out':tf.Variable(tf.random_normal([num_hidden,num_classes])),
        'status': tf.Variable(tf.random_normal([len_status,num_classes]))
    }
    biases = {
        'out':tf.Variable(tf.random_normal([num_classes]))
    }
    seq_embed = RNN(X_, weights, biases, timesteps, num_hidden)
    prediction = seq_embed + tf.matmul(X_status, weights['status']) 
    
    loss_op = tf.losses.mean_squared_error(Y_, prediction)
    #optimizer = tf.train.AdadeltaOptimizer(lr).minimize(loss_op)
    #optimizer = tf.train.AdamOptimizer(lr).minimize(loss_op)
    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss_op)
     
    correct_pred = tf.equal(tf.cast( (prediction/1.8) - tf.round(prediction/1.8), tf.float32), tf.cast( (prediction/1.8)-tf.round(Y_/1.8), tf.float32))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
    
        # Run the initializer
        sess.run(init)
        saver = tf.train.Saver()

        n_batches = len(train_x)//batch_size
        
        for e in range(epochs):
            if (epochs%30 == 0):
                learning_rate = learning_rate*0.98
            train_acc = []
            for ii, (x, y) in enumerate(utl.get_batches(train_x, train_y, batch_size), 1):
                x_seq = x[:, :timesteps*num_input]
                x_status = x[:, timesteps*num_input:timesteps*num_input+len_status]
                x_status = utl.norm_status(x_status, [0,1])
                x_seq = x_seq.reshape((batch_size, timesteps, num_input))
           
                feed = {X_: x_seq, Y_: y[:, None], X_status: x_status, lr:learning_rate}
                loss, acc, _ = sess.run([loss_op, accuracy, optimizer], feed_dict=feed)
                train_acc.append(acc)
    
                if (ii+1) % n_batches == 0:
                    val_acc = []
                    for xx, yy in utl.get_batches(val_x, val_y, batch_size):
                        xx_seq = xx[:, :timesteps*num_input]
                        xx_status = xx[:, timesteps*num_input:timesteps*num_input+len_status]
                        xx_status = utl.norm_status(xx_status, [0,1])
                        xx_seq = xx_seq.reshape((batch_size, timesteps, num_input))

                        feed = {X_:xx_seq, Y_:yy[:,None], X_status:xx_status, lr:learning_rate}
                        val_batch_loss = sess.run([loss_op], feed_dict=feed)
                        val_acc.append(val_batch_loss)
    
                    print("Epoch: {}/{}...".format(e+1, epochs),
                          "Batch: {}/{}...".format(ii+1, n_batches),
                          "Train Loss: {:.3f}...".format(loss),
                          #"Train Accruacy: {:.3f}...".format(np.mean(train_acc)),
                          "Val Loss: {:.3f}".format(np.mean(val_acc)))
        
        test_data = test_x.reshape((-1, timesteps*num_input+len_status+1))
        test_x = test_data[:, :timesteps*num_input]
        test_xs = test_data[:, timesteps*num_input:timesteps*num_input+len_status]
        test_xs = utl.norm_status(test_xs, [0,1])
        test_x = test_x.reshape((-1, timesteps, num_input))
        test_label = test_y
        print("Testing Loss:", sess.run(loss_op, feed_dict={X_: test_x, Y_: test_label[:, None], X_status: test_xs, lr:learning_rate}))
        
        # Model Checkpoint
        saver.save(sess, ckpt_dir) 
if __name__ == "__main__": 
    main()
