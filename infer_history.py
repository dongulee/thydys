# coding: utf-8

'''
# # Training of Thyroid dysfunction prediction model w/ LSTM network
# 
# * Using pre-processed data from dat files (X.dat, Y.dat)
data description
X: 
- sequence of HR and acitvity
- historical data
  + previous sequence and freeT4 value
  + another ones
Y:
- free T4

'''


import os
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

#def var_RNN(x, x_len, weights, biases, timesteps, num_hidden): #FIXME: dynamic
def var_RNN(x, weights, biases, timesteps, num_hidden):

    
    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1) #FIXME

    # Define a lstm cell with tensorflow
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    # Get lstm cell output
    #outputs, states = tf.contrib.rnn.dynamic_rnn(lstm_cell, x, dtype=tf.float32, sequence_length=x_len) #FIXME: dynamic
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    # Y (freeT4) = W_x * o_T (final output from LSTM) + W_status * S + biases
    return outputs, states 

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
    #parser.add_option("-o", "--output-dir", dest="ckpt_dir", action="store", type="string", help="ckpt data dir")
    parser.add_option("-t", "--timesteps", dest="timesteps", action="store", type="int", help="timesteps")
    parser.add_option("-n", "--num-input", dest="num_input", action="store", type="int", help="number of input (input vector's width)")
    parser.add_option("-c", "--ckpt-dir", dest="ckpt_dir", action="store", type="string", help="directory of checkpoint")

    (options, args) = parser.parse_args()
    input_dir = options.input_dir
    timesteps = options.timesteps
    num_input = options.num_input
    ckpt_dir = options.ckpt_dir
    len_status = 4
    # Training Parameters
    learning_rate = 0.001
    epochs =800 
    batch_size = 40
    #display_step = 200
    
    # Network Parameters
    #num_input = 2 
    #timesteps = 480 
    num_hidden =2048 
    num_classes = 4
    max_length = timesteps * num_classes

    X = np.fromfile(input_dir + '/X.dat', dtype=float) #padded sequence N x 7,680
    cardinality = int(X.shape[0]/(timesteps * num_input *num_classes))
    X = X.reshape([cardinality, max_length * num_input])

    #X_len = np.fromfile(input_dir + '/X_len.dat', dtype=float) #FIXME: dynamic
    #X_len = X_len.reshape([cardinality, 1])

    Y = np.fromfile(input_dir + '/Y.dat', dtype=float)
    Y = Y.reshape([cardinality, num_classes])

    train_x, val_x, test_x, train_y, val_y, test_y = utl.train_val_test_split(X, Y, split_frac=0.80)
    #train_x, val_x, test_x, train_x_len, val_x_len, test_x_len, train_y, val_y, test_y = utl.train_val_test_split2(X, X_len, Y, split_frac=0.80)
    #print("Data Set Size")
    #print("Train set: \t\t{}".format(train_x.shape), 
    #      "\nValidation set: \t{}".format(val_x.shape),
    #      "\nTest set: \t\t{}".format(test_x.shape))
    
    
    # In[ ]:
    
    


    print("### Network Parameters ###")
    print("Learning Rate: {}".format(learning_rate))
    print("Batch Size: {}".format(batch_size))
    print("Size of Hidden Layer: {}".format(num_hidden))
    print("Timestep: {}".format(timesteps)) 
    print("------------------")
    X_ = tf.placeholder("float", [None, max_length, num_input])
    
    #X_len_ = tf.placeholder("float", [None, 1]) #FIXME: dynamic 

    Y_ = tf.placeholder("float", [None, num_classes])
    lr = tf.placeholder("float")
    
    weights = {
        'out1':tf.Variable(tf.random_normal([num_hidden,1])),
        'out2':tf.Variable(tf.random_normal([num_hidden,1])),
        'out3':tf.Variable(tf.random_normal([num_hidden,1])),
        'out4':tf.Variable(tf.random_normal([num_hidden,1]))
    }
    biases = {
        'out1':tf.Variable(tf.random_normal([1])),
        'out2':tf.Variable(tf.random_normal([1])),
        'out3':tf.Variable(tf.random_normal([1])),
        'out4':tf.Variable(tf.random_normal([1]))
    }
    #LSTM_out, LSTM_states = var_RNN(X_, X_len_, weights, biases, timesteps, num_hidden) #FIXME: dynamic
    LSTM_out, LSTM_states = var_RNN(X_, weights, biases, max_length, num_hidden)
    prediction = []
    prediction.append(tf.matmul(LSTM_out[timesteps*1], weights['out1']) + biases['out1'])
    prediction.append(tf.matmul(LSTM_out[timesteps*1], weights['out2']) + biases['out2'])
    prediction.append(tf.matmul(LSTM_out[timesteps*1], weights['out3']) + biases['out3'])
    prediction.append(tf.matmul(LSTM_out[timesteps*1], weights['out4']) + biases['out4'])
    #prediction = seq_embed + tf.matmul(X_status, weights['status']) 
    prediction = tf.reshape(tf.concat(prediction, 1), [-1, 4]) 
    loss_op = tf.losses.mean_squared_error(Y_, prediction)
    #optimizer = tf.train.AdadeltaOptimizer(lr).minimize(loss_op)
    #optimizer = tf.train.AdamOptimizer(lr).minimize(loss_op)
    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss_op)
     
    #correct_pred = tf.equal(tf.cast( (prediction/1.8) - tf.round(prediction/1.8), tf.float32), tf.cast( (prediction/1.8)-tf.round(Y_/1.8), tf.float32))
    #accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    # Restore the ckpt
    SAVER_DIR = options.ckpt_dir 
    saver = tf.train.Saver()
    checkpoint_path = os.path.join(SAVER_DIR, SAVER_DIR)
    ckpt = tf.train.get_checkpoint_state(SAVER_DIR)
    
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
    
        # Run the initializer
        sess.run(init)
        saver.restore(sess, ckpt.model_checkpoint_path)
        
        #Y=Y.reshape([4,-1,1])
        X = X.reshape([-1, max_length, 2])
        pred = np.array(prediction.eval(feed_dict = {X_:X, Y_:Y}))
        
        pred_diagnosis = [1 if x[3]>=1.8 else 0 for x in list(pred)]
        y_diagnosis = [1 if x[3]>=1.8 else 0 for x in list(Y)]
        evaluation = np.equal(pred_diagnosis, y_diagnosis)
        print(np.mean(evaluation))
        f = open(SAVER_DIR+'result.txt', 'w')
        for i in range(0, len(Y)):
            f.write(str(pred[i]) + ', ' + str(Y[i])+'\n')
        f2 = open(SAVER_DIR+'result_diagnosis.txt', 'w')
        for i in range(0, len(Y)):
            f2.write(str(pred_diagnosis[i]) + ', ' + str(y_diagnosis[i])+'\n')
        f2.close()
        f.close()

if __name__ == "__main__": 
    main()
