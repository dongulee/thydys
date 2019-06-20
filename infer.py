import tensorflow as tf
import numpy as np
import sys
from optparse import OptionParser
import os
import utils as utl

def RNN(x, weights, biases, timesteps, num_hidden):

    
    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    # Get lstm cell output
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


def main():
    if (len(sys.argv) <= 1):
        print("infer.py -h or --help to get guideline of input options")
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
    #ckpt_dir = options.ckpt_dir

    X = np.fromfile(input_dir + '/X.dat', dtype=float)
    cardinality = int(X.shape[0]/(timesteps * num_input))
    X = X.reshape([cardinality, timesteps, num_input])
    Y = np.fromfile(input_dir + '/Y.dat', dtype=float)
    

    train_x, val_x, test_x, train_y, val_y, test_y = utl.train_val_test_split(X, Y, split_frac=0.80)
     
    # Training Parameters
    learning_rate = 0.001
    epochs =800 
    batch_size = 40
    #display_step = 200
    
    # Network Parameters
    #num_input = 2 
    #timesteps = 480 
    num_hidden = 2048 
    num_classes = 1
   
    print("### Network Parameters ###")
    print("Learning Rate: {}".format(learning_rate))
    print("Batch Size: {}".format(batch_size))
    print("Size of Hidden Layer: {}".format(num_hidden))
    print("Timestep: {}".format(timesteps)) 
    print("------------------")
    X_ = tf.placeholder("float", [None, timesteps, num_input])
    Y_ = tf.placeholder("float", [None, num_classes])
    lr = tf.placeholder("float")
    
    weights = {
        'out':tf.Variable(tf.random_normal([num_hidden,num_classes]))
    }
    biases = {
        'out':tf.Variable(tf.random_normal([num_classes]))
    }
    prediction = RNN(X_, weights, biases, timesteps, num_hidden)
    
    loss_op = tf.losses.mean_squared_error(Y_, prediction)
    #optimizer = tf.train.AdadeltaOptimizer(lr).minimize(loss_op)
    #optimizer = tf.train.AdamOptimizer(lr).minimize(loss_op)
    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss_op)
     
    correct_pred = tf.equal(tf.cast( (prediction/1.8) - tf.round(prediction/1.8), tf.float32), tf.cast( (prediction/1.8)-tf.round(Y_/1.8), tf.float32))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    # Restore the ckpt
    SAVER_DIR = options.ckpt_dir 
    saver = tf.train.Saver()
    checkpoint_path = os.path.join(SAVER_DIR, SAVER_DIR)
    ckpt = tf.train.get_checkpoint_state(SAVER_DIR)
    
    
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        #new_saver = tf.train.import_meta_graph('ckpt.meta')
        saver.restore(sess, ckpt.model_checkpoint_path) 
        print("loss test: %f" % loss_op.eval(feed_dict = {X_:test_x, Y_:test_y[:, None]}))
        pred = np.array(prediction.eval(feed_dict = {X_:X, Y_:Y[:, None]}))
        
        pred_diagnosis = [1 if x[0]>=1.8 else 0 for x in list(pred)]
        y_diagnosis = [1 if x>=1.8 else 0 for x in list(Y)]
        evaluation = np.equal(pred_diagnosis, y_diagnosis)
        print(np.mean(evaluation))
        f = open('result.txt', 'w')
        for i in range(0, len(Y)):
            f.write(str(pred[i][0]) + ', ' + str(Y[i])+'\n')
        f2 = open('result_diagnosis.txt', 'w')
        for i in range(0, len(Y)):
            f2.write(str(pred_diagnosis[i]) + ', ' + str(y_diagnosis[i])+'\n')
        f2.close()
        f.close()

#        sess.close()


if __name__=="__main__":
    main()
