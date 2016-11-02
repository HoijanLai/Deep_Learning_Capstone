
##############################################################################
#                                                                            #
#This script implement the same model as the cnn_mat.ipynb, but with summary #
#                                                                            #
##############################################################################

from __future__ import print_function
from six.moves import cPickle as pickle
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import display_center
import numpy as np

imsize = 32
num_labels = 10
num_channels = 1


#===================#
# unpickle the file #
#===================#
def unpickle(file_name):
    '''
    unpickle pickle file
    '''
    print("loading %s" %file_name)
    try:
        with open(file_name, "rb") as f:
            data = pickle.load(f)
            print("SUCCESS!")
            return data['X'].astype(np.float32), data['y'].astype(np.float32)
    except Exception as e:
        print("FAILED: ", e)
                
train_X, train_y = unpickle("train.pickle")
print(train_X.shape, train_y.shape)
val_X, val_y = unpickle("validation.pickle")
print(val_X.shape, val_y.shape)
test_X, test_y = unpickle("test.pickle")
print(test_X.shape, test_y.shape)


#===============#
# channel the X #
#===============#
train_X = train_X.reshape([-1, imsize, imsize, 1])
test_X = test_X.reshape([-1, imsize, imsize, 1])
val_X = val_X.reshape([-1, imsize, imsize, 1])
print(train_X.shape)
print(test_X.shape)
print(val_X.shape)

#===========#
# ohe the y #
#===========#
def ohe(y):
    y = y.reshape([-1, 1])
    return (np.arange(num_labels) == y).astype(np.float32)
    
train_y = ohe(train_y)
test_y = ohe(test_y)
val_y = ohe(val_y)


#===================#
# NODES DEFINITIONS #
#===================#

## fully connected
def fc_layer(ts, w, b, name, record = True, activation = tf.nn.relu):
    with tf.name_scope(name):
        with tf.name_scope('Weights'):
            var_summary(w, name + '/Weights') 
        with tf.name_scope('Biases'):
            var_summary(b, name + '/Biases')
        with tf.name_scope('Pre_Actv'):
            pre_actv = tf.matmul(ts, w) + b
            histo_summary(pre_actv, name + '/Pre_Actv')
        with tf.name_scope('Actv'): 
            actv = activation(pre_actv, name = 'Activation')
            histo_summary(actv, name + '/Actv')
        return actv
    
## convolutional
def conv_layer(img, f, b, name, stride = [1, 2, 2, 1], activation = tf.nn.relu):
    with tf.name_scope(name):
        with tf.name_scope('Raw_Conv'):
            pre_actv = tf.nn.conv2d(img, f, stride, padding = 'SAME') + b
            img_summary(pre_actv, name + '/Raw_Convs')
        with tf.name_scope('Actv_Conv'):
            actv = activation(pre_actv, name = 'Actv_Conv')
            img_summary(actv, name + '/Actv_Convs')
        return actv
        

## dropout 
def dropout(ts, name, keep_prob = 0.5):
   with tf.name_scope(name):
        tf.scalar_summary('Keep_Prob/' + name, keep_prob)
        dropped = tf.nn.dropout(ts, keep_prob)
        return dropped
   
## max_pooling
def max_pool(ts, name, kernel = [1, 2, 2, 1], stride = [1, 2, 2, 1]):
    with tf.name_scope(name):
        pool = tf.nn.max_pool(ts, kernel, stride, padding = 'SAME')
        img_summary(pool, name + 'Max_Pool')
    return pool

    


#====================#
# small init pieces  #
#====================#

def img_summary(ts, name):
    # image summary grayscale  
    with tf.name_scope('Summaries'):
        pic = tf.reduce_mean(ts, 3)
        shape = pic.get_shape().as_list()+[1]
        tf.image_summary(name, tf.reshape(pic, shape)) 

def histo_summary(ts, name):
    with tf.name_scope('Summaries'):
        tf.histogram_summary(name, ts)

def var_summary(ts, name):
    # statistic for a tensor, this is created to support tensorboard actions
    with tf.name_scope('Summaries'):
        mean = tf.reduce_mean(ts)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(ts - mean)))
        tf.scalar_summary('Mean/' + name, mean)
        tf.scalar_summary('Stddev/' + name, stddev)
        tf.scalar_summary('Min/' + name, tf.reduce_min(ts))
        tf.scalar_summary('Max/' + name, tf.reduce_max(ts))
    histo_summary(ts, name)

def accuracy(pred, y, tf_style = False):
    if tf_style: 
        return 100.0 * tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)), tf.float32))
    return 100.0 * np.sum(np.argmax(y, axis = 1) == np.argmax(pred, axis = 1))/float(y.shape[0])


#==========================================#
# next is to build the computational graph #
#==========================================#
batch_size = 256
patch_size = [5, 5, 3]
depth = [32, 64, 256]
num_hiddens = 128
graph = tf.Graph()

with graph.as_default():

    ## the Input placeholder to feed
    with tf.name_scope('Training_Set'):
        net_train_X = tf.placeholder(tf.float32, shape = (batch_size, imsize, imsize, num_channels))
        tf.image_summary('Batch_Input', net_train_X)
        
        net_train_y = tf.placeholder(tf.float32, shape = (batch_size, num_labels))
        tf.histogram_summary('Batch_Label', net_train_y)
    
    with tf.name_scope('Validation_Set'):    
        net_val_X = tf.constant(val_X)
        net_val_y = tf.constant(val_y)
    
    with tf.name_scope('Test_Set'):
        net_test_X = tf.constant(test_X)
        
    ## Variables
    with tf.name_scope('Variables'):  
        f1 = tf.Variable(tf.truncated_normal([patch_size[0], patch_size[0], num_channels, depth[0]], stddev = 0.01), name = 'f1')
        conv_b1 = tf.Variable(tf.zeros([depth[0]]), name = 'conv_b1')
        f2 = tf.Variable(tf.truncated_normal([patch_size[1], patch_size[1], depth[0], depth[1]], stddev = 0.05), name = 'f2')
        conv_b2 = tf.Variable(tf.zeros([depth[1]]), name = 'conv_b2')
        f3 = tf.Variable(tf.truncated_normal([patch_size[2], patch_size[2], depth[1], depth[2]], stddev = 0.07), name = 'f3')
        conv_b3 = tf.Variable(tf.zeros([depth[2]]), name = 'conv_b3')
        w1 = tf.Variable(tf.truncated_normal([imsize//16*imsize//16*depth[-1], num_hiddens], stddev = 0.1), name = 'w1')
        b1 = tf.Variable(tf.constant(1.0, shape = [num_hiddens]), name = 'b1')
        w2 = tf.Variable(tf.truncated_normal([num_hiddens, num_labels], stddev = 0.2), name = 'w2')
        b2 = tf.Variable(tf.constant(1.0, shape = [num_labels]), name = 'b2')  
             
    ## conv - conv - drop - conv - pool - fc - drop - fc
    def model(data, name = '', drop = False):
        conv = conv_layer(data, f1, conv_b1, name + 'Conv_1', stride = [1, 2, 2, 1])
        conv = conv_layer(conv, f2, conv_b2, name + 'Conv_2', stride = [1, 2, 2, 1])
        dropped = dropout(conv, name + 'Dropout_3', keep_prob = .5) if drop else conv
        conv = conv_layer(dropped, f3, conv_b3, name + 'Conv_4', stride = [1, 2, 2, 1])
        pool = max_pool(conv, name + 'MaxPool_5')
        ## fc - drop -fc
        with tf.name_scope(name + 'Flatten'):
            shape = pool.get_shape().as_list()
            reshape = tf.reshape(pool, [shape[0], shape[1]*shape[2]*shape[3]])
        dropped = dropout(reshape, name + 'Dropout_6', keep_prob = .5) if drop else reshape
        fc = fc_layer(dropped, w1, b1, name + 'FC_7')
        fc = fc_layer(fc, w2, b2, name + 'FC_8', activation = tf.identity)
        return fc
    
    with tf.name_scope('Training'):
        with tf.name_scope('Model'):
            logits = model(net_train_X, drop = True)
        with tf.name_scope('Cross_entropy'):
            diff = tf.nn.softmax_cross_entropy_with_logits(logits, net_train_y)
            with tf.name_scope('Total'):
                loss = tf.reduce_mean(diff)
                tf.scalar_summary('Cross_entropy', loss)    
        with tf.name_scope('Learning_Rate'):    
            global_step = tf.Variable(0, name = 'global_step')
            learning_rate = tf.train.exponential_decay(1e-3, global_step, 500, .92, staircase=True, name = 'learning_rate')
            tf.scalar_summary('lerning_Rate', learning_rate)        
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step = global_step)
    
   
    with tf.name_scope('Train'):
        with tf.name_scope('Model'):
            train_pred = tf.nn.softmax(model(net_train_X, 'Train/'))
        with tf.name_scope('Accuracy'):
            tf.scalar_summary('Accuracy', accuracy(train_pred, net_train_y, True)) 
        
    with tf.name_scope('Validation'):
        with tf.name_scope('Model'):
            val_pred = tf.nn.softmax(model(net_val_X, 'Validation/'))
        with tf.name_scope('Accuracy'):
            tf.scalar_summary('Val_Accuracy', accuracy(val_pred, net_val_y, True))
        
    with tf.name_scope('Test'):
        test_pred = tf.nn.softmax(model(net_test_X, 'Test/')) 
        
    merged = tf.merge_all_summaries() 
    


    
   
    
    
train_writer = tf.train.SummaryWriter('board_summary/val', graph)    
#val_writer = tf.train.SummaryWriter('board_summary/val')

num_steps = 30001
with tf.Session(graph = graph) as session:
    tf.initialize_all_variables().run()
    print('Initialize') 
    for step in range(num_steps):
        offset = (step * batch_size) % (train_y.shape[0] - batch_size)
        batch_X = train_X[offset:(offset + batch_size), :, :, :]
        batch_y = train_y[offset:(offset + batch_size)]
        feed_dict = {net_train_X : batch_X, net_train_y : batch_y}
        
        if step % 100 == 0:
            summary, _, stage_loss, pred, valpred = session.run(\
                    [merged, optimizer, loss, train_pred, val_pred], feed_dict=feed_dict)
            train_writer.add_summary(summary, step)
            if step % 500 == 0:
                print('Minibatch loss at step %d: %f' % (step, stage_loss))
                print('Minibatch accuracy: %.2f%%' % accuracy(pred, batch_y))
                print('Validation accuracy: %.2f%%' % accuracy(valpred, val_y)) 
        else:
            session.run(optimizer, feed_dict=feed_dict)
        
     
    print('Test accuracy: %.1f%%' % accuracy(test_pred.eval(), test_y))



























        
        

