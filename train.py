#following the tuorial found in page    http://dataaspirant.com/2017/05/03/handwritten-digits-recognition-tensorflow-python/

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("model_data/", one_hot=True) #Download the MNIS
import tensorflow as tf #Import Tensorflow

#Initializing parameters for the model
BATCH = 100
LEARNING_RATE = 0.01
TRAINING_EPOCHS = 30


#Creating Placeholders
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# creating variables
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# initializing the model
test = tf.matmul(x, W) + b
y = tf.nn.softmax(test)


def train(filename):
    

    # Defining Cost Function
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

   