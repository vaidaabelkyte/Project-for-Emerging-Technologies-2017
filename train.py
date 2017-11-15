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
    mnist = input_data.read_data_sets("model_data/", one_hot=True) #Download the MNIS

    # Defining Cost Function
     #This is the cost function of the model – a cost function is a difference between the predicted value
    # and the actual value that we are trying to minimize to improve the accuracy of the model
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
  
    # Determining the accuracy of parameters
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Implementing Gradient Descent Algorithm
    train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)

    # Initializing the session
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        # Creating batches of data for epochs
        for epoch in range(TRAINING_EPOCHS):
            batch_count = int(mnist.train.num_examples / BATCH)
            for i in range(batch_count):
                # getting batch from dataset
                batch_x, batch_y = mnist.train.next_batch(BATCH)

                # Executing the model
                sess.run([train_op], feed_dict={x: batch_x, y_: batch_y})
                # Print accuracy of the model
            if epoch % 2 == 0:
                print("Epoch: ", epoch)
                print("Accuracy: ", accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
                print("Model Execution Complete")

                 # saving tensorflow model to file
        oSaver = tf.train.Saver()
        oSaver.save(sess, filename)


if __name__ == '__main__':
    train('./model_data/model/')


       