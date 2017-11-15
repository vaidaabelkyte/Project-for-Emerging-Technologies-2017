#following the tuorial found in page    http://dataaspirant.com/2017/05/03/handwritten-digits-recognition-tensorflow-python/

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("model_data/", one_hot=True)  #Download the MNIS
