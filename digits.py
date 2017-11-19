import os
import tensorflow as tf
import train
import numpy as np
from flask import Flask, send_file, request


app = Flask(__name__)
# model path
MODEL = './model_data/model/'
# variable for storing loaded model
session = None

#using variable from train process to process input image
# y = Wx + b
func = tf.argmax(train.y, 1)


def recognize_image(image):
    # reshape image vector to vector-column
    image = image.reshape((1, 784))
    result = session.run([func], feed_dict={train.x: image})
    return result[0][0]


def check_model():
    
   
    
    if not os.path.exists(MODEL):      #Checks is model exists and restores session from save
        train.train(MODEL)

    session = tf.Session()
    oSaver = tf.train.Saver()
    oSaver.restore(session, MODEL)
    return session


@app.route('/')
def hello_world():
   
    return send_file('index.html')


@app.route('/digit', methods=['POST'])    #post-data: image array of length 784 with elements [0.0 - 1.0]
def digit():
   
  
  
    data = np.array(request.json, dtype=np.float32)
    res = recognize_image(data)        #result: int - recognized digit
    return str(res)


if __name__ == '__main__':
    session = check_model()
    app.run()


    #some of the links where i looked for code
    #https://www.tensorflow.org/programmers_guide/saved_model
    #https://stackoverflow.com/questions/45270298/vectorize-reshape-and-normalise-my-image-to-image-like-mnist-train-images
    #https://stackoverflow.com/questions/37898795/tensorflow-accuracy-at-99-but-predictions-awful