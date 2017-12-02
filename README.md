<h1>Project-for-Emerging-Technologies-2017</h1>
<h3><i>Flask framework web application in Python to recognise digits in images using Tensorflow</i></h3>
<h4>Emerging Technologies Module<br>
Student - Vaida Abelkyte<br>
Year - 4</h4>
<hr/>


<h3><i>Project Overview</h3>
<p>
In this project I have created a web application in Python to recognise digits in images. User can visit the web 
application through their browser (instructions below - How to run the application).<br> User can draw an image containing a single digit 
in the canvas, and the web application will respond with the digit contained in the image. 
I used tensorflow and flask to do the project.
</p>

<h3><i>Technologies</h3>
<p>
- <a href="http://flask.pocoo.org/">Flask microframework for Python</a><br>
  - <a href="https://www.python.org/">Python programming language</a><br>
    - <a href="https://www.tensorflow.org/">TensorFlow - Software library for machine learning</a><br>
      - <a href="https://getbootstrap.com/">Bootstrap - front-end component library</a><br>
</p>

<h3><i>How to run the application</h3>
<p>
The app can be run locally :<br>
$ python digits.py<br>
  For the first time the user must first run: <br>
$ python train.py   ---->   to Download the <a href="http://yann.lecun.com/exdb/mnist/">MNIST</a> dataset and train the model<br>
$ python digits.py   -----> to star the app.

</p>

<h3><i>Screenshot of the application</h3>

![digit](https://user-images.githubusercontent.com/15648433/33519729-e14cf5f8-d7a4-11e7-9936-7f1b75597ce2.png)


<br>
<h5>References</h5>
<h6>
To accomplish the task I was following the  <a href="https://www.tensorflow.org/get_started/mnist/beginners">MNIST For ML Beginners tutorial by TensorFlow</a>  <br>
And also I've used an article -  <a href="http://dataaspirant.com/2017/05/03/handwritten-digits-recognition-tensorflow-python/"> "Handwritten digits recognition using google tensorflow with python" </a> written by data scientist  <a href="http://dataaspirant.com/about/"> Saimadhu Polamuri</a>, to come up with learning process and evaluate correctness of the model.

</h6>

