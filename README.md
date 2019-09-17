# Yelp-Recommender
Recommender system for Yelp users based on features such as previous reviews

This repository aimed to build a recommender system using hybrid models for Yelp's dataset. The dataset was supplied by [Yelp](https://www.yelp.com/dataset/challenge). By using content-based and collaboritive filtering, we were able to output a list of 10 restaurants for a test user in the city of Las Vegas. Feel free to clone this repository to your own machine and improve on our methods.

Take a look at our blog for more details: [Hangry? Not Anymore.](https://nycdatascience.com/blog/student-works/hangry-not-anymore/)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Installing

In order to run the ipynb files, make sure to have the following libraries:
* [Jupyter](https://jupyter.readthedocs.io/en/latest/install.html)
* [Numpy](https://docs.scipy.org/doc/numpy/user/install.html)
* [Matplotlib](https://matplotlib.org/)
* [Scikit-Learn](https://scikit-learn.org/stable/)
* [Keras](https://keras.io/#installation)
* [Tensorflow](https://www.tensorflow.org/install/)

Then, clone this repository into your local machine and try manipulating our tools yourself! The dataset can be downloaded using this [link](https://www.yelp.com/dataset/challenge).

```
git clone https://github.com/wonchankim97/Yelp-Recommender
```

## Running the recommender

After downloading the dataset, clean the raw dataset using the clean.py file. Then, run the recommender by using the recommend.py file.

```
python clean.py
python recommend.py
```

You can try out different users within the dataset in order to get the most relevant results. Otherwise, the model will output the most popular, highly rated results.

## Authors

* **Hyelee Lee** - [Hyelee](https://github.com/hayley01145553)
* **Kisoo Cho** - [Necronia](https://github.com/necronia)
* **Wonchan Kim** - [Wonchan Kim](https://github.com/wonchankim97)
