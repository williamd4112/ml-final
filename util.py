import cv2
import numpy as np
import keras
import csv

import sys, os
from tqdm import *

import argparse
import logging

import tensorflow as tf

from keras.models import load_model
from keras.models import Model
from keras import backend as K
from keras.backend.tensorflow_backend import set_session

from preprocess import *

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def get_predictor(mode, ckpt):
    if mode == 'single':
        model = load_model(ckpt)
        def f(x):
            return model.predict(x).argmax(axis=1)
    elif mode == 'dual':
        ckpts = ckpt.split(',')
        model_gender = load_model(ckpts[0])
        model_age = load_model(ckpts[1])
        def f(x):
            g = model_gender.predict(x).argmax(axis=1)
            a = model_age.predict(x).argmax(axis=1)
            return g * 4 + a
    return f

