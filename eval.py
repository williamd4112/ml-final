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
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file

from keras import backend as K
from keras.backend.tensorflow_backend import set_session

from keras.preprocessing.image import ImageDataGenerator

from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape

def load_eval_dataset(x_path, t_path):
    X = np.load(x_path)
    T = np.load(t_path)
    return X, keras.utils.to_categorical(T, T.max() + 1)

def main(args):
    X, T = load_eval_dataset(args.X, args.T)
    
    assert args.ckpt != None

    logging.info('X = %s; T = %s' % (X.shape, T.shape))
    logging.info('Load from %s' % args.ckpt)
    
    # Config tf session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    # Preprocess
    X = preprocess_input(X.astype(np.float64))
    X = X * (1. / 255.)

    # Load model and predict
    if args.mode == 'single':
        model = load_model(args.ckpt)
    elif args.mode == 'dual':
        ckpts = args.ckpt.split(',')
        model_gender = load_model(ckpts[0])
        model_age = load_model(ckpts[1])
       
    # Postprocess
    T = T.argmax(axis=1)
    Y = Y.argmax(axis=1)
    
    print Y.shape, T.shape
    print float(np.equal(Y, T).sum()) / float(len(T))
  

if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', help='dual/single', 
            choices=['dual', 'single'], type=str, default='single') 
    parser.add_argument('--X', help='X', type=str, required=True)
    parser.add_argument('--T', help='T', type=str)
    parser.add_argument('--ckpt', help='ckpt', type=str)
    parser.add_argument('--augment', help='augment', type=int, default=0)
    args = parser.parse_args()
    main(args)
 
