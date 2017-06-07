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

from dataset import Dataset
from preprocess import *

def get_predictor(args):
    if args.mode == 'single':
        model = load_model(args.ckpt)
        def f(x):
            return model.predict_on_batch(x)
    elif args.mode == 'dual':
        ckpts = args.ckpt.split(',')
        model_gender = load_model(ckpts[0])
        model_age = load_model(ckpts[1])
        def f(x):
            g = model_gender.predict_on_batch(x).argmax(axis=1)
            a = model_age.predict_on_batch(x).argmax(axis=1)
            return g * 4 + a
    return f

def main(args): 
    assert args.ckpt != None

    logging.info('Load from %s' % args.ckpt)
    
    # Config tf session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    # Load model and setup prediction function
    predictor = get_predictor(args)        

    # Test
    scale = 1.0 / 255.0
    mean = np.load(args.mean).astype(np.float32)
    dataset = Dataset(label_type=args.label, size=args.size, crop=bool(args.crop), color=args.color)

    logging.info('Load data from %s' % args.X)
    for filename, x in dataset(args.X):
        print filename, predictor(x[np.newaxis,:])
    '''
    # Postprocess
    T = T.argmax(axis=1)
    Y = Y.argmax(axis=1)
    
    print Y.shape, T.shape
    print float(np.equal(Y, T).sum()) / float(len(T))
    '''

if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', help='dual/single', 
            choices=['dual', 'single'], type=str, required=True) 
    parser.add_argument('--X', help='X', type=str, required=True)
    parser.add_argument('--mean', help='mean', type=str, required=True)
    parser.add_argument('--ckpt', help='ckpt', type=str, required=True)
    parser.add_argument('--label', help='label type', 
            choices=['gender', 'age', 'mix'], type=str, required=True) 
    parser.add_argument('--size', help='size', type=int, default=80)
    parser.add_argument('--crop', help='crop face', type=int, default=1)
    parser.add_argument('--color', help='color mode', type=str, default='rgb')    

    args = parser.parse_args()
    main(args)
 
