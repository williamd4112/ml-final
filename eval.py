import cv2
import numpy as np
import keras
import csv

import sys, os
import argparse
import logging

import tensorflow as tf

from keras.models import load_model
from keras.models import Model
from keras import backend as K
from keras.backend.tensorflow_backend import set_session

from dataset import Dataset
from preprocess import pre_process, post_process
from util import get_predictor, get_session
from tqdm import *

def load_eval_dataset(x_path, t_path):
    X = np.load(x_path)
    T = np.load(t_path)
    #return X.astype(np.float32), T
    return X.astype(np.float32), keras.utils.to_categorical(T, T.max() + 1)

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
    scale = 1.0 / 255.0
    mean = np.load(args.mean).astype(np.float32)
    X = pre_process(X, mean, scale)

    # Get predictor
    predictor = get_predictor(args.mode, args.ckpt)
   
    # Predict
    Y = predictor(X)
    
    # Postprocess
    T = T.argmax(axis=1)
    
    print Y.shape, T.shape
    print float(np.equal(Y, T).sum()) / float(len(T))

if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', help='dual/single', 
            choices=['dual', 'single'], type=str, default='single') 
    parser.add_argument('--X', help='X', type=str, required=True)
    parser.add_argument('--T', help='T', type=str, required=True)
    parser.add_argument('--ckpt', help='ckpt', type=str, required=True)
    parser.add_argument('--mean', help='mean', type=str, required=True)
    parser.add_argument('--augment', help='augment', type=int, default=0)
    args = parser.parse_args()
    main(args)
 
