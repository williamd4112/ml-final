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

from dataset import Dataset
from preprocess import pre_process, post_process
from util import get_predictor, get_session

def main(args): 
    assert args.ckpt != None

    logging.info('Load from %s' % args.ckpt)
    
    # Config tf session
    sess = get_session()
    set_session(sess)    

    # Load model and setup prediction function
    predictor = get_predictor(args.mode, args.ckpt)        

    # Test
    scale = 1.0 / 255.0
    mean = np.load(args.mean).astype(np.float32)
    dataset = Dataset(label_type=args.label, size=args.size, crop=bool(args.crop), color=args.color)

    logging.info('Load data from %s' % args.X)
    for filename, x in dataset(args.X):
        index = filename
        x = x.astype(np.float32)
        x = pre_process(x, mean, scale)
        label = predictor(x[np.newaxis,:])[0]
        print '%s,%d' % (index, label)

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
 
