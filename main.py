import cv2
import numpy as np
import keras
import csv

import sys, os
from tqdm import *

import argparse
import logging
from model import VGG16Model


NUM_CLASSES = 8

def load_train_dataset(x_path, t_path):
    return np.load(x_path), keras.utils.to_categorical(np.load(t_path), NUM_CLASSES)

def main(args):
    if args.task == 'train':
        model = VGG16Model()
        X, T = load_train_dataset(args.X, args.T)
        logging.info('X = %s; T = %s' % (X.shape, T.shape))
        model.train(X, T)

if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', help='train/test/validate', 
            choices=['train', 'test', 'validate'], type=str, default='train') 
    parser.add_argument('--X', help='X', type=str, default='data/X.npy')
    parser.add_argument('--T', help='T', type=str, default='data/T.npy')
    args = parser.parse_args()
    main(args)
 
